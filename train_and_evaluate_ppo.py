"""Train and evaluate an isolated PPO policy on chronological crude trade logs."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crude_ppo_env import (
    CrudePPOEnv,
    load_trade_data,
    missing_feature_groups,
    prepare_trade_data,
)


def chronological_split(
    frame: pd.DataFrame, validation_fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split completed trades in chronological order; never shuffle the holdout."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if len(frame) < 2:
        raise ValueError("at least two completed trades are required for a holdout split")
    split_index = int(math.floor(len(frame) * (1.0 - validation_fraction)))
    split_index = min(max(split_index, 1), len(frame) - 1)
    return (
        frame.iloc[:split_index].reset_index(drop=True),
        frame.iloc[split_index:].reset_index(drop=True),
    )


def calculate_metrics(trades: list[dict[str, Any]]) -> dict[str, float | int]:
    """Calculate realized PnL metrics from executed, observed-side trades only."""
    pnls = np.asarray([float(trade["pnl"]) for trade in trades], dtype=float)
    if not len(pnls):
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "average_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    gains = float(pnls[pnls > 0].sum())
    losses = abs(float(pnls[pnls < 0].sum()))
    equity = np.concatenate(([0.0], np.cumsum(pnls)))
    drawdown = np.maximum.accumulate(equity) - equity
    return {
        "trades": int(len(pnls)),
        "wins": int((pnls > 0).sum()),
        "losses": int((pnls < 0).sum()),
        "win_rate": float((pnls > 0).mean()),
        "total_pnl": float(pnls.sum()),
        "average_pnl": float(pnls.mean()),
        "profit_factor": gains / losses if losses else (999.0 if gains else 0.0),
        "max_drawdown": float(drawdown.max()),
    }


def _make_vec_env(frame: pd.DataFrame):
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "PPO dependencies are missing. Install them with "
            "'pip install -r requirements-crude-ppo.txt'."
        ) from exc
    return DummyVecEnv([lambda: CrudePPOEnv(frame)])


def load_sheet1_trade_data() -> pd.DataFrame:
    """Load all populated Crude trades directly from Google Sheet 1."""
    from sheets_logger import get_gspread_client

    worksheet = get_gspread_client().open("Crude_Algo_Trade_Logs").sheet1
    values = worksheet.get_all_values()
    if not values:
        raise ValueError("Google Sheet 1 is empty")

    headers = [str(header).strip() for header in values[0]]
    required_headers = {"Symbol", "Action", "PnL", "Entry Timestamp"}
    missing_headers = sorted(required_headers - set(headers))
    if missing_headers:
        raise ValueError(f"Google Sheet 1 is missing required headers: {missing_headers}")

    symbol_index = headers.index("Symbol")
    rows = [
        (list(row) + [""] * len(headers))[:len(headers)]
        for row in values[1:]
        if any(str(cell).strip() for cell in row)
        and symbol_index < len(row)
        and "CRUDE" in str(row[symbol_index]).upper()
    ]
    if not rows:
        raise ValueError("Google Sheet 1 contains no populated Crude trade rows")

    frame = pd.DataFrame(rows, columns=headers, dtype=object)
    timestamps = frame["Entry Timestamp"].astype("string").str.strip()
    has_date = timestamps.str.match(
        r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})(?:$|[ T])",
        na=False,
    )
    parsed_timestamps = pd.to_datetime(
        timestamps.where(has_date),
        errors="coerce",
        dayfirst=True,
        format="mixed",
    )
    year_first = timestamps.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", na=False)
    parsed_year_first = pd.to_datetime(
        timestamps.where(year_first),
        errors="coerce",
        dayfirst=False,
        format="mixed",
    )
    parsed_timestamps.loc[year_first] = parsed_year_first.loc[year_first]
    invalid_timestamps = parsed_timestamps.isna()
    dropped_timestamp_rows = int(invalid_timestamps.sum())
    if dropped_timestamp_rows:
        warnings.warn(
            f"Dropping {dropped_timestamp_rows} Crude row(s) with empty or invalid Entry Timestamp.",
            stacklevel=2,
        )
    frame = frame.loc[~invalid_timestamps].copy()
    if frame.empty:
        raise ValueError("Google Sheet 1 contains no Crude rows with valid Entry Timestamp values")
    frame["Entry Timestamp"] = parsed_timestamps.loc[~invalid_timestamps]

    prepared = prepare_trade_data(frame)
    if prepared.empty:
        raise ValueError("Google Sheet 1 contains no valid completed Crude trades")
    return prepared


def evaluate_model(
    model: Any,
    normalization_path: Path,
    holdout: pd.DataFrame,
) -> dict[str, Any]:
    """Backtest deterministically on the untouched chronological holdout."""
    from stable_baselines3.common.vec_env import VecNormalize

    raw_env = _make_vec_env(holdout)
    env = VecNormalize.load(str(normalization_path), raw_env)
    env.training = False
    env.norm_reward = False
    observation = env.reset()

    executed: list[dict[str, Any]] = []
    skipped = 0
    unsupported = 0
    invalid_reentries = 0

    for expected_index in range(len(holdout)):
        episode_complete = False
        for _ in range(3):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, dones, infos = env.step(action)
            if not bool(dones[0]):
                continue

            info = infos[0]
            if int(info["row_index"]) != expected_index:
                raise RuntimeError(
                    "evaluation row order changed; refusing to report mismatched metrics"
                )
            if info.get("entered_trade"):
                executed.append(
                    {
                        "row_index": expected_index,
                        "side": info["recorded_side"],
                        "pnl": float(info["pnl"]),
                        "reward": float(info["reward"]),
                    }
                )
            else:
                skipped += int(bool(info.get("skipped_trade", False)))
                unsupported += int(bool(info.get("unsupported_counterfactual", False)))
                invalid_reentries += int(bool(info.get("invalid_reentry", False)))
            episode_complete = True
            break

        if not episode_complete:
            raise RuntimeError("policy did not close a replayed trade within three actions")

    env.close()
    return {
        "metrics": calculate_metrics(executed),
        "holdout_rows": len(holdout),
        "skipped_rows": skipped,
        "unsupported_counterfactual_entries": unsupported,
        "invalid_reentries": invalid_reentries,
        "executed_trades": executed,
        "interpretation": (
            "PnL and win rate use only trades where the model selected the side "
            "recorded in the historical log. Opposite-side outcomes are unknown."
        ),
    }


def _model_paths(model_base: Path) -> tuple[Path, Path]:
    return (
        Path(f"{model_base}.zip"),
        Path(f"{model_base}_vecnormalize.pkl"),
    )


def run_pipeline(
    data_path: Path | None,
    model_base: Path,
    total_timesteps: int,
    seed: int,
    mode: str,
    sheet1: bool = False,
) -> dict[str, Any]:
    if sheet1:
        frame = load_sheet1_trade_data()
        data_source = "Google Sheet: Crude_Algo_Trade_Logs / Sheet1"
    else:
        if data_path is None:
            raise ValueError("data_path is required unless sheet1=True")
        frame = load_trade_data(data_path)
        data_source = str(data_path)
    training, holdout = chronological_split(frame)
    if not frame.attrs.get("has_timestamp", False):
        warnings.warn(
            "No timestamp column found; preserving input row order for the 80/20 split.",
            stacklevel=2,
        )

    absent_features = missing_feature_groups(frame)
    if absent_features:
        warnings.warn(
            "State features absent from the dataset will be zero-filled: "
            + ", ".join(absent_features),
            stacklevel=2,
        )

    model_file, normalization_file = _model_paths(model_base)
    model_base.parent.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    training_env = _make_vec_env(training)
    if mode == "train":
        normalized_training_env = VecNormalize(
            training_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
        model = PPO(
            "MlpPolicy",
            normalized_training_env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            gamma=0.99,
            seed=seed,
            verbose=1,
        )
        model.learn(total_timesteps=total_timesteps)
        model.save(str(model_base))
        normalized_training_env.save(str(normalization_file))
        normalized_training_env.close()
    else:
        training_env.close()
        if not model_file.is_file() or not normalization_file.is_file():
            raise FileNotFoundError(
                "evaluation requires both saved files: "
                f"{model_file} and {normalization_file}"
            )
        eval_env = VecNormalize.load(str(normalization_file), _make_vec_env(holdout))
        eval_env.training = False
        eval_env.norm_reward = False
        model = PPO.load(str(model_base), env=eval_env)
        eval_env.close()

    report = {
        "data_file": data_source,
        "data_source": "google_sheet1" if sheet1 else "local_file",
        "total_rows": len(frame),
        "training_rows": len(training),
        "holdout_rows": len(holdout),
        "split": "chronological; oldest 80% train, newest 20% holdout",
        "model_file": str(model_file),
        "normalization_file": str(normalization_file),
        "training_timesteps": total_timesteps if mode == "train" else 0,
        "holdout": evaluate_model(model, normalization_file, holdout),
        "warning": (
            "Historical replay is not proof of future profitability. This offline "
            "environment has no counterfactual prices for the opposite option side."
        ),
    }
    report_path = Path(f"{model_base}_holdout_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Holdout report saved to {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    input_source = parser.add_mutually_exclusive_group(required=True)
    input_source.add_argument(
        "--data",
        type=Path,
        help="completed trade rows in CSV or JSONL format",
    )
    input_source.add_argument(
        "--sheet1",
        action="store_true",
        help="read all Crude trade rows directly from Google Sheet 1",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/crude_ppo"),
        help="base path for the isolated model and holdout report",
    )
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be greater than zero")

    run_pipeline(
        data_path=args.data,
        model_base=args.model_path,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        mode=args.mode,
        sheet1=args.sheet1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
