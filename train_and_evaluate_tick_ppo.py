"""Train a tick-risk PPO policy on the oldest trades and report a later holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crude_tick_env import CrudeTickEnv, prepare_tick_trades


def chronological_split(
    trades: pd.DataFrame, test_fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split valid trades chronologically, keeping the newest trades for testing."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    if len(trades) < 2:
        raise ValueError("at least two valid trades are required for an 80/20 split")

    ordered = trades.sort_values("_entry_timestamp", kind="stable").reset_index(drop=True)
    split_index = int(math.floor(len(ordered) * (1.0 - test_fraction)))
    split_index = min(max(split_index, 1), len(ordered) - 1)
    return (
        ordered.iloc[:split_index].reset_index(drop=True),
        ordered.iloc[split_index:].reset_index(drop=True),
    )


def evaluate_model(model: Any, holdout: pd.DataFrame) -> list[dict[str, Any]]:
    """Evaluate one deterministic policy episode for each held-out trade."""
    env = CrudeTickEnv(holdout)
    results: list[dict[str, Any]] = []

    for row_index in range(len(env.data)):
        observation, reset_info = env.reset(options={"row_index": row_index})
        terminated = False
        info: dict[str, Any] = {}
        while not terminated:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            if truncated:
                raise RuntimeError("tick replay unexpectedly truncated an episode")

        results.append(
            {
                "source_row": reset_info["source_row"],
                "entry_timestamp": reset_info["entry_timestamp"],
                "exit_timestamp": reset_info["exit_timestamp"],
                "symbol": reset_info["symbol"],
                "entry_price": float(env.data.iloc[row_index]["_entry_price"]),
                "simulated_exit_price": float(info["exit_price"]),
                "simulated_pnl_points": float(info["pnl_points"]),
                "initial_risk_points": float(info["initial_risk_points"]),
                "simulated_pnl_r": (
                    float(info["pnl_points"]) / float(info["initial_risk_points"])
                ),
                "exit_reason": info["exit_reason"],
                "sl_atr_multiplier": float(info["sl_atr_multiplier"]),
                "trail_gap_r": float(info["trail_gap_r"]),
                "sl_buffer_atr": float(info["sl_buffer_atr"]),
                "max_favorable_r": float(info["max_favorable_r"]),
                "path_peak_r_hypothetical": (
                    max(env.data.iloc[row_index]["_price_path"])
                    - float(env.data.iloc[row_index]["_entry_price"])
                )
                / float(info["initial_risk_points"]),
            }
        )

    return results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize simulated premium-point outcomes without assuming lot size."""
    pnls = np.asarray(
        [result["simulated_pnl_points"] for result in results], dtype=float
    )
    pnls_r = np.asarray([result["simulated_pnl_r"] for result in results], dtype=float)
    gains = float(pnls[pnls > 0].sum())
    losses = abs(float(pnls[pnls < 0].sum()))
    gains_r = float(pnls_r[pnls_r > 0].sum())
    losses_r = abs(float(pnls_r[pnls_r < 0].sum()))
    equity = np.concatenate(([0.0], np.cumsum(pnls)))
    drawdown = np.maximum.accumulate(equity) - equity
    exit_reasons = pd.Series(
        [result["exit_reason"] for result in results], dtype="string"
    ).value_counts()

    return {
        "trades_tested": int(len(pnls)),
        "wins": int((pnls > 0).sum()),
        "losses": int((pnls < 0).sum()),
        "breakeven": int((pnls == 0).sum()),
        "win_rate": float((pnls > 0).mean()) if len(pnls) else 0.0,
        "total_simulated_pnl_points": float(pnls.sum()),
        "average_simulated_pnl_points": float(pnls.mean()) if len(pnls) else 0.0,
        "total_simulated_pnl_r": float(pnls_r.sum()),
        "average_simulated_pnl_r": float(pnls_r.mean()) if len(pnls_r) else 0.0,
        "average_win_r": float(pnls_r[pnls_r > 0].mean())
        if (pnls_r > 0).any()
        else 0.0,
        "average_loss_r": float(pnls_r[pnls_r < 0].mean())
        if (pnls_r < 0).any()
        else 0.0,
        "profit_factor": (
            gains / losses if losses else (999.0 if gains else 0.0)
        ),
        "profit_factor_r": (
            gains_r / losses_r if losses_r else (999.0 if gains_r else 0.0)
        ),
        "max_drawdown_points": float(drawdown.max()),
        "stop_exit_count": int(exit_reasons.get("stop_loss", 0)),
        "worst_loss_r": float(pnls_r.min()) if len(pnls_r) else 0.0,
        "reached_1r": sum(r.get("max_favorable_r", 0.0) >= 1.0 for r in results),
        "reached_2r": sum(r.get("max_favorable_r", 0.0) >= 2.0 for r in results),
        "exited_at_or_above_2r": int((pnls_r >= 2.0).sum()),
        "path_peak_reached_2r_hypothetical": sum(
            r.get("path_peak_r_hypothetical", 0.0) >= 2.0 for r in results
        ),
        "exit_reasons": {str(reason): int(count) for reason, count in exit_reasons.items()},
    }


def _train_model(
    train_data: pd.DataFrame,
    total_timesteps: int,
    seed: int,
    *,
    env_kwargs: dict[str, Any] | None = None,
) -> Any:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "PPO dependencies are missing. Install them with "
            "'python -m pip install -r requirements-crude-ppo.txt'."
        ) from exc

    training_env = DummyVecEnv(
        [lambda: CrudeTickEnv(train_data, **(env_kwargs or {}))]
    )
    model = PPO("MlpPolicy", training_env, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    return model


def _load_model(model_path: Path) -> Any:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "PPO dependencies are missing. Install them with "
            "'python -m pip install -r requirements-crude-ppo.txt'."
        ) from exc
    return PPO.load(str(model_path))


def load_sheet1_tick_data() -> pd.DataFrame:
    """Read all populated Crude trade rows from Google Sheet 1."""
    from sheets_logger import get_gspread_client

    worksheet = get_gspread_client().open("Crude_Algo_Trade_Logs").sheet1
    values = worksheet.get_all_values()
    if not values:
        raise ValueError("Google Sheet 1 is empty")

    headers = [str(header).strip() for header in values[0]]
    symbol_index = next(
        (
            index
            for index, header in enumerate(headers)
            if " ".join(header.lower().replace("_", " ").split()) == "symbol"
        ),
        None,
    )
    if symbol_index is None:
        raise ValueError("Google Sheet 1 is missing the required Symbol header")

    rows = [
        (list(row) + [""] * len(headers))[: len(headers)]
        for row in values[1:]
        if any(str(cell).strip() for cell in row)
        and symbol_index < len(row)
        and "CRUDE" in str(row[symbol_index]).upper()
    ]
    if not rows:
        raise ValueError("Google Sheet 1 contains no populated Crude trade rows")
    return pd.DataFrame(rows, columns=headers, dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/crude_closed_trades.csv"),
        help="CSV of completed crude trades and intratrade option prices",
    )
    parser.add_argument(
        "--sheet1",
        action="store_true",
        help="read Crude trade rows directly from Google Sheet 1",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/crude_tick_ppo_run_trailing_1r"),
        help="model path (without the .zip suffix)",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate"),
        default="train",
        help="train then evaluate the holdout, or evaluate a saved model only",
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-rr", type=float, default=None)
    parser.add_argument("--target-bonus-weight", type=float, default=0.0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="report/CSV path prefix (without suffix); defaults to --model-path",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="allow replacing existing model, report or CSV files",
    )
    args = parser.parse_args()

    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be a positive integer")
    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test-fraction must be strictly between 0 and 1")

    prefix = args.output_prefix or args.model_path
    report_path = prefix.with_name(f"{prefix.name}_holdout_report.json")
    trades_path = prefix.with_name(f"{prefix.name}_holdout_trades.csv")
    protected = [report_path, trades_path]
    if args.mode == "train":
        protected.append(args.model_path.with_name(args.model_path.name + ".zip"))
    if not args.allow_overwrite and any(path.exists() for path in protected):
        parser.error("output file already exists; choose a new name or --allow-overwrite")

    dataset_sha256 = None
    if args.sheet1:
        raw_data = load_sheet1_tick_data()
        source = "Google Sheet 1: Crude_Algo_Trade_Logs"
    else:
        raw_data = pd.read_csv(args.data, encoding="utf-8-sig")
        source = str(args.data)
        dataset_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    trades = prepare_tick_trades(raw_data)
    skipped_rows = dict(trades.attrs.get("skipped_rows", {}))
    train_data, holdout = chronological_split(trades, args.test_fraction)

    if args.mode == "train":
        model = _train_model(
            train_data,
            args.total_timesteps,
            args.seed,
            env_kwargs={
                "target_rr": args.target_rr,
                "target_bonus_weight": args.target_bonus_weight,
            },
        )
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.model_path))
    else:
        model = _load_model(args.model_path)

    results = evaluate_model(model, holdout)
    summary = summarize_results(results)
    report = {
        "source": source,
        "model": str(args.model_path),
        "mode": args.mode,
        "test_fraction": args.test_fraction,
        "total_valid_trade_paths": int(len(trades)),
        "training_trade_paths": int(len(train_data)),
        "holdout_trade_paths": int(len(holdout)),
        "training_period": {
            "first_entry": train_data["_entry_timestamp"].min().isoformat(),
            "last_entry": train_data["_entry_timestamp"].max().isoformat(),
        },
        "holdout_period": {
            "first_entry": holdout["_entry_timestamp"].min().isoformat(),
            "last_entry": holdout["_entry_timestamp"].max().isoformat(),
        },
        "excluded_rows": skipped_rows,
        "pnl_unit": "option premium points (not rupees; quantity and costs are not applied)",
        "reward_unit": "R multiples; 1R is the initial entry-to-stop risk",
        "dataset_sha256": dataset_sha256,
        "reward_config": {
            "target_rr": args.target_rr,
            "target_bonus_weight": args.target_bonus_weight,
        },
        "holdout_source_rows": [int(v) for v in holdout["_source_row"]],
        "limitations": (
            "premium points only; no quantity, brokerage or slippage; rows without "
            "price path excluded; single seed; reach metrics use observed samples"
        ),
        "results": summary,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(trades_path, index=False, encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Per-trade holdout results: {trades_path}")
    print(f"Holdout summary report: {report_path}")


if __name__ == "__main__":
    main()
