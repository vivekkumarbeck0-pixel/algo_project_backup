"""Compare PnL and R reward mixes for the crude tick-risk PPO."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crude_tick_env import CrudeTickEnv
from train_and_evaluate_tick_ppo import (
    _train_model,
    evaluate_model,
    load_sheet1_tick_data,
    summarize_results,
)


PNL_REWARD_WEIGHTS = (0.0, 0.5, 1.0)
RR_REWARD_WEIGHTS = (0.5, 1.0, 2.0)


def chronological_three_way_split(
    trades: pd.DataFrame, test_fraction: float = 0.20, validation_fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep newest data for final test and the preceding period for selection."""
    if not 0.0 < test_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("test and validation fractions must be strictly between 0 and 1")
    if test_fraction + validation_fraction >= 1.0:
        raise ValueError("test and validation fractions must sum to less than 1")

    ordered = trades.sort_values("_entry_timestamp", kind="stable").reset_index(drop=True)
    if len(ordered) < 3:
        raise ValueError("at least three valid trades are required for three-way split")
    test_start = int(np.floor(len(ordered) * (1.0 - test_fraction)))
    validation_start = int(np.floor(test_start * (1.0 - validation_fraction)))
    validation_start = min(max(validation_start, 1), test_start - 1)
    test_start = min(max(test_start, validation_start + 1), len(ordered) - 1)
    return (
        ordered.iloc[:validation_start].reset_index(drop=True),
        ordered.iloc[validation_start:test_start].reset_index(drop=True),
        ordered.iloc[test_start:].reset_index(drop=True),
    )


def rank_reward_trials(
    trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank by positive validation PnL, then points, net R, and profit factor."""
    return sorted(
        trials,
        key=lambda trial: (
            float(trial["validation"]["total_simulated_pnl_points"]) > 0.0,
            float(trial["validation"]["total_simulated_pnl_points"]),
            float(trial["validation"]["total_simulated_pnl_r"]),
            float(trial["validation"]["profit_factor_r"]),
            float(trial["validation"]["win_rate"]),
        ),
        reverse=True,
    )


def run_reward_grid(
    trades: pd.DataFrame,
    *,
    model_path: Path,
    timesteps_per_trial: int,
    seed: int,
) -> dict[str, Any]:
    """Train reward variants, select on validation, and report final holdout once."""
    development, validation, holdout = chronological_three_way_split(trades)
    baseline = CrudeTickEnv(development)
    risk_scale = float(
        np.median(
            (baseline.initial_sl_atr + baseline.initial_sl_buffer_atr)
            * baseline.data["_atr"].to_numpy(dtype=float)
        )
    )
    if not np.isfinite(risk_scale) or risk_scale <= 0.0:
        raise ValueError("training paths do not provide a positive PnL reward scale")

    configurations = list(
        itertools.product(PNL_REWARD_WEIGHTS, RR_REWARD_WEIGHTS)
    )
    trials: list[dict[str, Any]] = []
    best_model = None
    best_normalized_score: tuple[float, float, float, float, float] | None = None

    for index, (pnl_weight, rr_weight) in enumerate(configurations, start=1):
        env_kwargs = {
            "pnl_reward_weight": pnl_weight,
            "rr_reward_weight": rr_weight,
            "pnl_reward_scale": risk_scale,
        }
        model = _train_model(
            development,
            timesteps_per_trial,
            seed,
            env_kwargs=env_kwargs,
        )
        validation_results = evaluate_model(model, validation)
        validation_summary = summarize_results(validation_results)
        trial = {
            "trial": index,
            "pnl_reward_weight": pnl_weight,
            "rr_reward_weight": rr_weight,
            "pnl_reward_scale_points": risk_scale,
            "validation": validation_summary,
        }
        trials.append(trial)

        score = (
            float(validation_summary["total_simulated_pnl_points"]) > 0.0,
            float(validation_summary["total_simulated_pnl_points"]),
            float(validation_summary["total_simulated_pnl_r"]),
            float(validation_summary["profit_factor_r"]),
            float(validation_summary["win_rate"]),
        )
        if best_normalized_score is None or score > best_normalized_score:
            if best_model is not None:
                del best_model
            best_model = model
            best_normalized_score = score
        else:
            del model
        print(
            f"[{index}/{len(configurations)}] PnL weight={pnl_weight:.2f}, "
            f"R weight={rr_weight:.2f}, validation PnL="
            f"{validation_summary['total_simulated_pnl_points']:.2f}, "
            f"net R={validation_summary['total_simulated_pnl_r']:.2f}, "
            f"trades={validation_summary['trades_tested']}"
        )

    if best_model is None:
        raise RuntimeError("reward grid completed without producing a model")

    ranked = rank_reward_trials(trials)
    selected = ranked[0]
    best_model.save(str(model_path))
    final_results = evaluate_model(best_model, holdout)
    final_summary = summarize_results(final_results)
    report = {
        "objective": (
            "validation positive PnL first, then total PnL points, net R, "
            "R profit factor, and win rate"
        ),
        "selection_data": {
            "training_trades": int(len(development)),
            "validation_trades": int(len(validation)),
            "final_holdout_trades": int(len(holdout)),
            "pnl_reward_scale_points": risk_scale,
        },
        "selected_trial": selected,
        "ranked_trials": ranked,
        "final_holdout": final_summary,
        "final_holdout_warning": (
            "The final holdout was not used to select reward weights. "
            "Results are historical simulations and do not guarantee future profit."
        ),
    }
    report_path = model_path.with_name(f"{model_path.name}_reward_grid_report.json")
    trades_path = model_path.with_name(f"{model_path.name}_holdout_trades.csv")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(final_results).to_csv(trades_path, index=False, encoding="utf-8")
    print(
        f"Selected PnL/R weights: {selected['pnl_reward_weight']:.2f}/"
        f"{selected['rr_reward_weight']:.2f}; final holdout net R="
        f"{final_summary['total_simulated_pnl_r']:.2f}, "
        f"PnL={final_summary['total_simulated_pnl_points']:.2f}, "
        f"trades={final_summary['trades_tested']}"
    )
    print(f"Saved model: {model_path}.zip")
    print(f"Saved report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet1", action="store_true", help="read crude paths from Sheet1")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/crude_closed_trades.csv"),
        help="CSV of closed trades and intratrade option price paths",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/crude_tick_ppo_reward_grid_best"),
        help="output path for selected model (without .zip)",
    )
    parser.add_argument("--timesteps-per-trial", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.timesteps_per_trial <= 0:
        parser.error("--timesteps-per-trial must be positive")

    raw_data = load_sheet1_tick_data() if args.sheet1 else pd.read_csv(
        args.data, encoding="utf-8-sig"
    )
    trades = CrudeTickEnv(raw_data).data
    run_reward_grid(
        trades,
        model_path=args.model_path,
        timesteps_per_trial=args.timesteps_per_trial,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
