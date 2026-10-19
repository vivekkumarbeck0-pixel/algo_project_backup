"""Search crude PPO reward settings using the Sheet1 chronological holdout."""

from __future__ import annotations

import argparse
import gc
import itertools
import tempfile
from pathlib import Path

import pandas as pd

import crude_ppo_env as reward_env
from train_and_evaluate_ppo import (
    _make_vec_env,
    chronological_split,
    evaluate_model,
    load_sheet1_trade_data,
)


ENTRY_BONUSES = (0.5, 1.0, 1.5)
IDLE_REWARDS = (-0.5, -1.0, -1.5)
TRADE_REWARD_SCALES = (0.5, 1.0, 2.0)
TOTAL_TIMESTEPS = 30_000


def _train_and_evaluate(
    training: pd.DataFrame,
    holdout: pd.DataFrame,
    seed: int,
    normalization_path: Path,
) -> dict[str, float | int]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError as exc:
        raise RuntimeError(
            "PPO dependencies are missing. Install them with "
            "'pip install -r requirements-crude-ppo.txt'."
        ) from exc

    training_env = _make_vec_env(training)
    normalized_env = None
    try:
        normalized_env = VecNormalize(
            training_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
        model = PPO(
            "MlpPolicy",
            normalized_env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            gamma=0.99,
            seed=seed,
            verbose=0,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        normalized_env.save(str(normalization_path))
        evaluation = evaluate_model(model, normalization_path, holdout)
        return evaluation["metrics"]
    finally:
        if normalized_env is not None:
            normalized_env.close()
        else:
            training_env.close()


def _print_summary(results: list[dict[str, float | int]]) -> None:
    ranked = sorted(results, key=lambda result: float(result["total_pnl"]), reverse=True)
    headers = (
        "Rank",
        "Entry Bonus",
        "Idle Reward",
        "Trade Scale",
        "Total PnL",
        "Win Rate",
        "Executed Trades",
        "Profit Factor",
    )
    rows = []
    for rank, result in enumerate(ranked, start=1):
        rows.append(
            (
                f"{rank}{' *' if rank == 1 else ''}",
                f"{result['entry_bonus']:.1f}",
                f"{result['idle_reward']:.1f}",
                f"{result['trade_reward_scale']:.1f}",
                f"{result['total_pnl']:.2f}",
                f"{float(result['win_rate']) * 100:.1f}%",
                str(result["trades"]),
                f"{result['profit_factor']:.2f}",
            )
        )

    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.rjust(widths[index]) for index, value in enumerate(row)))

    if ranked:
        best = ranked[0]
        print(
            "\nBest configuration by holdout total PnL (*): "
            f"ENTRY_BONUS={best['entry_bonus']}, "
            f"IDLE_REWARD={best['idle_reward']}, "
            f"TRADE_REWARD_SCALE={best['trade_reward_scale']} "
            f"(Total PnL={best['total_pnl']:.2f})"
        )


def run_grid_search(seed: int = 42) -> list[dict[str, float | int]]:
    """Train and evaluate one PPO model for each reward-parameter combination."""
    original_constants = (
        reward_env.ENTRY_BONUS,
        reward_env.IDLE_REWARD,
        reward_env.TRADE_REWARD_SCALE,
    )
    results: list[dict[str, float | int]] = []

    try:
        sheet_data = load_sheet1_trade_data()
        configurations = list(
            itertools.product(ENTRY_BONUSES, IDLE_REWARDS, TRADE_REWARD_SCALES)
        )
        print(
            f"Running {len(configurations)} configurations "
            f"({TOTAL_TIMESTEPS:,} PPO steps each; "
            f"{len(configurations) * TOTAL_TIMESTEPS:,} total steps)."
        )

        for index, (entry_bonus, idle_reward, trade_reward_scale) in enumerate(
            configurations, start=1
        ):
            reward_env.ENTRY_BONUS = entry_bonus
            reward_env.IDLE_REWARD = idle_reward
            reward_env.TRADE_REWARD_SCALE = trade_reward_scale

            prepared = reward_env.prepare_trade_data(sheet_data)
            training, holdout = chronological_split(prepared)
            print(
                f"[{index}/{len(configurations)}] "
                f"ENTRY_BONUS={entry_bonus}, IDLE_REWARD={idle_reward}, "
                f"TRADE_REWARD_SCALE={trade_reward_scale}"
            )

            with tempfile.TemporaryDirectory(prefix="crude-ppo-reward-grid-") as temp_dir:
                metrics = _train_and_evaluate(
                    training,
                    holdout,
                    seed,
                    Path(temp_dir) / "vecnormalize.pkl",
                )

            results.append(
                {
                    "entry_bonus": entry_bonus,
                    "idle_reward": idle_reward,
                    "trade_reward_scale": trade_reward_scale,
                    "total_pnl": float(metrics["total_pnl"]),
                    "win_rate": float(metrics["win_rate"]),
                    "trades": int(metrics["trades"]),
                    "profit_factor": float(metrics["profit_factor"]),
                }
            )
            gc.collect()
    finally:
        (
            reward_env.ENTRY_BONUS,
            reward_env.IDLE_REWARD,
            reward_env.TRADE_REWARD_SCALE,
        ) = original_constants

    _print_summary(results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="PPO random seed per run")
    args = parser.parse_args()
    run_grid_search(seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
