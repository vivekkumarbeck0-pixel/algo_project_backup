"""Find the best training length for the crude PPO agent.

Trains one PPO run continuously, pausing at each checkpoint timestep to save the
model and score it on the same chronological holdout used by
``train_crude_trading_ppo.py``. The best checkpoint is copied to ``--best-path``.

    python tune_timesteps.py --sheet1
    python tune_timesteps.py --data data/crude_closed_trades.csv --checkpoints 100000 300000
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from crude_tick_env import CrudeTradingEnv, TradingRewardConfig, load_sheet1_trades
from crude_tick_policy import (
    TRAIL_ACTIVATION_POINTS,
    TRAIL_BREAKEVEN_BUFFER_POINTS,
    EngineConfig,
)

DEFAULT_CHECKPOINTS = (100_000, 200_000, 300_000, 500_000, 1_000_000)


@dataclass
class CheckpointResult:
    timesteps: int
    trades: int
    total_r: float
    expectancy_r: float
    path: str
    exit_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)


def evaluate(
    model, holdout_env: CrudeTradingEnv
) -> tuple[int, float, dict[str, dict[str, float]]]:
    """Deterministically replay every holdout episode.

    Returns (trades, total R, per-exit-reason {"count", "total_r", "avg_r"}).
    """
    trades, total_r = 0, 0.0
    by_reason: dict[str, dict[str, float]] = {}
    for index in range(len(holdout_env.episodes)):
        obs, _ = holdout_env.reset(options={"episode_index": index})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = holdout_env.step(action)
            for event in info["events"]:
                if event["type"] == "exit":
                    pnl_r = float(event["pnl_r"])
                    trades += 1
                    total_r += pnl_r
                    row = by_reason.setdefault(
                        str(event["reason"]),
                        {"count": 0, "total_r": 0.0, "avg_r": 0.0, "trail_armed": 0},
                    )
                    row["count"] += 1
                    row["total_r"] += pnl_r
                    row["trail_armed"] += int(bool(event.get("trail_active")))
    for row in by_reason.values():
        row["avg_r"] = row["total_r"] / row["count"]
    return trades, total_r, by_reason


def format_exit_breakdown(result: CheckpointResult) -> str:
    """Table of exit reasons for one checkpoint, ending with the overall row."""
    lines = [
        f"Exit-reason breakdown @ {result.timesteps:,} timesteps",
        f"{'Exit reason':<14} {'Trades':>7} {'Total R':>10} {'Avg R':>9} {'% trades':>9} {'Trail armed':>12}",
        "-" * 66,
    ]
    for reason, row in sorted(result.exit_breakdown.items(), key=lambda kv: kv[1]["total_r"]):
        share = 100.0 * row["count"] / result.trades if result.trades else 0.0
        lines.append(
            f"{reason:<14} {int(row['count']):>7} {row['total_r']:>10.2f} "
            f"{row['avg_r']:>9.3f} {share:>8.1f}% {int(row.get('trail_armed', 0)):>12}"
        )
    lines.append("-" * 66)
    lines.append(
        f"{'OVERALL':<14} {result.trades:>7} {result.total_r:>10.2f} "
        f"{result.expectancy_r:>9.3f} {100.0 if result.trades else 0.0:>8.1f}% "
        f"{sum(int(r.get('trail_armed', 0)) for r in result.exit_breakdown.values()):>12}"
    )
    return "\n".join(lines)


def pick_best(results: list[CheckpointResult], min_trades: int) -> CheckpointResult | None:
    """Highest expectancy among checkpoints with enough trades; ties go to total R."""
    eligible = [r for r in results if r.trades >= min_trades]
    if not eligible:
        return None
    return max(eligible, key=lambda r: (r.expectancy_r, r.total_r))


def print_summary(results: list[CheckpointResult], best: CheckpointResult | None) -> None:
    print("\nTimestep tuning summary (holdout)")
    print(f"{'Timesteps':>12} {'Trades':>8} {'Total PnL (R)':>15} {'Expectancy (R)':>16}")
    print("-" * 55)
    for r in results:
        marker = "  <-- best" if best is not None and r is best else ""
        print(
            f"{r.timesteps:>12,} {r.trades:>8} {r.total_r:>15.2f} "
            f"{r.expectancy_r:>16.3f}{marker}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sheet1", action="store_true",
                        help="load closed trades and option price paths from Google Sheet 1")
    source.add_argument("--data", type=Path, default=Path("data/crude_closed_trades.csv"),
                        help="local CSV of closed trades (used when --sheet1 is not given)")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(DEFAULT_CHECKPOINTS),
                        help="cumulative training timesteps at which to evaluate")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--best-path", type=Path, default=Path("models/crude_trading_ppo_best"),
                        help="where the winning checkpoint is saved (.zip is added)")
    parser.add_argument("--min-trades", type=int, default=5,
                        help="ignore checkpoints with fewer holdout trades than this")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--slippage-points", type=float, default=0.5)
    parser.add_argument("--cost-points", type=float, default=0.0)
    parser.add_argument("--trail-activation-points", type=float, default=TRAIL_ACTIVATION_POINTS,
                        help="favorable premium move that arms the trailing stop")
    parser.add_argument("--trail-breakeven-buffer-points", type=float,
                        default=TRAIL_BREAKEVEN_BUFFER_POINTS,
                        help="stop jumps to entry + this many points when the trail arms")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoints = sorted(set(args.checkpoints))
    if not checkpoints or checkpoints[0] <= 0:
        raise SystemExit("--checkpoints must be positive timestep counts")
    if not 0.0 < args.holdout_fraction < 1.0:
        raise SystemExit("--holdout-fraction must be between 0 and 1")

    from stable_baselines3 import PPO

    data = load_sheet1_trades() if args.sheet1 else args.data
    full = CrudeTradingEnv.from_tick_trades(data)
    print(f"Loaded {len(full.episodes)} usable episodes from "
          f"{'Google Sheet 1' if args.sheet1 else args.data}")
    if len(full.episodes) < 5:
        raise SystemExit("Need at least 5 usable episodes with price paths to train.")
    split = int(len(full.episodes) * (1.0 - args.holdout_fraction))
    if split < 1 or split >= len(full.episodes):
        raise SystemExit("Holdout fraction leaves no training or holdout episodes.")
    kwargs = dict(
        engine_config=EngineConfig(
            args.slippage_points,
            args.cost_points,
            trail_activation_points=args.trail_activation_points,
            trail_breakeven_buffer_points=args.trail_breakeven_buffer_points,
        ),
        reward_config=TradingRewardConfig(),
    )
    train_env = CrudeTradingEnv(full.episodes[:split], **kwargs)
    holdout_env = CrudeTradingEnv(full.episodes[split:], **kwargs)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    model = PPO("MlpPolicy", train_env, verbose=0, seed=args.seed)
    results: list[CheckpointResult] = []
    trained = 0
    for target in checkpoints:
        print(f"Training to {target:,} timesteps...")
        model.learn(total_timesteps=target - trained, reset_num_timesteps=False)
        trained = target
        path = args.models_dir / f"crude_trading_ppo_{target // 1000}k"
        model.save(str(path))
        trades, total_r, breakdown = evaluate(model, holdout_env)
        result = CheckpointResult(
            target, trades, total_r, total_r / trades if trades else 0.0,
            f"{path}.zip", breakdown,
        )
        results.append(result)
        print(f"  holdout: {trades} trades, {total_r:.2f}R, "
              f"expectancy {result.expectancy_r:.3f}R")

    best = pick_best(results, args.min_trades)
    print_summary(results, best)
    for result in results:
        print(f"\n{format_exit_breakdown(result)}")
    report = {"results": [asdict(r) for r in results], "best": asdict(best) if best else None}
    (args.models_dir / "tune_timesteps_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if best is None:
        print(f"\nNo checkpoint reached {args.min_trades} holdout trades; no best model saved.")
        return
    destination = args.best_path.with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best.path, destination)
    print(f"\nBest: {best.timesteps:,} timesteps "
          f"(expectancy {best.expectancy_r:.3f}R) -> {destination}")


if __name__ == "__main__":
    main()
