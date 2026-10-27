"""Train the independent entry/exit/SL/TP PPO agent on logged price paths."""

from __future__ import annotations

import argparse
from pathlib import Path

from crude_tick_env import CrudeTradingEnv, TradingRewardConfig, load_sheet1_trades
from crude_tick_policy import EngineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--sheet1",
        action="store_true",
        help="load closed trades and option price paths from Google Sheet 1",
    )
    source.add_argument(
        "--data",
        type=Path,
        default=Path("data/crude_closed_trades.csv"),
        help="local CSV of closed trades (used when --sheet1 is not given)",
    )
    parser.add_argument("--model-path", type=Path, default=Path("models/crude_trading_ppo"))
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--slippage-points", type=float, default=0.5)
    parser.add_argument("--cost-points", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    engine_defaults, reward_defaults = EngineConfig(), TradingRewardConfig()
    parser.add_argument("--sl-atr-bounds", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=engine_defaults.sl_atr_bounds)
    parser.add_argument("--tp-rr-bounds", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=engine_defaults.tp_rr_bounds)
    parser.add_argument("--trail-gap-r-bounds", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=engine_defaults.trailing_gap_r_bounds)
    parser.add_argument("--min-entry-momentum-atr", type=float, nargs="*",
                        default=list(engine_defaults.min_entry_momentum_atr),
                        help="per-window (5, 20, ...) minimum signed momentum; empty disables")
    for name in ("target_bonus", "target_3r_bonus", "extra_r_bonus", "early_stop_penalty",
                 "expectancy_weight", "idle_penalty", "dead_chase_penalty"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float,
                            default=getattr(reward_defaults, name))
    args = parser.parse_args()

    from stable_baselines3 import PPO

    data = load_sheet1_trades() if args.sheet1 else args.data
    full = CrudeTradingEnv.from_tick_trades(data)
    print(f"Loaded {len(full.episodes)} usable episodes from "
          f"{'Google Sheet 1' if args.sheet1 else args.data}")
    if len(full.episodes) < 5:
        raise SystemExit("Need at least 5 usable episodes with price paths to train.")
    # Oldest episodes train; newest are held back for out-of-sample evaluation.
    split = int(len(full.episodes) * (1.0 - args.holdout_fraction))
    kwargs = dict(
        engine_config=EngineConfig(
            args.slippage_points,
            args.cost_points,
            sl_atr_bounds=tuple(args.sl_atr_bounds),
            tp_rr_bounds=tuple(args.tp_rr_bounds),
            trailing_gap_r_bounds=tuple(args.trail_gap_r_bounds),
            min_entry_momentum_atr=tuple(args.min_entry_momentum_atr),
        ),
        reward_config=TradingRewardConfig(
            **{name: getattr(args, name) for name in (
                "target_bonus", "target_3r_bonus", "extra_r_bonus", "early_stop_penalty",
                "expectancy_weight", "idle_penalty", "dead_chase_penalty",
            )}
        ),
    )
    train_env = CrudeTradingEnv(full.episodes[:split], **kwargs)
    holdout_env = CrudeTradingEnv(full.episodes[split:], **kwargs)

    model = PPO("MlpPolicy", train_env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.total_timesteps)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_path))

    total_r, trades = 0.0, 0
    for index in range(len(holdout_env.episodes)):
        obs, _ = holdout_env.reset(options={"episode_index": index})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = holdout_env.step(action)
            for event in info["events"]:
                if event["type"] == "exit":
                    trades += 1
                    total_r += event["pnl_r"]
    expectancy = total_r / trades if trades else 0.0
    print(f"Holdout: {trades} trades, total {total_r:.2f}R, expectancy {expectancy:.3f}R")


if __name__ == "__main__":
    main()
