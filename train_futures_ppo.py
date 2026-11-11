"""Train an experimental PPO from the sampled futures Google Sheet or an exported CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from crude_tick_env import CrudeTradingEnv
from crude_tick_policy import EngineConfig, SIDE_NAMES, map_action
from ppo_futures_sheet import HEADERS, WORKSHEET_TITLE

IST = ZoneInfo("Asia/Kolkata")
MIN_DAYS = 20
MIN_EPISODE_SAMPLES = 121
MAX_EPISODE_SAMPLES = 360
MAX_GAP_SECONDS = 30


def load_sheet_rows() -> list[dict[str, str]]:
    from sheets_logger import get_gspread_client

    worksheet = get_gspread_client().open("Crude_Algo_Trade_Logs").worksheet(WORKSHEET_TITLE)
    values = worksheet.get_all_values()
    if not values or values[0] != HEADERS:
        raise ValueError(f"{WORKSHEET_TITLE} is missing the expected headers")
    return [dict(zip(HEADERS, row)) for row in values[1:]]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADERS:
            raise ValueError(f"CSV headers must be {HEADERS}")
        return list(reader)


def build_episodes(rows: list[dict[str, str]]) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], dict[str, int]]:
    """Split at dates, sessions, tokens, invalid samples, long gaps and one-hour boundaries."""
    parsed = []
    counts: Counter[str] = Counter()
    for row in rows:
        try:
            timestamp = datetime.fromisoformat(row["Timestamp"])
            if timestamp.tzinfo is None:
                raise ValueError("timezone missing")
            timestamp = timestamp.astimezone(IST)
            price = float(row["Price"])
            token, session = row["Token"].strip(), row["Session ID"].strip()
            if not math.isfinite(price) or price <= 0 or not token or not session:
                raise ValueError("invalid price, token or session")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid futures sample: {exc}") from exc
        try:
            atr = float(row.get("ATR", ""))
            if not math.isfinite(atr) or atr <= 0:
                raise ValueError("invalid ATR")
        except (TypeError, ValueError):
            atr = None
            counts["atr_not_ready"] += 1
        parsed.append((timestamp, price, atr, token, session))
    parsed.sort(key=lambda item: item[0])

    episodes: list[tuple[str, np.ndarray, np.ndarray]] = []
    segment: list[tuple] = []

    def finish() -> None:
        if len(segment) >= MIN_EPISODE_SAMPLES:
            episodes.append((
                segment[0][0].date().isoformat(),
                np.asarray([sample[1] for sample in segment], dtype=float),
                np.asarray([sample[2] for sample in segment], dtype=float),
            ))
        elif segment:
            counts["short_segments"] += 1
        segment.clear()

    previous = None
    for sample in parsed:
        if previous is not None and sample[0] == previous[0]:
            if sample == previous:
                counts["duplicates"] += 1
                continue
            raise ValueError(f"Conflicting samples at {sample[0].isoformat()}")
        if sample[2] is None:
            finish()
        else:
            if segment:
                last = segment[-1]
                if (
                    sample[0].date() != last[0].date()
                    or sample[3:] != last[3:]
                    or (sample[0] - last[0]).total_seconds() > MAX_GAP_SECONDS
                    or len(segment) >= MAX_EPISODE_SAMPLES
                ):
                    finish()
            segment.append(sample)
        previous = sample
    finish()
    counts["input_rows"] = len(rows)
    counts["episodes"] = len(episodes)
    return episodes, dict(counts)


def split_days(episodes: list[tuple[str, np.ndarray, np.ndarray]], holdout_fraction: float):
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout fraction must be between 0 and 1")
    days = sorted({episode[0] for episode in episodes})
    if len(days) < MIN_DAYS:
        raise ValueError(f"Need at least {MIN_DAYS} distinct usable trading days; found {len(days)}")
    split = int(len(days) * (1 - holdout_fraction))
    if split < 1 or split >= len(days):
        raise ValueError("holdout fraction leaves an empty training or holdout period")
    train_days, holdout_days = days[:split], days[split:]
    train = [(p, a) for day, p, a in episodes if day in train_days]
    holdout = [(p, a) for day, p, a in episodes if day in holdout_days]
    return train, holdout, train_days, holdout_days


def evaluate(model: Any, env: CrudeTradingEnv) -> dict[str, Any]:
    intents: Counter[str] = Counter()
    side_metrics = {
        side: {"trades": 0, "wins": 0, "net_r": 0.0, "net_points": 0.0}
        for side in ("LONG", "SHORT")
    }
    equity, peak, max_drawdown = 0.0, 0.0, 0.0
    for index in range(len(env.episodes)):
        obs, _ = env.reset(options={"episode_index": index})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            intents[SIDE_NAMES[map_action(action)[0]]] += 1
            obs, _, done, _, info = env.step(action)
            for event in info["events"]:
                if event["type"] == "exit":
                    side = side_metrics[event["side"]]
                    side["trades"] += 1
                    side["wins"] += int(event["pnl_points"] > 0)
                    side["net_r"] += float(event["pnl_r"])
                    side["net_points"] += float(event["pnl_points"])
                    equity += float(event["pnl_points"])
                    peak = max(peak, equity)
                    max_drawdown = max(max_drawdown, peak - equity)
    return {"intents": dict(intents), "sides": side_metrics, "max_drawdown_points": max_drawdown}


def run(rows: list[dict[str, str]], model_path: Path, timesteps: int, seed: int, holdout_fraction: float) -> dict[str, Any]:
    if timesteps <= 0:
        raise ValueError("total timesteps must be positive")
    if model_path.suffix:
        raise ValueError("--model-path must be a filename without an extension")
    output = model_path.with_suffix(".zip")
    report_path = model_path.with_name(model_path.name + "_holdout_report.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("Output already exists; choose a new --model-path")
    episodes, counts = build_episodes(rows)
    train, holdout, train_days, holdout_days = split_days(episodes, holdout_fraction)
    from stable_baselines3 import PPO

    config = EngineConfig(cost_points=0.5, trailing_enabled=False)
    train_env = CrudeTradingEnv(train, engine_config=config, warmup_ticks=100)
    holdout_env = CrudeTradingEnv(holdout, engine_config=config, warmup_ticks=100)
    model = PPO("MlpPolicy", train_env, verbose=1, seed=seed, n_steps=256, batch_size=64)
    model.learn(total_timesteps=timesteps)
    metrics = evaluate(model, holdout_env)
    report = {
        "input": counts,
        "train_days": train_days,
        "holdout_days": holdout_days,
        "train_episodes": len(train),
        "holdout_episodes": len(holdout),
        "timesteps": timesteps,
        "actual_timesteps": int(model.num_timesteps),
        "seed": seed,
        "cost_points": config.cost_points,
        "sampling_seconds": 10,
        "metrics": metrics,
        "limitations": [
            "Experimental sampled futures replay; no intrainterval stop fills.",
            "Per-episode resets, no daily loss/trade limits and different sampling from current paper runner.",
            "Not validated for live or paper deployment.",
        ],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sheet", action="store_true", help="read PPO_Futures_Samples")
    source.add_argument("--csv", type=Path, help="exported PPO_Futures_Samples CSV")
    parser.add_argument("--model-path", type=Path, required=True, help="new experimental output path; existing files are not overwritten")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = load_sheet_rows() if args.sheet else load_csv_rows(args.csv)
    report = run(rows, args.model_path, args.total_timesteps, args.seed, args.holdout_fraction)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
