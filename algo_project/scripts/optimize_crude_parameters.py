"""Walk-forward optimization and guarded deployment for the MCX Crude bot."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import settings
from engine.adaptive_config import publish_payload


REQUIRED_COLUMNS = {
    "Entry Timestamp",
    "PnL",
    "Entry Volume",
    "Entry OI_Change",
}


@dataclass(frozen=True)
class Candidate:
    min_volume: float
    min_abs_oi_change: float
    min_atr: float
    max_atr: float
    allowed_regimes: tuple[str, ...]
    atr_multiplier: float
    buffer_points: float
    risk_reward_ratio: float
    trailing_activation_points: float


def load_trades(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path:
        frame = pd.read_csv(csv_path)
    else:
        from train_model import fetch_sheet1_rows

        frame = fetch_sheet1_rows()
    return prepare_trades(frame)


def prepare_trades(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"trade data is missing required columns: {missing}")

    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["Entry Timestamp"], errors="coerce")
    numeric_sources = {
        "pnl": "PnL",
        "atr": "Entry ATR",
        "volume": "Entry Volume",
        "oi_change": "Entry OI_Change",
        "entry_price": "Entry Price",
        "momentum_strength": "Entry Momentum Strength",
    }
    for destination, source in numeric_sources.items():
        prepared[destination] = (
            pd.to_numeric(prepared[source], errors="coerce")
            if source in prepared.columns
            else np.nan
        )
    if "Entry Market Regime" in prepared.columns:
        prepared["regime"] = (
            prepared["Entry Market Regime"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
            .str.strip()
            .replace("", "UNKNOWN")
        )
    else:
        prepared["regime"] = "UNKNOWN"
    prepared["momentum_strength"] = prepared["momentum_strength"].fillna(0.0).clip(0.0, 1.0)
    prepared["price_path"] = (
        prepared["Intratrade Option Prices"].map(_parse_price_path)
        if "Intratrade Option Prices" in prepared.columns
        else [[] for _ in range(len(prepared))]
    )
    prepared.loc[prepared["atr"] <= 0, "atr"] = np.nan
    prepared = prepared.dropna(
        subset=["timestamp", "pnl", "volume", "oi_change"]
    ).sort_values("timestamp").reset_index(drop=True)
    if len(prepared) < 40:
        raise ValueError("at least 40 complete chronological trades are required")
    return prepared


def _parse_price_path(value: object) -> list[float]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str) and value.strip():
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return []
    else:
        return []
    if not isinstance(raw, list):
        return []
    try:
        return [float(item) for item in raw if math.isfinite(float(item))]
    except (TypeError, ValueError):
        return []


def replay_trade(row: pd.Series, candidate: Candidate) -> float:
    path = row["price_path"]
    entry_price = float(row["entry_price"])
    if not path or not math.isfinite(entry_price):
        return float(row["pnl"])

    atr = float(row["atr"])
    momentum_strength = float(row["momentum_strength"])
    dynamic_multiplier = candidate.atr_multiplier * (1.0 + 0.25 * momentum_strength)
    trail_multiplier = dynamic_multiplier * (1.0 + 0.15 * momentum_strength)
    stop_distance = dynamic_multiplier * atr + candidate.buffer_points
    target_distance = candidate.risk_reward_ratio * dynamic_multiplier * atr
    trail_distance = trail_multiplier * atr + 5.0
    stop = entry_price - stop_distance
    target = entry_price + target_distance
    trailing_stop = stop

    for price in path:
        if price >= entry_price + candidate.trailing_activation_points:
            trailing_stop = max(trailing_stop, price - trail_distance)
        if price >= target:
            return target_distance
        if price <= stop:
            return -stop_distance
        if trailing_stop > stop and price <= trailing_stop:
            return trailing_stop - entry_price
    return float(path[-1]) - entry_price


def candidate_pnls(frame: pd.DataFrame, candidate: Candidate, replay_risk: bool) -> list[float]:
    atr_gate_enabled = candidate.min_atr > 0.0 or candidate.max_atr < 1_000_000.0
    atr_matches = frame["atr"].between(candidate.min_atr, candidate.max_atr)
    if not atr_gate_enabled:
        atr_matches = pd.Series(True, index=frame.index)
    selected = frame[
        (frame["volume"] >= candidate.min_volume)
        & (frame["oi_change"].abs() >= candidate.min_abs_oi_change)
        & atr_matches
        & frame["regime"].isin(candidate.allowed_regimes)
    ]
    if replay_risk:
        return [replay_trade(row, candidate) for _, row in selected.iterrows()]
    return selected["pnl"].astype(float).tolist()


def metrics(pnls: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(pnls), dtype=float)
    if not len(values):
        return {"trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}
    gains = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    equity = np.concatenate(([0.0], np.cumsum(values)))
    drawdown = np.maximum.accumulate(equity) - equity
    return {
        "trades": int(len(values)),
        "win_rate": float(np.mean(values > 0)),
        "total_pnl": float(values.sum()),
        "profit_factor": gains / losses if losses else (999.0 if gains else 0.0),
        "max_drawdown": float(drawdown.max()),
    }


def _candidate_grid(calibration: pd.DataFrame, replay_risk: bool) -> list[Candidate]:
    volume_values = sorted({0.0, float(calibration["volume"].quantile(0.25))})
    oi_values = sorted({0.0, float(calibration["oi_change"].abs().quantile(0.25))})
    valid_atr = calibration["atr"].dropna()
    atr_bands = [(0.0, 1_000_000.0)]
    if len(valid_atr) >= 40:
        atr_bands.extend([
            (float(valid_atr.quantile(0.10)), float(valid_atr.quantile(0.90))),
            (float(valid_atr.quantile(0.25)), float(valid_atr.quantile(0.75))),
        ])
    known_regimes = calibration["regime"].isin(("TRENDING", "SIDEWAYS")).sum()
    regimes = [("TRENDING", "SIDEWAYS", "UNKNOWN")]
    if known_regimes >= 40:
        regimes.extend([("TRENDING",), ("SIDEWAYS",)])
    if replay_risk:
        risk_values = itertools.product(
            (1.25, 1.5, 2.0, 2.5),
            (2.0, 4.0, 6.0),
            (1.5, 2.0, 2.5),
            (5.0, 8.0, 12.0),
        )
    else:
        risk_values = [(
            float(settings.crude_atr_multiplier),
            float(settings.buffer_points),
            float(settings.crude_risk_reward_ratio),
            float(settings.crude_trailing_activation_points),
        )]
    risk_values = list(risk_values)
    return [
        Candidate(volume, oi, atr_min, atr_max, regime, atr_mult, buffer, rr, trail)
        for volume, oi, (atr_min, atr_max), regime, (atr_mult, buffer, rr, trail)
        in itertools.product(volume_values, oi_values, atr_bands, regimes, risk_values)
    ]


def optimize(
    frame: pd.DataFrame,
    target_win_rate: float = 0.60,
    folds: int = 4,
    min_trades_per_fold: int = 5,
    holdout_fraction: float = 0.20,
) -> tuple[Candidate, dict[str, object], bool]:
    frame = prepare_trades(frame) if "timestamp" not in frame.columns else frame.copy()
    holdout_count = max(10, int(round(len(frame) * holdout_fraction)))
    development = frame.iloc[:-holdout_count]
    holdout = frame.iloc[-holdout_count:]
    initial_count = max(20, len(development) // 2)
    test_indexes = [chunk for chunk in np.array_split(np.arange(initial_count, len(development)), folds) if len(chunk)]
    if not test_indexes:
        raise ValueError("not enough trades for walk-forward folds")

    calibration = development.iloc[:initial_count]
    replayable = (
        development["price_path"].map(bool)
        & development["entry_price"].notna()
        & development["atr"].notna()
    )
    replayable_count = int(replayable.sum())
    path_coverage = float(replayable.mean())
    replay_risk = replayable_count >= 40
    candidates = _candidate_grid(calibration, replay_risk)

    best_candidate: Candidate | None = None
    best_key: tuple[float, ...] | None = None
    best_development_metrics: dict[str, float | int] | None = None
    best_fold_metrics: list[dict[str, float | int]] = []
    for candidate in candidates:
        fold_metrics = [metrics(candidate_pnls(development.iloc[indexes], candidate, replay_risk)) for indexes in test_indexes]
        if any(result["trades"] < min_trades_per_fold for result in fold_metrics):
            continue
        aggregate = metrics(
            pnl
            for indexes in test_indexes
            for pnl in candidate_pnls(development.iloc[indexes], candidate, replay_risk)
        )
        key = (
            float(aggregate["win_rate"] >= target_win_rate),
            float(aggregate["total_pnl"] > 0),
            float(aggregate["win_rate"]),
            float(aggregate["profit_factor"]),
            -float(aggregate["max_drawdown"]),
            float(aggregate["trades"]),
        )
        if best_key is None or key > best_key:
            best_candidate = candidate
            best_key = key
            best_development_metrics = aggregate
            best_fold_metrics = fold_metrics

    if best_candidate is None or best_development_metrics is None:
        raise ValueError("no candidate met the minimum trade count in every walk-forward fold")

    holdout_metrics = metrics(candidate_pnls(holdout, best_candidate, replay_risk))
    minimum_holdout_trades = max(5, min_trades_per_fold)
    approved = bool(
        best_development_metrics["win_rate"] >= target_win_rate
        and best_development_metrics["total_pnl"] > 0
        and holdout_metrics["trades"] >= minimum_holdout_trades
        and holdout_metrics["win_rate"] >= target_win_rate
        and holdout_metrics["total_pnl"] > 0
        and holdout_metrics["profit_factor"] >= 1.10
    )
    report: dict[str, object] = {
        "approved": approved,
        "target_win_rate": target_win_rate,
        "total_rows": len(frame),
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "risk_replay_enabled": replay_risk,
        "risk_replay_rows": replayable_count,
        "intratrade_path_coverage": path_coverage,
        "atr_rows": int(frame["atr"].notna().sum()),
        "known_regime_rows": int(frame["regime"].isin(("TRENDING", "SIDEWAYS")).sum()),
        "candidate_count": len(candidates),
        "walk_forward": best_fold_metrics,
        "development": best_development_metrics,
        "holdout": holdout_metrics,
        "parameters": asdict(best_candidate),
    }
    return best_candidate, report, approved


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="CSV export; omit to read Google Sheet 1")
    parser.add_argument("--target-win-rate", type=float, default=0.60)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-trades-per-fold", type=int, default=5)
    parser.add_argument("--output", default=settings.adaptive_crude_config_file)
    parser.add_argument("--report", default="data/adaptive_crude_report.json")
    args = parser.parse_args()

    frame = load_trades(args.csv)
    candidate, report, approved = optimize(
        frame,
        target_win_rate=args.target_win_rate,
        folds=args.folds,
        min_trades_per_fold=args.min_trades_per_fold,
    )
    _atomic_json(Path(args.report), report)
    print(json.dumps(report, indent=2, default=str))
    if not approved:
        print("Deployment rejected: untouched holdout did not pass every safety gate.")
        return 2

    parameters = {
        "crude_atr_multiplier": candidate.atr_multiplier,
        "buffer_points": candidate.buffer_points,
        "crude_risk_reward_ratio": candidate.risk_reward_ratio,
        "crude_trailing_activation_points": candidate.trailing_activation_points,
        "crude_min_entry_volume": candidate.min_volume,
        "crude_min_abs_oi_change": candidate.min_abs_oi_change,
        "crude_min_entry_atr": candidate.min_atr,
        "crude_max_entry_atr": candidate.max_atr,
        "crude_allowed_regimes": candidate.allowed_regimes,
    }
    publish_payload(args.output, parameters, report)
    print(f"Approved config published atomically to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())