"""Backtest every crude parameter candidate on the complete trade dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.optimize_crude_parameters import (
    _candidate_grid,
    candidate_pnls,
    load_trades,
    metrics,
)


def _selected_rows(frame: pd.DataFrame, candidate) -> pd.DataFrame:
    atr_gate_enabled = candidate.min_atr > 0.0 or candidate.max_atr < 1_000_000.0
    atr_matches = frame["atr"].between(candidate.min_atr, candidate.max_atr)
    if not atr_gate_enabled:
        atr_matches = pd.Series(True, index=frame.index)
    return frame[
        (frame["volume"] >= candidate.min_volume)
        & (frame["oi_change"].abs() >= candidate.min_abs_oi_change)
        & atr_matches
        & frame["regime"].isin(candidate.allowed_regimes)
    ]


def _daily_comparison(frame: pd.DataFrame, candidate, replay_risk: bool) -> dict[str, object]:
    selected = _selected_rows(frame, candidate)
    all_daily_pnl = frame.groupby(frame["timestamp"].dt.date)["pnl"].sum()
    selected_daily_pnl = selected.groupby(selected["timestamp"].dt.date)["pnl"].sum()
    selected_pnls = candidate_pnls(frame, candidate, replay_risk)
    raw_selected_pnl = selected["pnl"].astype(float).tolist()
    return {
        "all_data": {
            "trades": len(frame),
            "days": int(frame["timestamp"].dt.date.nunique()),
            "average_trades_per_day": float(len(frame) / frame["timestamp"].dt.date.nunique()),
            "total_pnl": float(frame["pnl"].sum()),
            "average_daily_pnl": float(all_daily_pnl.mean()),
        },
        "selected": {
            "trades": len(selected_pnls),
            "days": int(selected["timestamp"].dt.date.nunique()),
            "average_trades_per_day": float(
                len(selected_pnls) / selected["timestamp"].dt.date.nunique()
            ) if len(selected) else 0.0,
            "trade_retention": float(len(selected) / len(frame)),
            "total_pnl": float(sum(raw_selected_pnl)),
            "average_daily_pnl": float(selected_daily_pnl.mean()) if len(selected) else 0.0,
        },
        "replayed_selected": {
            "trades": len(selected_pnls),
            "total_pnl": float(sum(selected_pnls)),
            "win_rate": float(metrics(selected_pnls)["win_rate"]),
            "profit_factor": float(metrics(selected_pnls)["profit_factor"]),
        },
    }


def run_full_backtest(
    frame: pd.DataFrame,
    target_win_rate: float = 0.50,
    tolerance: float = 0.05,
    min_trades: int = 20,
) -> list[dict[str, object]]:
    """Return candidates close to the requested win-rate on all rows.

    This is an in-sample diagnostic. It must not be used as deployment approval.
    """
    replayable = (
        frame["price_path"].map(bool)
        & frame["entry_price"].notna()
        & frame["atr"].notna()
    )
    replay_risk = int(replayable.sum()) >= 40
    candidates = _candidate_grid(frame, replay_risk)
    results: list[dict[str, object]] = []
    for candidate in candidates:
        result = metrics(candidate_pnls(frame, candidate, replay_risk))
        trades = int(result["trades"])
        if trades < min_trades:
            continue
        result["win_rate_gap"] = abs(float(result["win_rate"]) - target_win_rate)
        result["within_tolerance"] = result["win_rate_gap"] <= tolerance
        result["daily_comparison"] = _daily_comparison(frame, candidate, replay_risk)
        result["parameters"] = {
            "min_volume": candidate.min_volume,
            "min_abs_oi_change": candidate.min_abs_oi_change,
            "min_atr": candidate.min_atr,
            "max_atr": candidate.max_atr,
            "allowed_regimes": candidate.allowed_regimes,
            "atr_multiplier": candidate.atr_multiplier,
            "buffer_points": candidate.buffer_points,
            "risk_reward_ratio": candidate.risk_reward_ratio,
            "trailing_activation_points": candidate.trailing_activation_points,
        }
        results.append(result)

    results.sort(
        key=lambda item: (
            not bool(item["within_tolerance"]),
            float(item["win_rate_gap"]),
            -float(item["total_pnl"]),
            -float(item["profit_factor"]),
        )
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="CSV export; omit to read Google Sheet 1")
    parser.add_argument("--target-win-rate", type=float, default=0.50)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", default="data/crude_full_backtest.json")
    args = parser.parse_args()

    frame = load_trades(args.csv)
    results = run_full_backtest(
        frame,
        target_win_rate=args.target_win_rate,
        tolerance=args.tolerance,
        min_trades=args.min_trades,
    )
    payload = {
        "warning": "In-sample diagnostic only; do not deploy from this report.",
        "total_rows": len(frame),
        "target_win_rate": args.target_win_rate,
        "tolerance": args.tolerance,
        "min_trades": args.min_trades,
        "matching_candidates": sum(bool(item["within_tolerance"]) for item in results),
        "results": results[:args.top],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    print(f"Saved full-data backtest to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())