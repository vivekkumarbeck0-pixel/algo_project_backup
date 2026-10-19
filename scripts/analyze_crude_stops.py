"""Read-only breakdown of Crude Sheet 1 stop-loss exits by entry context."""

from __future__ import annotations

import pandas as pd

from config import settings
from scripts.optimize_crude_parameters import _parse_price_path, metrics
from train_model import fetch_sheet1_rows


def summarize(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    summary = frame.groupby(group, dropna=False).agg(
        trades=("pnl", "size"),
        stops=("stop_loss", "sum"),
        trailing_stops=("trailing_stop", "sum"),
        targets=("target_hit", "sum"),
        stop_loss_pnl=("stop_pnl", "sum"),
        total_pnl=("pnl", "sum"),
    )
    return summary.assign(
        stop_rate_pct=(100 * summary["stops"] / summary["trades"]).round(1)
    ).sort_values("stop_loss_pnl").round(2)


def compare_break_even(frame: pd.DataFrame) -> None:
    replayable = frame.assign(
        path=frame["Intratrade Option Prices"].map(_parse_price_path),
        entry=pd.to_numeric(frame["Entry Price"], errors="coerce"),
        exit=pd.to_numeric(frame["Exit Price"], errors="coerce"),
    )
    replayable = replayable.loc[replayable["path"].map(bool)].dropna(subset=["entry", "exit"]).copy()
    activation = float(settings.crude_trailing_activation_points)
    buffer = max(0.0, float(settings.crude_trailing_breakeven_buffer_points))
    print(f"\nEntry +{buffer:g} floor after +{activation:g} premium points (observed-price fills only):")
    for period, subset in (
        ("all replayable", replayable),
        ("earlier replayable", replayable.loc[replayable.index < len(frame) - 100]),
        ("last 100 trades", replayable.loc[replayable.index >= len(frame) - 100]),
    ):
        changed = []
        hypothetical = []
        for index, row in subset.iterrows():
            path = row["path"]
            activated = False
            exit_price = row["exit"]
            for price in path[:-1]:
                if price >= row["entry"] + activation:
                    activated = True
                if activated and price <= row["entry"] + buffer:
                    exit_price = price
                    changed.append(index)
                    break
            hypothetical.append(row["pnl"] + (exit_price - row["exit"]) * 100)
        changed_rows = subset.loc[changed]
        print(
            f"{period}: paths={len(subset)}, earlier exits={len(changed_rows)}, "
            f"original SL={int(changed_rows['stop_loss'].sum())}, "
            f"original trailing={int(changed_rows['trailing_stop'].sum())}, "
            f"original targets={int(changed_rows['target_hit'].sum())}"
        )
        print("  recorded:", metrics(subset["pnl"]))
        print("  hypothetical:", metrics(hypothetical))
        if changed:
            changed_pnl = pd.Series(hypothetical, index=subset.index).loc[changed]
            print(
                f"  earlier-exit PnL: recorded={changed_rows['pnl'].sum():.2f}, "
                f"hypothetical={changed_pnl.sum():.2f}"
            )
    print("This counterfactual uses only recorded paths; missing paths, spreads, fees and changed future entries are not modeled.")


def main() -> int:
    frame = fetch_sheet1_rows()
    frame = frame.assign(pnl=pd.to_numeric(frame["PnL"], errors="coerce"))
    frame = frame.dropna(subset=["timestamp", "pnl"]).copy()
    reason = frame["Exit Scenario"].astype(str).str.upper().str.strip()
    stop_loss = reason.eq("STOP LOSS")
    frame = frame.assign(
        stop_loss=stop_loss,
        trailing_stop=reason.eq("TRAILING STOP"),
        target_hit=reason.isin(("TARGET HIT", "S&R TARGET HIT")),
        stop_pnl=frame["pnl"].where(stop_loss, 0.0),
        entry_hour=frame["timestamp"].dt.hour,
        side=frame["Action"].astype(str).str.upper().str.strip(),
        scenario=frame["Entry Scenario"].astype(str).str.strip().replace("", "UNKNOWN"),
        regime=frame["Entry Market Regime"].astype(str).str.upper().str.strip().replace("", "UNKNOWN"),
    )

    print(f"Trades: {len(frame)}, recorded total PnL: {frame['pnl'].sum():.2f}")
    print("STOP LOSS means the recorded exit reason, not every negative-PnL trade.")
    for group in ("entry_hour", "side", "scenario", "regime"):
        print(f"\nBy {group} (sorted by total PnL on STOP LOSS exits):")
        print(summarize(frame, group).to_string())
    split = int(len(frame) * 0.75)
    for period, subset in (("first 75%", frame.iloc[:split]), ("last 25%", frame.iloc[split:])):
        for group in ("scenario", "entry_hour"):
            print(f"\n{period} by {group} (chronological, descriptive only):")
            print(summarize(subset, group).to_string())
        for name, skipped in (
            ("Short Buildup", subset["scenario"].eq("Short Buildup")),
            ("entry hour 19", subset["entry_hour"].eq(19)),
        ):
            rejected = subset.loc[skipped]
            retained = subset.loc[~skipped]
            print(
                f"{period} hypothetical skip {name}: "
                f"skipped={len(rejected)}, SL missed={rejected['stop_loss'].sum()}, "
                f"targets missed={rejected['target_hit'].sum()}, "
                f"skipped PnL={rejected['pnl'].sum():.2f}, "
                f"retained trades={len(retained)}, retained PnL={retained['pnl'].sum():.2f}"
            )
    print("These are retrospective skips, not proof of future profitability or approval for live filtering.")
    compare_break_even(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())