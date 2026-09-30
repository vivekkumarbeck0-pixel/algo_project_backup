"""Outcome filter for veto-only Crude trade qualification."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

OUTCOME_FEATURES = (
    "side_pe",
    "volume",
    "oi_change_pct",
    "candle_return",
    "candle_range",
    "pivot_gap",
    "nymex_green",
    "regime_trending",
    "entry_atr",
    "momentum_strength",
)


def _metrics(pnls) -> dict[str, float | int]:
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


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def build_outcome_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build identical, entry-time-only features for training and live scoring."""
    features = pd.DataFrame(index=frame.index)
    symbol = frame.get("Symbol", pd.Series("", index=frame.index)).astype(str).str.upper()
    volume = _numeric_column(frame, "Entry Volume")
    oi = _numeric_column(frame, "Entry OI")
    oi_change = _numeric_column(frame, "Entry OI_Change")
    index_value = _numeric_column(frame, "Entry Index Value")
    pivot_price = _numeric_column(frame, "Entry Pivot Price")
    candle_open = _numeric_column(frame, "Entry Candle Open")
    candle_high = _numeric_column(frame, "Entry Candle High")
    candle_low = _numeric_column(frame, "Entry Candle Low")
    candle_close = _numeric_column(frame, "Entry Candle Close")

    features["side_pe"] = symbol.str.endswith("PE").astype(float)
    features["volume"] = volume
    features["oi_change_pct"] = oi_change / oi.replace(0, np.nan)
    features["candle_return"] = (candle_close - candle_open) / candle_open.replace(0, np.nan)
    features["candle_range"] = (candle_high - candle_low) / candle_close.replace(0, np.nan)
    features["pivot_gap"] = (index_value - pivot_price) / index_value.replace(0, np.nan)
    features["nymex_green"] = (
        frame.get("Entry NYMEX_Trend", pd.Series("", index=frame.index))
        .astype(str).str.upper().eq("GREEN").astype(float)
    )
    features["regime_trending"] = (
        frame.get("Entry Market Regime", pd.Series("", index=frame.index))
        .astype(str).str.upper().eq("TRENDING").astype(float)
    )
    features["entry_atr"] = _numeric_column(frame, "Entry ATR")
    features["momentum_strength"] = _numeric_column(frame, "Entry Momentum Strength")
    return features.replace([np.inf, -np.inf], np.nan).reindex(columns=OUTCOME_FEATURES)


def build_live_outcome_features(signal: dict[str, Any]) -> pd.DataFrame:
    """Convert a live rule-approved signal to the historical feature schema."""
    row = {
        "Symbol": f"CRUDEOIL{signal.get('side', '')}",
        "Entry Volume": signal.get("volume"),
        "Entry OI": signal.get("oi"),
        "Entry OI_Change": signal.get("oi_change"),
        "Entry Index Value": signal.get("futures_price"),
        "Entry Pivot Price": signal.get("entry_pivot_price", signal.get("pivot_level")),
        "Entry Candle Open": signal.get("candle_open"),
        "Entry Candle High": signal.get("candle_high"),
        "Entry Candle Low": signal.get("candle_low"),
        "Entry Candle Close": signal.get("candle_close"),
        "Entry NYMEX_Trend": signal.get("nymex_trend"),
        "Entry Market Regime": signal.get("market_regime"),
        "Entry ATR": signal.get("atr"),
        "Entry Momentum Strength": (signal.get("ai_evaluation") or {}).get("momentum_strength"),
    }
    return build_outcome_features(pd.DataFrame([row]))


def fit_outcome_filter(frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a small outcome classifier and approve veto use only on a holdout pass."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    required = {"Entry Timestamp", "Symbol", "PnL"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"outcome training data is missing columns: {missing}")

    features = build_outcome_features(frame)
    pnl = pd.to_numeric(frame["PnL"], errors="coerce")
    timestamp = pd.to_datetime(frame["Entry Timestamp"], errors="coerce")
    symbol = frame["Symbol"].astype(str).str.upper()
    eligible = (
        pnl.notna()
        & timestamp.notna()
        & symbol.str.endswith(("CE", "PE"))
        & features.notna().any(axis=1)
    )
    features = features.loc[eligible].reset_index(drop=True)
    pnl = pnl.loc[eligible].reset_index(drop=True)
    timestamp = timestamp.loc[eligible].reset_index(drop=True)
    order = np.argsort(timestamp.to_numpy(), kind="stable")
    features = features.iloc[order].reset_index(drop=True)
    pnl = pnl.iloc[order].reset_index(drop=True)

    if len(pnl) < 100:
        raise ValueError("at least 100 eligible Crude outcomes are required for a veto-filter backtest")
    train_end = int(len(pnl) * 0.60)
    validation_end = int(len(pnl) * 0.80)
    if train_end < 40 or validation_end - train_end < 10 or len(pnl) - validation_end < 10:
        raise ValueError("not enough chronological rows for 60/20/20 outcome-filter evaluation")

    estimator = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000, random_state=42),
    )
    labels = (pnl > 0).astype(int)
    if labels.iloc[:train_end].nunique() < 2:
        raise ValueError("outcome-filter training period must include both winning and losing trades")
    estimator.fit(features.iloc[:train_end], labels.iloc[:train_end])

    validation_probabilities = estimator.predict_proba(features.iloc[train_end:validation_end])[:, 1]
    validation_pnl = pnl.iloc[train_end:validation_end].reset_index(drop=True)
    minimum_validation_trades = max(10, int(math.ceil(len(validation_pnl) * 0.25)))
    maximum_validation_trades = int(len(validation_pnl) * 0.80)
    thresholds = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    eligible_thresholds = [
        threshold for threshold in thresholds
        if minimum_validation_trades <= int((validation_probabilities >= threshold).sum()) <= maximum_validation_trades
    ]
    threshold = max(
        eligible_thresholds,
        key=lambda value: _metrics(validation_pnl[validation_probabilities >= value])["total_pnl"],
    ) if eligible_thresholds else 0.5

    test_features = features.iloc[validation_end:].reset_index(drop=True)
    test_pnl = pnl.iloc[validation_end:].reset_index(drop=True)
    test_probabilities = estimator.predict_proba(test_features)[:, 1]
    selected_test = test_probabilities >= threshold
    baseline_metrics = _metrics(test_pnl)
    selected_metrics = _metrics(test_pnl[selected_test])
    minimum_test_trades = max(10, int(math.ceil(len(test_pnl) * 0.25)))
    approved = bool(
        int(selected_test.sum()) >= minimum_test_trades
        and selected_metrics["total_pnl"] > 0
        and selected_metrics["profit_factor"] >= 1.10
        and selected_metrics["max_drawdown"] < baseline_metrics["max_drawdown"]
    )

    # Keep the final test untouched: the shadow/live artifact is fit only on
    # data preceding that test period and cannot veto unless its holdout passes.
    report = {
        "approved_for_veto": approved,
        "rows": len(pnl),
        "train_rows": train_end,
        "validation_rows": validation_end - train_end,
        "test_rows": len(pnl) - validation_end,
        "threshold": threshold,
        "validation_selected": _metrics(validation_pnl[validation_probabilities >= threshold]),
        "test_baseline": baseline_metrics,
        "test_selected": selected_metrics,
        "test_retention": float(selected_test.mean()),
    }
    artifact = {
        "estimator": estimator,
        "feature_names": list(OUTCOME_FEATURES),
        "threshold": float(threshold),
        "approved_for_veto": approved,
        "report": report,
    }
    return artifact, report


def predict_win_probability(artifact: dict[str, Any], features: pd.DataFrame) -> float:
    estimator = artifact["estimator"]
    names = artifact.get("feature_names", list(OUTCOME_FEATURES))
    probability = float(estimator.predict_proba(features.reindex(columns=names))[:, 1][0])
    if not math.isfinite(probability):
        raise ValueError("outcome filter returned a non-finite probability")
    return probability