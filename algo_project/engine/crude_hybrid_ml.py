"""Offline hybrid ML research components for MCX Crude Oil.

This module has no import path from trading_crude.py. Train and evaluate it
locally before considering a separately reviewed live integration.

Expected input columns (case-insensitive aliases are supported):
    timestamp, open, high, low, close, volume, oi
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "sma_gap",
    "breakout_position",
    "atr_pct",
    "range_pct",
    "volatility_10",
    "volume_ratio",
    "oi_change_pct",
    "trade_side",
    "entry_atr_pct",
    "entry_regime_code",
    "entry_nymex_code",
    "entry_pivot_gap_pct",
]


def load_crude_data(source: str | Path) -> pd.DataFrame:
    """Load CSV data, including a Google Sheet CSV export URL."""
    frame = pd.read_csv(source)
    aliases = {
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
        "ltp": "close",
        "last": "close",
        "open_interest": "oi",
        "openinterest": "oi",
    }
    frame = frame.rename(columns={column: aliases.get(str(column).lower(), str(column).lower()) for column in frame.columns})
    required = {"open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Crude data is missing required columns: {sorted(missing)}")
    for column in ("open", "high", "low", "close", "volume", "oi"):
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.sort_values("timestamp")
    frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if len(frame) < 40:
        raise ValueError("At least 40 clean Crude bars are required for features and labels")
    return frame


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal momentum, volatility, volume, and OI features."""
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float).clip(lower=0)
    oi = frame["oi"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    rolling_high = high.shift(1).rolling(20, min_periods=10).max()
    rolling_low = low.shift(1).rolling(20, min_periods=10).min()
    average_volume = volume.shift(1).rolling(20, min_periods=10).mean()

    features = pd.DataFrame(index=frame.index)
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["sma_gap"] = close / close.rolling(10, min_periods=10).mean() - 1.0
    features["breakout_position"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
    features["atr_pct"] = true_range.rolling(14, min_periods=10).mean() / close
    features["range_pct"] = (high - low) / close
    features["volatility_10"] = close.pct_change().rolling(10, min_periods=10).std()
    features["volume_ratio"] = volume / average_volume.replace(0, np.nan)
    features["oi_change_pct"] = oi.pct_change().replace([np.inf, -np.inf], np.nan)
    side = frame.get("trade_side", pd.Series(0.0, index=frame.index))
    features["trade_side"] = pd.to_numeric(side, errors="coerce").fillna(0.0)
    entry_atr = pd.to_numeric(
        frame.get("Entry ATR", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    entry_atr = entry_atr.fillna(true_range.rolling(14, min_periods=10).mean())
    features["entry_atr_pct"] = entry_atr / close
    regime = frame.get(
        "Entry Market Regime", pd.Series("UNKNOWN", index=frame.index)
    ).astype(str).str.upper()
    features["entry_regime_code"] = regime.map({"TRENDING": 1.0, "SIDEWAYS": -1.0}).fillna(0.0)
    nymex = frame.get(
        "Entry NYMEX_Trend", pd.Series("UNKNOWN", index=frame.index)
    ).astype(str).str.upper()
    features["entry_nymex_code"] = nymex.map({"GREEN": 1.0, "RED": -1.0}).fillna(0.0)
    index_value = pd.to_numeric(
        frame.get("Entry Index Value", close), errors="coerce"
    )
    nearest_pivot = pd.to_numeric(
        frame.get("Entry Nearest Pivot", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    pivot_price = pd.to_numeric(
        frame.get("Entry Pivot Price", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    nearest_pivot = nearest_pivot.fillna(pivot_price).fillna(index_value)
    features["entry_pivot_gap_pct"] = (
        (index_value - nearest_pivot).abs() / index_value.abs().replace(0, np.nan)
    )
    return features.replace([np.inf, -np.inf], np.nan)


def build_labels(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Create outcome labels from exits when available, otherwise next-bar labels."""
    close = frame["close"].astype(float)
    next_return = close.shift(-1) / close - 1.0
    volatility_baseline = features["volatility_10"].rolling(50, min_periods=20).median()
    labels = pd.DataFrame(index=frame.index)
    labels["momentum_label"] = (next_return > 0).astype(float)
    pnl = pd.to_numeric(
        frame.get("PnL", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    symbols = frame.get("Symbol", pd.Series("", index=frame.index)).astype(str).str.upper()
    has_trade_outcome = pnl.notna() & symbols.str.endswith(("CE", "PE"))
    call_side = symbols.str.endswith("CE")
    labels.loc[has_trade_outcome, "momentum_label"] = (
        ((pnl > 0) == call_side).astype(float).loc[has_trade_outcome]
    )
    labels["volatility_label"] = (
        next_return.abs() > volatility_baseline.fillna(features["volatility_10"])
    ).astype(int)
    labels.loc[next_return.isna(), :] = np.nan
    return labels


@dataclass
class HybridDecision:
    action: Literal["CE", "PE", "NO_TRADE"]
    momentum_probability: float
    volatility_probability: float
    volatility_regime: str
    reason: str
    outcome_probability: float | None = None


class CrudeHybridModel:
    """Two-model ensemble: direction plus volatility regime."""

    def __init__(self, momentum_model, volatility_model, outcome_model=None, feature_columns: list[str] | None = None):
        self.momentum_model = momentum_model
        self.volatility_model = volatility_model
        self.outcome_model = outcome_model
        self.feature_columns = feature_columns or FEATURE_COLUMNS.copy()

    def decide(self, feature_row: pd.DataFrame, confidence_threshold: float = 0.60) -> HybridDecision:
        values = feature_row[self.feature_columns]
        momentum_probability = float(self.momentum_model.predict_proba(values)[0, 1])
        volatility_probability = float(self.volatility_model.predict_proba(values)[0, 1])
        outcome_probability = None
        if self.outcome_model is not None:
            outcome_probability = float(self.outcome_model.predict_proba(values)[0, 1])
        high_volatility = volatility_probability >= 0.50
        if high_volatility:
            return HybridDecision(
                "NO_TRADE", momentum_probability, volatility_probability, "HIGH",
                "high-volatility filter is active", outcome_probability,
            )
        if momentum_probability >= confidence_threshold:
            return HybridDecision(
                "CE", momentum_probability, volatility_probability, "NORMAL",
                "bullish momentum confidence passed", outcome_probability,
            )
        if momentum_probability <= 1.0 - confidence_threshold:
            return HybridDecision(
                "PE", momentum_probability, volatility_probability, "NORMAL",
                "bearish momentum confidence passed", outcome_probability,
            )
        return HybridDecision(
            "NO_TRADE", momentum_probability, volatility_probability, "NORMAL",
            "momentum confidence is inconclusive", outcome_probability,
        )


def train_hybrid_model(frame: pd.DataFrame, random_state: int = 42):
    """Train both models on an ordered 75/25 split and return metrics."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    features = build_features(frame)
    labels = build_labels(frame, features)
    dataset = pd.concat([features, labels], axis=1).dropna()
    if len(dataset) < 40:
        raise ValueError("Not enough complete feature/label rows after warm-up")
    split = int(len(dataset) * 0.75)
    if split <= 0 or split >= len(dataset):
        raise ValueError("Unable to create a chronological train/test split")
    train = dataset.iloc[:split]
    test = dataset.iloc[split:]
    for label_name in ("momentum_label", "volatility_label"):
        if train[label_name].nunique() < 2:
            raise ValueError(
                f"Training data has only one {label_name} class; collect a wider market sample"
            )
    momentum_model = RandomForestClassifier(n_estimators=250, min_samples_leaf=5, class_weight="balanced", random_state=random_state)
    volatility_model = RandomForestClassifier(n_estimators=250, min_samples_leaf=5, class_weight="balanced", random_state=random_state)
    momentum_model.fit(train[FEATURE_COLUMNS], train["momentum_label"].astype(int))
    volatility_model.fit(train[FEATURE_COLUMNS], train["volatility_label"].astype(int))
    momentum_probability = momentum_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    volatility_probability = volatility_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    outcome_model = None
    outcome_metrics = {}
    if "PnL" in frame.columns:
        pnl = pd.to_numeric(frame["PnL"], errors="coerce")
        outcome_labels = (pnl > 0).astype(float).where(pnl.notna())
        outcome_dataset = pd.concat([features, outcome_labels.rename("outcome_label")], axis=1).dropna()
        outcome_split = int(len(outcome_dataset) * 0.75)
        if outcome_split > 0 and outcome_split < len(outcome_dataset):
            outcome_train = outcome_dataset.iloc[:outcome_split]
            outcome_test = outcome_dataset.iloc[outcome_split:]
            if outcome_train["outcome_label"].nunique() >= 2:
                outcome_model = RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=random_state,
                )
                outcome_model.fit(outcome_train[FEATURE_COLUMNS], outcome_train["outcome_label"].astype(int))
                outcome_probability = outcome_model.predict_proba(outcome_test[FEATURE_COLUMNS])[:, 1]
                outcome_metrics = {
                    "outcome_rows": len(outcome_dataset),
                    "outcome_train_rows": len(outcome_train),
                    "outcome_test_rows": len(outcome_test),
                    "outcome_accuracy": accuracy_score(outcome_test["outcome_label"], outcome_probability >= 0.5),
                    "outcome_roc_auc": roc_auc_score(outcome_test["outcome_label"], outcome_probability)
                    if outcome_test["outcome_label"].nunique() > 1 else None,
                }

    metrics = {
        "rows": len(dataset),
        "train_rows": len(train),
        "test_rows": len(test),
        "momentum_accuracy": accuracy_score(test["momentum_label"], momentum_probability >= 0.5),
        "volatility_accuracy": accuracy_score(test["volatility_label"], volatility_probability >= 0.5),
        "momentum_roc_auc": roc_auc_score(test["momentum_label"], momentum_probability) if test["momentum_label"].nunique() > 1 else None,
        "volatility_roc_auc": roc_auc_score(test["volatility_label"], volatility_probability) if test["volatility_label"].nunique() > 1 else None,
        **outcome_metrics,
    }
    return CrudeHybridModel(momentum_model, volatility_model, outcome_model), metrics


def save_model(model: CrudeHybridModel, path: str | Path) -> None:
    from joblib import dump

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)


def load_model(path: str | Path) -> CrudeHybridModel:
    from joblib import load

    return load(path)
