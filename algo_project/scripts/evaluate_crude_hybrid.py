"""Evaluate the saved Crude model against recorded trades, without placing orders."""

from __future__ import annotations

import pandas as pd

from engine.crude_hybrid_ml import FEATURE_COLUMNS, build_features, build_labels, load_model
from scripts.optimize_crude_parameters import metrics
from train_model import MODEL_PATH, fetch_sheet1_rows


def evaluate_entry_outcome(frame: pd.DataFrame) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    columns = {
        "Entry Volume": "volume",
        "Entry OI": "oi",
        "Entry OI_Change": "oi_change",
        "Entry Index Value": "index_value",
        "Entry Pivot Price": "pivot_price",
        "Entry Candle Open": "open",
        "Entry Candle High": "high",
        "Entry Candle Low": "low",
        "Entry Candle Close": "close",
    }
    values = pd.DataFrame({name: pd.to_numeric(frame[source], errors="coerce") for source, name in columns.items()})
    features = pd.DataFrame(index=frame.index)
    features["side_pe"] = frame["Symbol"].str.upper().str.endswith("PE").astype(int)
    features["volume"] = values["volume"]
    features["oi_change_pct"] = values["oi_change"] / values["oi"].replace(0, float("nan"))
    features["candle_return"] = (values["close"] - values["open"]) / values["open"].replace(0, float("nan"))
    features["candle_range"] = (values["high"] - values["low"]) / values["close"].replace(0, float("nan"))
    features["pivot_gap"] = (values["index_value"] - values["pivot_price"]) / values["index_value"].replace(0, float("nan"))
    features["nymex_green"] = frame["Entry NYMEX_Trend"].astype(str).str.upper().eq("GREEN").astype(int)
    features = features.replace([float("inf"), -float("inf")], float("nan"))

    pnl = pd.to_numeric(frame["PnL"], errors="coerce")
    eligible = pnl.notna() & features.notna().any(axis=1)
    features = features.loc[eligible]
    pnl = pnl.loc[eligible]
    train_end = int(len(features) * 0.60)
    validation_end = int(len(features) * 0.80)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=250, min_samples_leaf=10, max_depth=4, class_weight="balanced", random_state=42),
    )
    model.fit(features.iloc[:train_end], (pnl.iloc[:train_end] > 0).astype(int))
    validation_probabilities = model.predict_proba(features.iloc[train_end:validation_end])[:, 1]
    validation_pnl = pnl.iloc[train_end:validation_end]
    thresholds = (0.5, 0.6, 0.7)
    eligible_thresholds = [
        threshold for threshold in thresholds
        if (validation_probabilities >= threshold).sum() >= 10
    ]
    threshold = max(
        eligible_thresholds,
        key=lambda value: metrics(validation_pnl[validation_probabilities >= value])["total_pnl"],
    ) if eligible_thresholds else 0.5
    test_probabilities = model.predict_proba(features.iloc[validation_end:])[:, 1]
    test_pnl = pnl.iloc[validation_end:]
    print("Entry-only candidate: 60% train / 20% threshold selection / 20% final test")
    print("Validation baseline:", metrics(validation_pnl))
    print(f"Validation selected (threshold={threshold}):", metrics(validation_pnl[validation_probabilities >= threshold]))
    print("Final test baseline:", metrics(test_pnl))
    print("Final test selected:", metrics(test_pnl[test_probabilities >= threshold]))


def main() -> int:
    frame = fetch_sheet1_rows()
    features = build_features(frame)
    labels = build_labels(frame, features)
    dataset = pd.concat([features, labels], axis=1).dropna()
    momentum_holdout = dataset.iloc[int(len(dataset) * 0.75):]
    pnl = pd.to_numeric(frame["PnL"], errors="coerce")
    outcome_dataset = pd.concat(
        [features, (pnl > 0).astype(float).where(pnl.notna()).rename("outcome_label")], axis=1
    ).dropna()
    outcome_holdout = outcome_dataset.iloc[int(len(outcome_dataset) * 0.75):]

    model = load_model(MODEL_PATH)
    print("Existing model, untouched chronological holdouts (recorded PnL units):")
    print("Momentum/volatility cohort:", metrics(pnl.loc[momentum_holdout.index]))
    selected = []
    for index, row in momentum_holdout.iterrows():
        decision = model.decide(row.to_frame().T[FEATURE_COLUMNS])
        if decision.action == frame.loc[index, "Symbol"].upper()[-2:]:
            selected.append(index)
    print("Matching directional decision:", metrics(pnl.loc[selected]))

    print("Outcome cohort:", metrics(pnl.loc[outcome_holdout.index]))
    if model.outcome_model is not None:
        probabilities = model.outcome_model.predict_proba(outcome_holdout[FEATURE_COLUMNS])[:, 1]
        for threshold in (0.5, 0.6):
            selected = outcome_holdout.index[probabilities >= threshold]
            print(f"Outcome probability >= {threshold}:", metrics(pnl.loc[selected]))
    evaluate_entry_outcome(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())