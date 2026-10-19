"""Train the optional signal probability model from collected paper-trade data.

Run on a closed market day:
    .venv\Scripts\python.exe scripts\train_probability_model.py
"""

import json
from pathlib import Path

DATA_PATH = Path("data/training_signals.jsonl")
MODEL_PATH = Path("data/signal_probability_model.joblib")


def load_rows():
    if not DATA_PATH.exists():
        return []
    rows = []
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("outcome") in ("WIN", "LOSS"):
            rows.append(row)
    return rows


def features(row):
    trend = str(row.get("trend") or "").upper()
    micro = str(row.get("micro_momentum") or "").upper()
    symbol = str(row.get("symbol") or "").upper()
    spot = float(row.get("spot") or 0)
    support = float(row.get("support") or spot)
    resistance = float(row.get("resistance") or spot)
    return [
        1.0 if symbol == "NIFTY" else 0.0,
        1.0 if symbol == "CRUDEOIL" else 0.0,
        spot - support,
        resistance - spot,
        1.0 if trend == "BULLISH" else 0.0,
        1.0 if trend == "BEARISH" else 0.0,
        1.0 if micro == "BULLISH" else 0.0,
        1.0 if micro == "BEARISH" else 0.0,
        1.0 if row.get("entry_confirmed") else 0.0,
        1.0 if row.get("aoc_color_available") else 0.0,
        1.0 if row.get("option_type") == "CE" else 0.0,
        1.0 if row.get("option_type") == "PE" else 0.0,
        1.0 if row.get("risk_approved") else 0.0,
    ]


def main():
    rows = load_rows()
    if len(rows) < 50:
        print(f"Only {len(rows)} labelled trades found. Collect at least 50; 200+ is preferable.")
        return 2

    try:
        from joblib import dump
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Training dependencies missing. Install: pip install scikit-learn joblib")
        return 2

    x = [features(row) for row in rows]
    y = [1 if row["outcome"] == "WIN" else 0 for row in rows]
    if len(set(y)) < 2:
        print("Need both WIN and LOSS outcomes before training.")
        return 2

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = [1 if value >= 0.5 else 0 for value in probabilities]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Rows: {len(rows)} | accuracy: {accuracy_score(y_test, predictions):.3f} | ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print("Use probability >= 0.75 only after out-of-sample performance is stable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
