"""Train the offline Crude hybrid model.

Examples:
    py scripts/train_crude_hybrid.py data/crude_bars.csv
    py scripts/train_crude_hybrid.py "https://docs.google.com/spreadsheets/d/<id>/export?format=csv&gid=<gid>"
"""

import argparse

from engine.crude_hybrid_ml import load_crude_data, save_model, train_hybrid_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Crude momentum + volatility models")
    parser.add_argument("source", help="CSV path or Google Sheet CSV export URL")
    parser.add_argument("--output", default="data/crude_hybrid_model.joblib")
    args = parser.parse_args()

    frame = load_crude_data(args.source)
    model, metrics = train_hybrid_model(frame)
    save_model(model, args.output)
    print(f"Saved hybrid model: {args.output}")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
