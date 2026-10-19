"""Manually train the offline MCX Crude hybrid model from Google Sheet 1.

Run only during off-market hours:
    py train_model.py

This module never imports trading_crude.py, starts a background job, writes to
Google Sheets, or changes live execution state.
"""

from __future__ import annotations

from pathlib import Path

from joblib import dump
import pandas as pd

from config import settings
from engine.crude_outcome_filter import fit_outcome_filter
from engine.crude_hybrid_ml import save_model, train_hybrid_model
from sheets_logger import get_gspread_client

MODEL_PATH = "crude_hybrid_model.pkl"

SHEET1_HEADERS = [
    "Entry Timestamp", "Exit Timestamp", "Symbol", "Action", "Entry Price",
    "Exit Price", "SL", "TP", "PnL", "Entry Scenario", "Exit Scenario",
    "Entry OI", "Exit OI", "Entry OI_Change", "Exit OI_Change",
    "Entry NYMEX_Trend", "Exit NYMEX_Trend", "Entry Volume", "Exit Volume",
    "Entry Index Value", "Entry Nearest Pivot", "Entry Pivot Number",
    "Entry Pivot Price", "Entry Candle Open", "Entry Candle High",
    "Entry Candle Low", "Entry Candle Close", "Exit Index Value",
    "Exit Nearest Pivot", "Exit Pivot Number", "Exit Pivot Price",
    "Exit Candle Open", "Exit Candle High", "Exit Candle Low", "Exit Candle Close",
]

NUMERIC_SHEET1_COLUMNS = [
    "Entry Price", "Exit Price", "SL", "TP", "PnL", "Entry OI", "Exit OI",
    "Entry OI_Change", "Exit OI_Change", "Entry Volume", "Exit Volume",
    "Entry Index Value", "Entry Pivot Number", "Entry Pivot Price",
    "Entry Candle Open", "Entry Candle High", "Entry Candle Low",
    "Entry Candle Close", "Exit Index Value", "Exit Pivot Number",
    "Exit Pivot Price", "Exit Candle Open", "Exit Candle High",
    "Exit Candle Low", "Exit Candle Close",
]


def fetch_sheet1_rows() -> pd.DataFrame:
    """Read and clean every populated Crude row from Sheet 1.

    ``get_all_values`` is intentional: it retrieves the complete worksheet
    matrix and does not apply a record/page slice. Sheet 1 stores the full
    option symbol, such as ``CRUDEOIL...CE``, rather than only ``CRUDEOIL``.
    """
    worksheet = get_gspread_client().open("Crude_Algo_Trade_Logs").sheet1
    values = worksheet.get_all_values()
    if len(values) < 2:
        raise ValueError("Sheet 1 contains no trade rows to train on")

    headers = [str(header).strip() for header in values[0]]
    required_headers = {
        "Symbol", "Entry Timestamp", "Entry Candle Open", "Entry Candle High",
        "Entry Candle Low", "Entry Candle Close", "Entry Volume", "Entry OI",
    }
    missing_headers = sorted(required_headers - set(headers))
    if missing_headers:
        raise ValueError(f"Sheet 1 is missing required headers: {missing_headers}")

    # Filter the raw worksheet matrix before constructing the frame. This
    # avoids pandas view-backed assignment and keeps all columns object-backed.
    symbol_index = headers.index("Symbol")
    crude_rows = [
        row for row in values[1:]
        if symbol_index < len(row)
        and "CRUDE" in str(row[symbol_index]).upper().strip()
    ]
    frame = pd.DataFrame(crude_rows, columns=headers, dtype=object).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Sheet 1 contains no Crude symbols")

    converted_columns = {
        column: pd.to_numeric(frame[column], errors="coerce")
        for column in NUMERIC_SHEET1_COLUMNS
        if column in frame.columns
    }
    frame = pd.DataFrame({**frame.to_dict(orient="series"), **converted_columns})
    for column in ("open", "high", "low", "close", "volume", "oi"):
        source_column = {
            "open": "Entry Candle Open", "high": "Entry Candle High",
            "low": "Entry Candle Low", "close": "Entry Candle Close",
            "volume": "Entry Volume", "oi": "Entry OI",
        }[column]
        frame = frame.assign(**{column: frame[source_column].astype(float)})
    frame = frame.assign(timestamp=pd.to_datetime(frame["Entry Timestamp"], errors="coerce"))
    required_training_columns = [
        "timestamp", "open", "high", "low", "close", "volume", "oi",
    ]
    valid_numeric_columns = [column for column in NUMERIC_SHEET1_COLUMNS if column in frame.columns]
    frame = frame.dropna(
        subset=required_training_columns + valid_numeric_columns
    ).sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Crude rows were found, but none contain valid entry OHLC data")
    return frame


def main() -> int:
    frame = fetch_sheet1_rows()
    print(f"Loaded {len(frame)} Crude rows from Google Sheet 1.")

    model, metrics = train_hybrid_model(frame)
    save_model(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print("Chronological split: 75% train / 25% test; shuffle disabled")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    outcome_filter, filter_report = fit_outcome_filter(frame)
    filter_path = Path(settings.crude_ai_filter_model_file)
    filter_path.parent.mkdir(parents=True, exist_ok=True)
    dump(outcome_filter, filter_path)
    print(f"Saved outcome filter: {filter_path}")
    print("Outcome filter chronological backtest:")
    for name, value in filter_report.items():
        print(f"{name}: {value}")
    if not filter_report["approved_for_veto"]:
        print("VETO mode remains inactive until a later holdout passes all approval gates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
