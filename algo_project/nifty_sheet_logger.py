"""Google Sheets logger for NIFTY entries on worksheet 2 only."""

import json
import os

import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

NIFTY_COLUMNS = [
    "Timestamp", "Symbol", "Action", "Execution Price (LTP)",
    "Candle Open", "Candle High", "Candle Low", "Candle Close",
    "Pivot / SR Level", "Strike", "Option Side", "CE OI", "CE OI Change",
    "PE OI", "PE OI Change", "Strike Context JSON", "Target", "Stop Loss",
]


def _client():
    json_env = os.environ.get("GOOGLE_JSON_KEY")
    if json_env:
        info = json.loads(json_env)
    else:
        with open("service_account.json", "r", encoding="utf-8") as handle:
            info = json.load(handle)
    if "private_key" in info:
        info["private_key"] = info["private_key"].replace("\\n", "\n")
    return gspread.authorize(Credentials.from_service_account_info(info, scopes=SCOPES))


def log_nifty_entry(entry: dict) -> None:
    """Append one NIFTY entry to the second tab without touching worksheet 1."""
    try:
        spreadsheet = _client().open("Crude_Algo_Trade_Logs")
        worksheet = spreadsheet.worksheets()[1]
        existing = worksheet.get_all_values()
        headers = existing[0] if existing else []
        if not headers:
            headers = NIFTY_COLUMNS.copy()
            worksheet.append_row(headers)
        missing = [column for column in NIFTY_COLUMNS if column not in headers]
        if missing:
            headers = headers + missing
            worksheet.update("A1", [headers])
        worksheet.append_row([entry.get(column, "") for column in headers])
        print("Successfully logged NIFTY entry to Google Sheet 2!")
    except Exception as exc:
        print(f"Failed to log NIFTY entry to Google Sheet 2: {exc}")