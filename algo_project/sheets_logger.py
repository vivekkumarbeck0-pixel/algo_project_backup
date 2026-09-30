import os
import json
import time
from pathlib import Path
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SHEET_COLUMNS = [
    "Entry Timestamp",
    "Exit Timestamp",
    "Symbol",
    "Action",
    "Entry Price",
    "Exit Price",
    "SL",
    "TP",
    "PnL",
    "Entry Scenario",
    "Exit Scenario",
    "Entry OI",
    "Exit OI",
    "Entry OI_Change",
    "Exit OI_Change",
    "Entry NYMEX_Trend",
    "Exit NYMEX_Trend",
    "Entry Volume",
    "Exit Volume",
    "Entry Index Value",
    "Entry Nearest Pivot",
    "Entry Pivot Number",
    "Entry Pivot Price",
    "Entry Candle Open",
    "Entry Candle High",
    "Entry Candle Low",
    "Entry Candle Close",
    "Exit Index Value",
    "Exit Nearest Pivot",
    "Exit Pivot Number",
    "Exit Pivot Price",
    "Exit Candle Open",
    "Exit Candle High",
    "Exit Candle Low",
    "Exit Candle Close",
    "Entry ATR",
    "Exit ATR",
    "Entry Market Regime",
    "Exit Market Regime",
    "Entry Momentum Strength",
    "Intratrade Option Prices",
    "Entry AI Score",
    "Entry AI Win Probability",
    "Entry AI Filter Mode",
]
def get_gspread_client():
    # Render cloud Environment check
    json_env = os.environ.get("GOOGLE_JSON_KEY")
    if json_env:
        info = json.loads(json_env)
        if "private_key" in info:
            info["private_key"] = info["private_key"].replace("\\n", "\n")
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return gspread.authorize(creds)
    
    # Local service_account.json loading
    credentials_path = Path(__file__).resolve().parent / "service_account.json"
    with credentials_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    if "private_key" in info:
        info["private_key"] = info["private_key"].replace("\\n", "\n")
        
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

def log_trade(trade_data: dict):
    max_attempts = 3
    base_delay_seconds = 2

    for attempt in range(1, max_attempts + 1):
        try:
            gc = get_gspread_client()
            sh = gc.open("Crude_Algo_Trade_Logs")
            worksheet = sh.sheet1

            # Check current rows in worksheet
            all_values = worksheet.get_all_values()

            headers = all_values[0] if all_values else []
            if not headers:
                headers = SHEET_COLUMNS.copy()
                worksheet.append_row(headers)
            else:
                # Extend only the header row; legacy columns, order, and rows stay intact.
                missing_headers = [header for header in SHEET_COLUMNS if header not in headers]
                if missing_headers:
                    headers = headers + missing_headers
                    worksheet.update("A1", [headers])

            row_values = [trade_data.get(header, "") for header in headers]
            worksheet.append_row(row_values)
            print(f"Successfully logged trade to Google Sheet on attempt {attempt}/{max_attempts}.")
            return True
        except APIError as exc:
            error_text = str(exc)
            if "503" not in error_text and "Service Unavailable" not in error_text:
                print(f"Failed to log trade to Google Sheet: {exc}")
                return False
            error = exc
        except Exception as exc:
            error_text = str(exc)
            if "503" not in error_text and "Service Unavailable" not in error_text:
                print(f"Failed to log trade to Google Sheet: {exc}")
                return False
            error = exc

        if attempt == max_attempts:
            print(f"Failed to log trade to Google Sheet after {max_attempts} attempts: {error}")
            return False

        delay = base_delay_seconds * (2 ** (attempt - 1))
        print(
            f"Google Sheets logging attempt {attempt}/{max_attempts} failed with 503; "
            f"retrying in {delay} seconds: {error}"
        )
        time.sleep(delay)