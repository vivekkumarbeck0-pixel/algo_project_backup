import os
import json
import gspread
from google.oauth2.service_account import Credentials

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
    with open("service_account.json", "r") as f:
        info = json.load(f)
    if "private_key" in info:
        info["private_key"] = info["private_key"].replace("\\n", "\n")
        
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

def log_trade(trade_data: dict):
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
        print("Successfully logged trade to Google Sheet!")
    except Exception as e:
        print(f"Failed to log trade to Google Sheet: {e}")