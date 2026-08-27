import os
import json
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SHEET_COLUMNS = [
    "Timestamp",
    "Symbol",
    "Action",
    "Buy_Action",
    "Sell_Action",
    "Strike",
    "Entry_Price",
    "Exit_Price",
    "Entry_Order_ID",
    "Exit_Order_ID",
    "PnL",
    "OI",
    "OI_Change",
    "Entry_Vol",
    "Exit_Vol",
    "Vol",
    "Scenario",
    "Pivot_Level",
    "NYMEX_Trend",
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
            headers = [key for key in SHEET_COLUMNS if key in trade_data]
            headers.extend(key for key in trade_data if key not in headers)
            worksheet.append_row(headers)
        else:
            ordered_headers = [key for key in SHEET_COLUMNS if key in headers]
            ordered_headers.extend(key for key in headers if key not in ordered_headers)
            ordered_headers.extend(key for key in trade_data if key not in ordered_headers)
            if ordered_headers != headers:
                existing_rows = [
                    dict(zip(headers, row)) for row in all_values[1:]
                ]
                migrated_values = [
                    ordered_headers,
                    *([
                        [row.get(header, "") for header in ordered_headers]
                        for row in existing_rows
                    ]),
                ]
                worksheet.update("A1", migrated_values)
                headers = ordered_headers
            
        row_values = [trade_data.get(header, "") for header in headers]
        worksheet.append_row(row_values)
        print("Successfully logged trade to Google Sheet!")
    except Exception as e:
        print(f"Failed to log trade to Google Sheet: {e}")