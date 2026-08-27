import os
import json
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
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
        
        # 1. Headers auto-create agar sheet bilkul empty hai
        headers = all_values[0] if all_values else []
        if not headers:
            headers = list(trade_data.keys())
            worksheet.append_row(headers)
        else:
            # 2. Dynamic Header sync (agar nayi column jaise NYMEX_Trend add hoti hai)
            for key in trade_data.keys():
                if key not in headers:
                    headers.append(key)
            if headers != all_values[0]:
                worksheet.update('1:1', [headers])
            
        # Append trade data values in order
        row_values = [trade_data.get(header, "") for header in headers]
        worksheet.append_row(row_values)
        print("Successfully logged trade to Google Sheet!")
    except Exception as e:
        print(f"Failed to log trade to Google Sheet: {e}")