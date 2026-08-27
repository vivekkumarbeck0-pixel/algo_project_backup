import os
import json
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_gspread_client():
    # Render cloud ke liye ENV check, warna local service_account.json
    json_env = os.environ.get("GOOGLE_JSON_KEY")
    if json_env:
        info = json.loads(json_env)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return gspread.authorize(creds)
    
    return gspread.service_account(filename="service_account.json")

def log_trade(trade_data: dict):
    try:
        gc = get_gspread_client()
        sh = gc.open("Crude_Algo_Trade_Logs")
        worksheet = sh.sheet1
        
        # Headers auto-create agar sheet empty hai
        if len(worksheet.get_all_values()) == 0:
            worksheet.append_row(list(trade_data.keys()))
            
        worksheet.append_row(list(trade_data.values()))
        print("Successfully logged trade to Google Sheet!")
    except Exception as e:
        print(f"Failed to log trade to Google Sheet: {e}")