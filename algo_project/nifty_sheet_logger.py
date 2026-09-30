"""Google Sheets logger for NIFTY entries on worksheet 2 only."""

import json
import os
import time
from pathlib import Path

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
    "Trade Number", "Event", "Exit Reason", "P&L",
    "IV", "VIX", "Intratrade Index Prices", "Intratrade Option Prices",
]
OUTBOX_PATH = Path(__file__).resolve().parent / "data" / "nifty_sheet_outbox.jsonl"
RETRY_BACKOFF_SECONDS = (1.0, 2.0)


def _client():
    json_env = os.environ.get("GOOGLE_JSON_KEY")
    if json_env:
        info = json.loads(json_env)
    else:
        credentials_path = Path(__file__).resolve().parent / "service_account.json"
        with credentials_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
    if "private_key" in info:
        info["private_key"] = info["private_key"].replace("\\n", "\n")
    return gspread.authorize(Credentials.from_service_account_info(info, scopes=SCOPES))


def _event_key(event: dict, headers: list[str] | None = None, row: list[str] | None = None):
    fields = ("Timestamp", "Symbol", "Trade Number", "Event")
    event_key = tuple(str(event.get(field, "")) for field in fields)
    if not all(event_key):
        return None
    if headers is None or row is None:
        return event_key
    values = {header: row[index] if index < len(row) else "" for index, header in enumerate(headers)}
    return tuple(str(values.get(field, "")) for field in fields)


def _append_once(event: dict) -> None:
    spreadsheet = _client().open("Crude_Algo_Trade_Logs")
    worksheet = spreadsheet.worksheets()[1]
    existing = worksheet.get_all_values()
    headers = existing[0] if existing else []
    if not headers:
        headers = NIFTY_COLUMNS.copy()
        worksheet.append_row(headers)
        existing = [headers]
    missing = [column for column in NIFTY_COLUMNS if column not in headers]
    if missing:
        headers = headers + missing
        worksheet.update("A1", [headers])

    event_key = _event_key(event)
    target_row = None
    for row_number, row in enumerate(existing[1:], start=2):
        if event_key is not None and _event_key(event, headers, row) == event_key:
            target_row = row_number
            existing_row = row
            break
    else:
        existing_row = None

    if target_row is not None and existing_row is not None:
        updated_row = []
        for index, column in enumerate(headers):
            if column in event and event.get(column) is not None:
                updated_row.append(event.get(column))
            elif index < len(existing_row):
                updated_row.append(existing_row[index])
            else:
                updated_row.append("")
        worksheet.update(f"A{target_row}", [updated_row])
        print(f"Updated existing NIFTY {event.get('Event', 'event')} row in Google Sheet 2.")
        return

    worksheet.append_row([event.get(column, "") for column in headers])
    print(f"Successfully logged NIFTY {event.get('Event', 'event')} to Google Sheet 2!")


def _append_with_retries(event: dict) -> bool:
    for attempt in range(len(RETRY_BACKOFF_SECONDS) + 1):
        try:
            _append_once(event)
            return True
        except Exception as exc:
            if attempt == len(RETRY_BACKOFF_SECONDS):
                print(f"Google Sheets logging failed after {attempt + 1} attempts: {exc}")
                return False
            delay = RETRY_BACKOFF_SECONDS[attempt]
            print(f"Google Sheets logging attempt {attempt + 1} failed; retrying in {delay:g}s: {exc}")
            time.sleep(delay)
    return False


def _queue_event(event: dict) -> None:
    OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTBOX_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"NIFTY {event.get('Event', 'event')} saved locally for later Google Sheets retry.")


def flush_pending_nifty_events() -> int:
    """Retry locally queued NIFTY events and retain anything still failing."""
    try:
        if not OUTBOX_PATH.exists():
            return 0
        pending = [
            json.loads(line)
            for line in OUTBOX_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as exc:
        print(f"Could not read pending NIFTY Sheets events: {exc}")
        return 0

    remaining = []
    sent = 0
    for event in pending:
        if _append_with_retries(event):
            sent += 1
        else:
            remaining.append(event)

    try:
        if remaining:
            temporary_path = OUTBOX_PATH.with_suffix(OUTBOX_PATH.suffix + ".tmp")
            temporary_path.write_text(
                "".join(json.dumps(event, separators=(",", ":"), default=str) + "\n" for event in remaining),
                encoding="utf-8",
            )
            temporary_path.replace(OUTBOX_PATH)
        else:
            OUTBOX_PATH.unlink(missing_ok=True)
    except OSError as exc:
        print(f"Could not update pending NIFTY Sheets events: {exc}")
    return sent


def _append_nifty_event(event: dict) -> None:
    flush_pending_nifty_events()
    if not _append_with_retries(event):
        try:
            _queue_event(event)
        except OSError as exc:
            print(f"Could not save NIFTY event to the local retry queue: {exc}")


def log_nifty_entry(entry: dict) -> None:
    """Append the BUY entry event to the second tab."""
    _append_nifty_event(entry)


def log_nifty_exit(exit_event: dict) -> None:
    """Append the SELL exit event to the second tab."""
    _append_nifty_event(exit_event)