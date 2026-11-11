from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import gspread
import pytest

from ppo_futures_sheet import FuturesSheetStore, FuturesTickRecorder, HEADERS, WORKSHEET_TITLE


class FakeSheet:
    def __init__(self, headers=None, timestamps=None):
        self.headers = headers if headers is not None else HEADERS.copy()
        self.timestamps = timestamps or []
        self.deleted = []
        self.batches = []

    def row_values(self, row):
        assert row == 1
        return self.headers

    def col_values(self, column):
        assert column == 1
        return ["Timestamp"] + self.timestamps

    def update(self, *, range_name, values):
        assert range_name == "A1"
        self.headers = values[0]

    def delete_rows(self, first, last):
        self.deleted.append((first, last))
        del self.timestamps[first - 2:last - 1]

    def append_rows(self, rows, value_input_option):
        assert value_input_option == "RAW"
        self.batches.append([row.copy() for row in rows])


def test_cleanup_retains_60_days_and_preserves_invalid_or_other_rows(caplog):
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    old = (now - timedelta(days=60)).isoformat()
    keep = (now - timedelta(days=59)).isoformat()
    sheet = FakeSheet(timestamps=[old, old, keep, "invalid", old, now.isoformat()])
    store = FuturesSheetStore()
    store._cleanup(sheet, now)
    assert sheet.deleted == [(6, 6), (2, 3)]
    assert sheet.timestamps == [keep, "invalid", now.isoformat()]
    assert "invalid timestamps" in caplog.text
    store._cleanup(sheet, now)
    assert len(sheet.deleted) == 2


@pytest.mark.parametrize("existing", [True, False])
def test_store_uses_only_dedicated_sheet_and_batches_rows(monkeypatch, existing):
    import sheets_logger

    sheet = FakeSheet(headers=HEADERS.copy() if existing else [])
    created = []

    class Book:
        def worksheet(self, title):
            assert title == WORKSHEET_TITLE
            if not existing:
                raise gspread.exceptions.WorksheetNotFound(title)
            return sheet

        def add_worksheet(self, title, rows, cols):
            created.append((title, rows, cols))
            return sheet

    monkeypatch.setattr(sheets_logger, "get_gspread_client", lambda: SimpleNamespace(open=lambda title: Book()))
    store = FuturesSheetStore()
    rows = [["2026-11-11T12:00:00+05:30", 8050, 17.2, "token", "session"]]
    store.append(rows)
    assert sheet.headers == HEADERS
    assert sheet.batches == [rows]
    assert bool(created) is not existing


def test_store_rejects_unexpected_existing_headers(monkeypatch):
    import sheets_logger

    sheet = FakeSheet(headers=["Timestamp", "Other"])
    book = SimpleNamespace(worksheet=lambda title: sheet)
    monkeypatch.setattr(sheets_logger, "get_gspread_client", lambda: SimpleNamespace(open=lambda title: book))
    with pytest.raises(ValueError, match="unexpected headers"):
        FuturesSheetStore().append([["sample"]])
    assert not sheet.deleted and not sheet.batches


def test_recorder_samples_every_10_seconds_and_flushes_in_one_batch():
    sheet = FakeSheet()
    store = SimpleNamespace(append=lambda rows: sheet.append_rows(rows, "RAW"))
    recorder = FuturesTickRecorder("123", store)
    start = datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc)
    for seconds in (0, 1, 9, 10, 19, 20):
        recorder.record(start + timedelta(seconds=seconds), 8050 + seconds, None if seconds == 0 else 17.2)
    recorder.close()
    assert len(sheet.batches) == 1
    rows = sheet.batches[0]
    assert [row[1] for row in rows] == [8050, 8060, 8070]
    assert rows[0][2] == "" and rows[1][2] == 17.2
    assert all(row[3] == "123" and row[4] == rows[0][4] for row in rows)


def test_failed_batch_is_logged_and_retried(monkeypatch, caplog):
    import threading
    import ppo_futures_sheet as module

    monkeypatch.setattr(module, "FLUSH_SECONDS", 0.05)
    completed = threading.Event()
    calls = []

    def append(rows):
        calls.append([row.copy() for row in rows])
        if len(calls) == 1:
            raise OSError("sheet unavailable")
        completed.set()

    recorder = FuturesTickRecorder("123", SimpleNamespace(append=append))
    recorder.record(datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc), 8050, 17.2)
    assert completed.wait(3)
    recorder.close()
    assert len(calls) == 2 and calls[0] == calls[1]
    assert "batch failed" in caplog.text
