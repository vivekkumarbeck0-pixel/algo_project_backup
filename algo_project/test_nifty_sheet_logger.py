import json

import nifty_sheet_logger


class FakeWorksheet:
    def __init__(self, failures=0, append_then_fail=False):
        self.rows = []
        self.failures = failures
        self.append_then_fail = append_then_fail
        self.event_attempts = 0

    def get_all_values(self):
        return [list(row) for row in self.rows]

    def append_row(self, row):
        if row and row[0] == "Timestamp":
            self.rows.append(list(row))
            return
        self.event_attempts += 1
        if self.failures:
            self.failures -= 1
            if self.append_then_fail:
                self.rows.append(list(row))
            raise RuntimeError("temporary Sheets outage")
        self.rows.append(list(row))

    def update(self, range_name, values):
        if isinstance(range_name, str) and range_name.startswith("A"):
            row_text = range_name[1:].split(":", 1)[0]
            try:
                row_index = int(row_text) - 1
            except ValueError:
                return None
            if 0 <= row_index < len(self.rows):
                self.rows[row_index] = list(values[0])
        return None


class FakeSpreadsheet:
    def __init__(self, worksheet):
        self.worksheet = worksheet

    def open(self, _name):
        return self

    def worksheets(self):
        return [None, self.worksheet]


def _event():
    return {
        "Timestamp": "2026-09-26T10:00:00",
        "Symbol": "NIFTY",
        "Trade Number": 1,
        "Event": "BUY",
        "Strike": 23350,
    }


def _setup_logger(monkeypatch, tmp_path, worksheet):
    monkeypatch.setattr(nifty_sheet_logger, "OUTBOX_PATH", tmp_path / "pending.jsonl")
    monkeypatch.setattr(nifty_sheet_logger, "_client", lambda: FakeSpreadsheet(worksheet))
    monkeypatch.setattr(nifty_sheet_logger.time, "sleep", lambda _seconds: None)


def test_transient_sheet_failure_retries_and_logs_once(monkeypatch, tmp_path):
    worksheet = FakeWorksheet(failures=1)
    _setup_logger(monkeypatch, tmp_path, worksheet)

    nifty_sheet_logger.log_nifty_entry(_event())

    assert worksheet.event_attempts == 2
    assert len(worksheet.rows) == 2
    assert not nifty_sheet_logger.OUTBOX_PATH.exists()


def test_exhausted_retries_queue_event_and_later_flush(monkeypatch, tmp_path):
    worksheet = FakeWorksheet(failures=3)
    _setup_logger(monkeypatch, tmp_path, worksheet)

    nifty_sheet_logger.log_nifty_entry(_event())

    queued = [json.loads(line) for line in nifty_sheet_logger.OUTBOX_PATH.read_text(encoding="utf-8").splitlines()]
    assert queued == [_event()]
    assert worksheet.event_attempts == 3

    worksheet.failures = 0
    assert nifty_sheet_logger.flush_pending_nifty_events() == 1
    assert worksheet.event_attempts == 4
    assert len(worksheet.rows) == 2
    assert not nifty_sheet_logger.OUTBOX_PATH.exists()


def test_retry_does_not_duplicate_if_sheet_saved_before_timeout(monkeypatch, tmp_path):
    worksheet = FakeWorksheet(failures=1, append_then_fail=True)
    _setup_logger(monkeypatch, tmp_path, worksheet)

    nifty_sheet_logger.log_nifty_entry(_event())

    assert worksheet.event_attempts == 1
    assert len(worksheet.rows) == 2
    assert not nifty_sheet_logger.OUTBOX_PATH.exists()


def test_repeated_trade_sync_updates_existing_row(monkeypatch, tmp_path):
    worksheet = FakeWorksheet()
    _setup_logger(monkeypatch, tmp_path, worksheet)

    event = _event()
    nifty_sheet_logger.log_nifty_entry(event)

    updated = dict(event)
    updated["Execution Price (LTP)"] = 120.0
    updated["Target"] = 23450.0
    nifty_sheet_logger.log_nifty_entry(updated)

    assert len(worksheet.rows) == 2
    assert worksheet.rows[1][3] == 120.0
    assert worksheet.rows[1][16] == 23450.0