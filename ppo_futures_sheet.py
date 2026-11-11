"""Batch sampled futures observations to a dedicated Google Sheets tab."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import gspread

logger = logging.getLogger("crude_ppo_paper")
IST = ZoneInfo("Asia/Kolkata")
WORKSHEET_TITLE = "PPO_Futures_Samples"
HEADERS = ["Timestamp", "Price", "ATR", "Token", "Session ID"]
SAMPLE_SECONDS = 10
FLUSH_SECONDS = 30
BATCH_SIZE = 200
QUEUE_SIZE = 10000


class FuturesSheetStore:
    def __init__(self) -> None:
        self._worksheet = None
        self._cleanup_day = None

    def _get_worksheet(self):
        if self._worksheet is None:
            from sheets_logger import get_gspread_client

            book = get_gspread_client().open("Crude_Algo_Trade_Logs")
            try:
                sheet = book.worksheet(WORKSHEET_TITLE)
            except gspread.exceptions.WorksheetNotFound:
                sheet = book.add_worksheet(title=WORKSHEET_TITLE, rows=1000, cols=len(HEADERS))
            headers = sheet.row_values(1)
            if not headers:
                sheet.update(values=[HEADERS], range_name="A1")
            elif headers != HEADERS:
                raise ValueError(f"{WORKSHEET_TITLE} has unexpected headers; refusing to alter it")
            self._worksheet = sheet
        return self._worksheet

    def _cleanup(self, sheet, now: datetime) -> None:
        day = now.astimezone(IST).date()
        if self._cleanup_day == day:
            return
        cutoff = day - timedelta(days=59)
        ranges: list[tuple[int, int]] = []
        invalid = 0
        for row_number, value in enumerate(sheet.col_values(1)[1:], start=2):
            try:
                timestamp = datetime.fromisoformat(value)
                if timestamp.tzinfo is None:
                    raise ValueError("timestamp has no timezone")
                expired = timestamp.astimezone(IST).date() < cutoff
            except (TypeError, ValueError):
                invalid += 1
                continue
            if expired:
                if ranges and ranges[-1][1] == row_number - 1:
                    ranges[-1] = (ranges[-1][0], row_number)
                else:
                    ranges.append((row_number, row_number))
        if invalid:
            logger.warning("PPO futures sheet has %d invalid timestamps; those rows were retained", invalid)
        for first, last in reversed(ranges):
            sheet.delete_rows(first, last)
        self._cleanup_day = day

    def append(self, rows: list[list[Any]]) -> None:
        sheet = self._get_worksheet()
        self._cleanup(sheet, datetime.now(IST))
        sheet.append_rows(rows, value_input_option="RAW")


class FuturesTickRecorder:
    """Sample at most once per 10 seconds and send batches off the tick thread."""

    def __init__(self, token: str, store: FuturesSheetStore | None = None):
        self.token = token
        self.session_id = uuid.uuid4().hex
        self.store = store or FuturesSheetStore()
        self._queue: queue.Queue[list[Any]] = queue.Queue(maxsize=QUEUE_SIZE)
        self._last_sample: datetime | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="ppo-futures-sheet", daemon=True)
        self._thread.start()

    def record(self, timestamp: datetime, price: float, atr: float | None) -> None:
        if self._last_sample is not None and (timestamp - self._last_sample).total_seconds() < SAMPLE_SECONDS:
            return
        row = [timestamp.isoformat(), price, "" if atr is None else atr, self.token, self.session_id]
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            logger.error("PPO futures sheet queue full; sample at %s was not recorded", timestamp.isoformat())
            return
        self._last_sample = timestamp

    def _worker(self) -> None:
        batch: list[list[Any]] = []
        next_flush = time.monotonic() + FLUSH_SECONDS
        while not self._stop.is_set() or not self._queue.empty() or batch:
            timeout = min(1.0, max(0.0, next_flush - time.monotonic()))
            if len(batch) < BATCH_SIZE:
                try:
                    batch.append(self._queue.get(timeout=0 if self._stop.is_set() else timeout))
                except queue.Empty:
                    pass
            if batch and (
                len(batch) >= BATCH_SIZE
                or time.monotonic() >= next_flush
                or (self._stop.is_set() and self._queue.empty())
            ):
                try:
                    self.store.append(batch)
                except Exception:
                    logger.exception(
                        "PPO futures sheet batch failed; retaining %d samples for retry "
                        "(an ambiguous append failure may produce duplicates)", len(batch)
                    )
                    if self._stop.wait(FLUSH_SECONDS):
                        logger.error("PPO futures sheet shutdown could not flush %d samples", len(batch) + self._queue.qsize())
                        return
                else:
                    for _ in batch:
                        self._queue.task_done()
                    batch.clear()
                next_flush = time.monotonic() + FLUSH_SECONDS
            elif not batch and time.monotonic() >= next_flush:
                next_flush = time.monotonic() + FLUSH_SECONDS

    def close(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.error("PPO futures sheet worker did not finish within %.1f seconds; queued samples may be lost", timeout)
