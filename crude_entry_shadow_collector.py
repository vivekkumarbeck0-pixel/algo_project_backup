"""Collect shadow-only signal decisions and later outcomes for entry training."""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")
OUTCOME_HORIZONS_MINUTES = (1, 3, 5)
MAX_PENDING_SIGNALS = 500
MAX_TICKS_PER_SIGNAL = 10_000
PAIRED_OPTION_SAMPLE_SECONDS = 15
PAIRED_OPTION_MAX_AGE_SECONDS = 20
DATA_PATH = (
    Path(__file__).resolve().parent / "data" / "crude_entry_opportunities.jsonl"
)
logger = logging.getLogger(__name__)


def _to_ist(value: datetime) -> datetime:
    return value.replace(tzinfo=IST) if value.tzinfo is None else value.astimezone(IST)


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _to_ist(value).isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class CrudeEntryShadowCollector:
    """Persist entry candidates and paired CE/PE option-price observations."""

    def __init__(
        self,
        path: Path = DATA_PATH,
        outcome_horizon_minutes: int = 5,
        *,
        sheet_sink: "PpoTrainingSheetSink | None" = None,
    ):
        if outcome_horizon_minutes < max(OUTCOME_HORIZONS_MINUTES):
            raise ValueError("outcome horizon must cover all configured label horizons")
        self.path = Path(path)
        self.outcome_horizon = timedelta(minutes=outcome_horizon_minutes)
        self._lock = threading.Lock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._paired: dict[str, dict[str, Any]] = {}
        self._paired_tokens: dict[str, set[str]] = {}
        self._option_ticks: dict[str, tuple[float, datetime]] = {}
        self.sheet_sink = sheet_sink

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_json_value(record), ensure_ascii=True, allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def record_signal(self, signal: dict[str, Any]) -> str:
        """Store one rule-generated candidate before downstream filters run."""
        if len(self._pending) >= MAX_PENDING_SIGNALS:
            raise RuntimeError("entry shadow collector reached its pending-signal limit")
        signal_id = str(uuid.uuid4())
        timestamp = _to_ist(signal["timestamp"])
        features = {
            key: signal.get(key)
            for key in (
                "scenario",
                "side",
                "strike",
                "futures_price",
                "oi",
                "oi_change",
                "volume",
                "atr",
                "nymex_trend",
                "market_regime",
                "market_reversal",
                "entry_ai_score",
                "ai_win_probability",
                "candle_open",
                "candle_high",
                "candle_low",
                "candle_close",
                "entry_pivot_price",
                "pivot_level_name",
                "pivot_level",
                "next_pivot",
            )
        }
        record = {
            "event": "candidate",
            "signal_id": signal_id,
            "timestamp": timestamp.isoformat(),
            "features": _json_value(features),
            "outcome_source": "MCX futures directional proxy; not counterfactual option PnL",
        }
        self._pending[signal_id] = {
            "timestamp": timestamp,
            "side": str(signal.get("side", "")).upper(),
            "entry_price": float(signal["futures_price"]),
            "ticks": [],
            "decision": "PENDING",
        }
        self._append(record)
        return signal_id

    def record_decision(
        self,
        signal_id: str,
        *,
        decision: str,
        reason: str,
        option_symbol: str = "",
        option_entry_price: float | None = None,
        stop_loss: float | None = None,
        target_price: float | None = None,
    ) -> None:
        pending = self._pending.get(signal_id)
        if pending is not None:
            pending["decision"] = decision
        self._append(
            {
                "event": "decision",
                "signal_id": signal_id,
                "timestamp": datetime.now(IST).isoformat(),
                "decision": decision,
                "reason": reason,
                "option_symbol": option_symbol,
                "option_entry_price": option_entry_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
            }
        )

    def record_futures_tick(self, timestamp: datetime, price: float) -> None:
        if not math.isfinite(price) or price <= 0:
            return
        current = _to_ist(timestamp)
        completed: list[str] = []
        for signal_id, pending in tuple(self._pending.items()):
            if current <= pending["timestamp"]:
                continue
            elapsed = current - pending["timestamp"]
            ticks = pending["ticks"]
            if len(ticks) < MAX_TICKS_PER_SIGNAL:
                ticks.append({"timestamp": current.isoformat(), "futures_price": price})
            if elapsed >= self.outcome_horizon:
                self._append_outcome(signal_id, pending, current)
                completed.append(signal_id)
        for signal_id in completed:
            del self._pending[signal_id]

    def _append_outcome(
        self, signal_id: str, pending: dict[str, Any], finalized_at: datetime
    ) -> None:
        entry_price = pending["entry_price"]
        side_sign = 1.0 if pending["side"] == "CE" else -1.0
        points = [
            side_sign * (float(tick["futures_price"]) - entry_price)
            for tick in pending["ticks"]
        ]
        horizons: dict[str, Any] = {}
        for minutes in OUTCOME_HORIZONS_MINUTES:
            deadline = pending["timestamp"] + timedelta(minutes=minutes)
            sample = next(
                (
                    tick
                    for tick in pending["ticks"]
                    if datetime.fromisoformat(tick["timestamp"]) >= deadline
                ),
                None,
            )
            horizons[f"directional_return_points_{minutes}m"] = (
                side_sign * (float(sample["futures_price"]) - entry_price)
                if sample is not None
                else None
            )
        self._append(
            {
                "event": "futures_outcome",
                "signal_id": signal_id,
                "timestamp": finalized_at.isoformat(),
                "decision": pending["decision"],
                "path_timestamps_available": True,
                "futures_price_path": pending["ticks"],
                "max_favorable_excursion_points": max([0.0, *points]),
                "max_adverse_excursion_points": min([0.0, *points]),
                **horizons,
            }
        )

    def record_trade_outcome(
        self,
        *,
        signal_id: str,
        option_symbol: str,
        option_entry_price: float,
        option_exit_price: float,
        pnl_amount: float,
        quantity: int,
        exit_reason: str,
        option_price_path: list[float],
    ) -> None:
        if not signal_id:
            return
        self._append(
            {
                "event": "executed_trade_outcome",
                "signal_id": signal_id,
                "timestamp": datetime.now(IST).isoformat(),
                "option_symbol": option_symbol,
                "option_entry_price": option_entry_price,
                "option_exit_price": option_exit_price,
                "pnl_points": option_exit_price - option_entry_price,
                "pnl_amount": pnl_amount,
                "quantity": quantity,
                "exit_reason": exit_reason,
                "option_price_path": option_price_path,
            }
        )

    def register_option_pair(
        self,
        signal_id: str,
        signal: dict[str, Any],
        call_contract: dict[str, Any],
        put_contract: dict[str, Any],
    ) -> bool:
        """Track matched CE/PE quotes for a normal strategy candidate."""
        call_token = str(call_contract.get("token") or "").strip()
        put_token = str(put_contract.get("token") or "").strip()
        call_expiry = str(call_contract.get("expiry") or "").strip()
        put_expiry = str(put_contract.get("expiry") or "").strip()
        if (
            not signal_id
            or not call_token
            or not put_token
            or call_token == put_token
            or not call_expiry
            or call_expiry != put_expiry
        ):
            return False

        with self._lock:
            if signal_id in self._paired:
                return True
            self._paired[signal_id] = {
                "started_at": _to_ist(signal["timestamp"]),
                "last_sample_at": None,
                "call_token": call_token,
                "put_token": put_token,
                "call_symbol": str(call_contract.get("symbol") or ""),
                "put_symbol": str(put_contract.get("symbol") or ""),
                "expiry": call_expiry,
                "features": {
                    key: signal.get(key)
                    for key in (
                        "scenario",
                        "side",
                        "strike",
                        "futures_price",
                        "oi",
                        "oi_change",
                        "volume",
                        "atr",
                        "nymex_trend",
                        "market_regime",
                        "market_reversal",
                        "entry_ai_score",
                        "candle_open",
                        "candle_high",
                        "candle_low",
                        "candle_close",
                        "entry_pivot_price",
                    )
                },
            }
            self._paired_tokens.setdefault(call_token, set()).add(signal_id)
            self._paired_tokens.setdefault(put_token, set()).add(signal_id)
        return True

    def record_option_tick(
        self, timestamp: datetime, token: str, price: float
    ) -> int:
        """Store synchronized option prices locally and queue sheet training rows."""
        token = str(token).strip()
        if not token or not math.isfinite(price) or price <= 0:
            return 0
        current = _to_ist(timestamp)
        rows: list[dict[str, Any]] = []
        with self._lock:
            signal_ids = self._paired_tokens.get(token)
            if not signal_ids:
                return 0
            self._option_ticks[token] = (float(price), current)
            for signal_id in tuple(signal_ids):
                pair = self._paired.get(signal_id)
                if pair is None:
                    continue
                elapsed = current - pair["started_at"]
                if elapsed > self.outcome_horizon:
                    continue
                last_sample = pair["last_sample_at"]
                if (
                    last_sample is not None
                    and (current - last_sample).total_seconds()
                    < PAIRED_OPTION_SAMPLE_SECONDS
                ):
                    continue
                call_tick = self._option_ticks.get(pair["call_token"])
                put_tick = self._option_ticks.get(pair["put_token"])
                if call_tick is None or put_tick is None:
                    continue
                quote_ages = [
                    (current - tick_time).total_seconds()
                    for _, tick_time in (call_tick, put_tick)
                ]
                if any(
                    age < 0 or age > PAIRED_OPTION_MAX_AGE_SECONDS
                    for age in quote_ages
                ):
                    continue
                features = pair["features"]
                row = {
                    "Signal ID": signal_id,
                    "Sample Timestamp": current.isoformat(),
                    "Seconds From Signal": max(0, int(elapsed.total_seconds())),
                    "Scenario": features.get("scenario"),
                    "Normal Bot Side": features.get("side"),
                    "Strike": features.get("strike"),
                    "Expiry": pair["expiry"],
                    "Futures Price": features.get("futures_price"),
                    "ATR": features.get("atr"),
                    "OI": features.get("oi"),
                    "OI Change": features.get("oi_change"),
                    "Volume": features.get("volume"),
                    "NYMEX Trend": features.get("nymex_trend"),
                    "Market Regime": features.get("market_regime"),
                    "Market Reversal": features.get("market_reversal"),
                    "Entry AI Score": features.get("entry_ai_score"),
                    "Candle Open": features.get("candle_open"),
                    "Candle High": features.get("candle_high"),
                    "Candle Low": features.get("candle_low"),
                    "Candle Close": features.get("candle_close"),
                    "Entry Pivot Price": features.get("entry_pivot_price"),
                    "CE Symbol": pair["call_symbol"],
                    "CE LTP": call_tick[0],
                    "PE Symbol": pair["put_symbol"],
                    "PE LTP": put_tick[0],
                }
                pair["last_sample_at"] = current
                rows.append(row)

        for row in rows:
            self._append({"event": "paired_option_sample", **row})
            if self.sheet_sink is not None:
                self.sheet_sink.submit(row)
        return len(rows)

    def _remove_pair(self, signal_id: str) -> None:
        pair = self._paired.pop(signal_id, None)
        if pair is None:
            return
        for token in (pair["call_token"], pair["put_token"]):
            signal_ids = self._paired_tokens.get(token)
            if signal_ids is not None:
                signal_ids.discard(signal_id)
                if not signal_ids:
                    del self._paired_tokens[token]

    def active_option_tokens(self) -> set[str]:
        with self._lock:
            return set(self._paired_tokens)

    def is_tracking_option_token(self, token: str) -> bool:
        with self._lock:
            return str(token).strip() in self._paired_tokens

    def expire_option_pairs(self, timestamp: datetime) -> set[str]:
        """Stop tracking completed windows and return tokens no longer needed."""
        current = _to_ist(timestamp)
        with self._lock:
            expired = [
                signal_id
                for signal_id, pair in self._paired.items()
                if current - pair["started_at"] > self.outcome_horizon
            ]
            expired_tokens = {
                token
                for signal_id in expired
                for token in (
                    self._paired[signal_id]["call_token"],
                    self._paired[signal_id]["put_token"],
                )
            }
            for signal_id in expired:
                self._remove_pair(signal_id)
            return expired_tokens - set(self._paired_tokens)


class PpoTrainingSheetSink:
    """Append paired-option samples to the dedicated PPO training worksheet."""

    def __init__(
        self,
        log_fn: Callable[[dict[str, Any]], Any] | None = None,
        *,
        max_pending_rows: int = 1000,
    ):
        if log_fn is None:
            from sheets_logger import log_ppo_training_sample as log_fn
        self._log_fn = log_fn
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending_rows)
        self._thread = threading.Thread(
            target=self._worker, name="ppo-training-sheet", daemon=True
        )
        self._thread.start()

    def _worker(self) -> None:
        while True:
            row = self._queue.get()
            try:
                if row is None:
                    return
                if self._log_fn(row) is False:
                    logger.error("PPO training sample was not written to Google Sheets.")
            except Exception:
                logger.exception("PPO training sheet logging failed.")
            finally:
                self._queue.task_done()

    def submit(self, row: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            logger.error("PPO training sheet queue is full; paired sample was dropped.")

    def close(self, timeout: float = 60.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout=timeout)
