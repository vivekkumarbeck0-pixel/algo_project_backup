"""Collect shadow-only signal decisions and later outcomes for entry training."""

from __future__ import annotations

import json
import math
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")
OUTCOME_HORIZONS_MINUTES = (1, 3, 5)
MAX_PENDING_SIGNALS = 500
MAX_TICKS_PER_SIGNAL = 10_000
DATA_PATH = (
    Path(__file__).resolve().parent / "data" / "crude_entry_opportunities.jsonl"
)


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
    """Persist signal-time features, taken/skipped decisions, and futures outcomes."""

    def __init__(self, path: Path = DATA_PATH, outcome_horizon_minutes: int = 5):
        if outcome_horizon_minutes < max(OUTCOME_HORIZONS_MINUTES):
            raise ValueError("outcome horizon must cover all configured label horizons")
        self.path = Path(path)
        self.outcome_horizon = timedelta(minutes=outcome_horizon_minutes)
        self._lock = threading.Lock()
        self._pending: dict[str, dict[str, Any]] = {}

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
