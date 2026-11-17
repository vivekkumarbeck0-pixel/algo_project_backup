"""Independent PPO paper-trading runner for crude oil.

Runs beside the rule-based engine but shares no state with it: it never imports
the order manager or Sheet 1 loggers and never places broker orders. Live PPO
signals use crude futures ticks to paper-buy ATM CE/PE contracts, with fills
and realized PnL taken from option-premium quotes. Completed trades go to the
separate ``PPO_Paper_Trades`` Google Sheet tab. Local event files are disabled
by default and can be enabled with ``--local-logs``. Each trade is charged a
fixed Rs 50.

    python crude_ppo_paper_runner.py --source csv --ticks ticks.csv
    python crude_ppo_paper_runner.py --source angel --token <MCX token>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

import numpy as np
from ppo_futures_sheet import FuturesTickRecorder

from crude_tick_policy import (
    FLAT,
    OBSERVATION_SIZE,
    EngineConfig,
    TickFeatureBuilder,
    TradeEngine,
    map_action,
)

IST = ZoneInfo("Asia/Kolkata")
LOG_DIR = Path(__file__).resolve().parent / "ppo_shadow_logs"
logger = logging.getLogger("crude_ppo_paper")


POINT_VALUE_INR = 100.0  # Rs 50 friction == 0.5 crude points
CHARGES_INR_PER_TRADE = 50.0
SYMBOL = "CRUDEOIL"


class PpoSheetSink:
    """Sends completed PPO trades to the dedicated PPO sheet tab without blocking ticks."""

    def __init__(self, log_fn: Callable[[dict[str, Any]], Any] | None = None):
        if log_fn is None:
            from sheets_logger import log_ppo_trade as log_fn
        self._log_fn = log_fn
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            row = self._queue.get()
            try:
                if row is None:
                    return
                self._log_fn(row)
            except Exception:
                logger.exception("PPO sheet logging failed")
            finally:
                self._queue.task_done()

    def submit(self, row: dict[str, Any]) -> None:
        self._queue.put(row)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=60)


class ShadowLogWriter:
    """Track paper-trade results and optionally persist local event logs."""

    def __init__(
        self,
        directory: Path = LOG_DIR,
        sheet_sink: PpoSheetSink | None = None,
        *,
        local_logging: bool = False,
    ):
        self.sheet_sink = sheet_sink
        self.directory = Path(directory)
        self.local_logging = local_logging
        if self.local_logging:
            self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.trades = 0
        self.wins = 0
        self.pnl_points = 0.0
        self.net_pnl_inr = 0.0
        self.pnl_r = 0.0
        self.gross_win = 0.0
        self.gross_loss = 0.0
        self.peak = 0.0
        self.max_drawdown = 0.0

    def log(self, event: dict[str, Any]) -> None:
        with self._lock:
            if self.local_logging:
                stamp = event.get("timestamp") or datetime.now(IST).isoformat()
                record = {"timestamp": stamp, "mode": "PPO_PAPER", **event}
                path = self.directory / f"ppo_paper_events_{stamp[:10]}.jsonl"
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, default=float) + "\n")
            if event.get("type") == "exit":
                self._record_exit(event)
                if self.sheet_sink is not None:
                    self.sheet_sink.submit(self._sheet_row(event))

    @staticmethod
    def _sheet_row(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "Entry Timestamp": event.get("entry_timestamp", ""),
            "Exit Timestamp": event["timestamp"],
            "Symbol": event.get("option_symbol", SYMBOL),
            "Action": event["side"],
            "Entry Price": round(event["entry_price"], 2),
            "Exit Price": round(event["exit_price"], 2),
            "Exit Reason": (
                f"underlying_{event['reason']}"
                if event.get("option_symbol")
                else event["reason"]
            ),
            "Gross PnL": round(event["gross_pnl_inr"], 2),
            "Charges": round(event["charges_inr"], 2),
            "Net PnL": round(event["net_pnl_inr"], 2),
            "Gross PnL Points": round(event["pnl_points"], 3),
            "Stop Loss": (
                round(event["option_stop_loss"], 2)
                if event.get("option_stop_loss")
                else ("" if event.get("option_symbol") else round(event["stop_loss"], 2))
            ),
            "Take Profit": (
                round(event["option_take_profit"], 2)
                if event.get("option_take_profit")
                else ("" if event.get("option_symbol") else round(event["take_profit"], 2))
            ),
            "Trail Active": bool(event.get("trail_active")),
            "Hold Ticks": event.get("hold_ticks", ""),
            "Option Side": event.get("option_side", ""),
            "Strike": event.get("strike", ""),
            "Expiry": event.get("expiry", ""),
            "Lot Size": event.get("lot_size", ""),
            "Lots": event.get("lots", ""),
            "Underlying Direction": event.get("underlying_direction", ""),
            "Underlying Entry Price": event.get("underlying_entry_price", ""),
            "Underlying Exit Price": event.get("underlying_exit_price", ""),
            "Underlying Stop Loss": event.get("underlying_stop_loss", ""),
            "Underlying Take Profit": event.get("underlying_take_profit", ""),
            "Underlying PnL Points": event.get("underlying_pnl_points", ""),
            "Underlying PnL R": event.get("underlying_pnl_r", ""),
        }

    def _record_exit(self, event: dict[str, Any]) -> None:
        pnl = float(event["pnl_points"])
        net_pnl_inr = float(event.get("net_pnl_inr", pnl * POINT_VALUE_INR))
        self.trades += 1
        self.wins += net_pnl_inr > 0
        self.pnl_points += pnl
        self.net_pnl_inr += net_pnl_inr
        self.pnl_r += float(event.get("underlying_pnl_r", event["pnl_r"]))
        self.gross_win += max(net_pnl_inr, 0.0)
        self.gross_loss += max(-net_pnl_inr, 0.0)
        self.peak = max(self.peak, self.net_pnl_inr)
        self.max_drawdown = max(self.max_drawdown, self.peak - self.net_pnl_inr)
        if self.local_logging:
            summary = {
                "trades": self.trades,
                "win_rate": self.wins / self.trades,
                "total_option_premium_pnl_points": self.pnl_points,
                "total_net_pnl_inr": self.net_pnl_inr,
                "total_underlying_pnl_r": self.pnl_r,
                "underlying_expectancy_r": self.pnl_r / self.trades,
                "net_profit_factor": (self.gross_win / self.gross_loss) if self.gross_loss else None,
                "max_drawdown_inr": self.max_drawdown,
                "updated": datetime.now(IST).isoformat(),
            }
            (self.directory / "ppo_paper_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )


class BarAtr:
    """ATR(period) over fixed-time bars built from ticks.

    The bar size must match the ATR the model was trained with.
    """

    def __init__(self, bar_seconds: int = 300, period: int = 14):
        self.bar_seconds, self.period = bar_seconds, period
        self._bucket: int | None = None
        self._high = self._low = 0.0
        self._bar_close: float | None = None
        self._prev_close: float | None = None
        self._ranges: list[float] = []
        self.value: float | None = None

    def seed(self, bars: list[tuple[float, float, float]]) -> None:
        """Preload completed (high, low, close) bars so ATR is ready without a live warm-up."""
        self._bucket = None
        self._bar_close = None
        self._prev_close = None
        self._ranges.clear()
        self.value = None
        for high, low, close in bars:
            true_range = high - low
            if self._prev_close is not None:
                true_range = max(true_range, abs(high - self._prev_close), abs(low - self._prev_close))
            self._ranges = (self._ranges + [true_range])[-self.period:]
            self._prev_close = close
        if len(self._ranges) >= self.period:
            self.value = float(np.mean(self._ranges))

    def update(self, timestamp: datetime, price: float) -> None:
        bucket = int(timestamp.timestamp()) // self.bar_seconds
        if self._bucket is None:
            self._bucket, self._high, self._low = bucket, price, price
            self._bar_close = price
        elif bucket != self._bucket:
            true_range = self._high - self._low
            if self._prev_close is not None:
                true_range = max(true_range, abs(self._high - self._prev_close), abs(self._low - self._prev_close))
            self._ranges = (self._ranges + [true_range])[-self.period:]
            if len(self._ranges) >= self.period:
                self.value = float(np.mean(self._ranges))
            self._prev_close = self._bar_close
            self._bucket, self._high, self._low = bucket, price, price
            self._bar_close = price
        else:
            self._high, self._low = max(self._high, price), min(self._low, price)
            self._bar_close = price


@dataclass(frozen=True)
class RunnerLimits:
    market_start: str = "09:00"
    market_close: str = "23:30"
    flatten_minutes_before_close: int = 5
    max_trades_per_day: int = 20
    daily_loss_limit_r: float = 5.0
    daily_loss_limit_inr: float = 5000.0


def _parse_clock(value: str) -> dt_time:
    hour, minute = (int(part) for part in value.split(":"))
    return dt_time(hour, minute)


class PaperRunner:
    """Consumes ticks, queries the policy and paper-trades independently."""

    def __init__(
        self,
        policy: Callable[[np.ndarray], np.ndarray],
        writer: ShadowLogWriter,
        *,
        engine_config: EngineConfig | None = None,
        limits: RunnerLimits | None = None,
        atr_source: BarAtr | None = None,
        fixed_atr: float | None = None,
        tick_recorder: FuturesTickRecorder | None = None,
        option_quote_source: Any | None = None,
    ):
        if fixed_atr is None and atr_source is None:
            raise ValueError("provide a BarAtr source or a fixed ATR")
        self.policy, self.writer = policy, writer
        self.limits = limits or RunnerLimits()
        self.engine = TradeEngine(engine_config)
        self.features = TickFeatureBuilder()
        self.atr_source, self.fixed_atr = atr_source, fixed_atr
        self.tick_recorder = tick_recorder
        self.option_quote_source = option_quote_source
        self._recording_failed = False
        self._day: str | None = None
        self._day_trades = 0
        self._day_r = 0.0
        self._day_net_pnl_inr = 0.0
        self._last_price: float | None = None
        self._entry_timestamp = ""
        self._option_contract: dict[str, Any] | None = None
        self._option_entry_price: float | None = None
        self._last_option_price: float | None = None
        self._last_option_quote_monotonic: float | None = None

    def _market_open(self, ts: datetime) -> bool:
        if ts.weekday() >= 5:
            return False
        now = ts.time().replace(tzinfo=None)
        return _parse_clock(self.limits.market_start) <= now < _parse_clock(self.limits.market_close)

    def _must_flatten(self, ts: datetime) -> bool:
        close = _parse_clock(self.limits.market_close)
        cutoff = close.hour * 60 + close.minute - self.limits.flatten_minutes_before_close
        return ts.hour * 60 + ts.minute >= cutoff

    def _emit(self, ts: datetime, events: list[dict[str, Any]]) -> None:
        for index, event in enumerate(events):
            event = {**event, "timestamp": ts.isoformat()}
            events[index] = event
            if event["type"] == "entry":
                self._day_trades += 1
                self._entry_timestamp = event["timestamp"]
            else:
                point_value = float(event.get("lot_size", POINT_VALUE_INR))
                gross = float(event["pnl_points"]) * point_value
                event.update(
                    entry_timestamp=self._entry_timestamp,
                    gross_pnl_inr=gross,
                    charges_inr=CHARGES_INR_PER_TRADE,
                    net_pnl_inr=gross - CHARGES_INR_PER_TRADE,
                )
                self._day_net_pnl_inr += event["net_pnl_inr"]
                underlying_pnl_points = float(
                    event.get("underlying_pnl_points", event["pnl_points"])
                )
                underlying_pnl_r = float(
                    event.get("underlying_pnl_r", event["pnl_r"])
                )
                if event.get("option_symbol"):
                    self._day_r += underlying_pnl_r
                else:
                    self._day_r += (
                        underlying_pnl_points
                        - CHARGES_INR_PER_TRADE / POINT_VALUE_INR
                    ) / float(event["risk_points"])
            self.writer.log(event)

    def on_tick(self, timestamp: datetime, price: float) -> list[dict[str, Any]]:
        """Process one tick; returns the entry/exit events it produced."""
        ts = timestamp.astimezone(IST) if timestamp.tzinfo else timestamp.replace(tzinfo=IST)
        price = float(price)

        if not np.isfinite(price) or price <= 0:
            return []
        day = ts.date().isoformat()
        if day != self._day:
            if (
                self._day is not None
                and self.option_quote_source is not None
                and self.engine.side != FLAT
            ):
                self._discard_unpriced_option_position(ts, "session_rollover_no_fresh_quote")
            self._day, self._day_trades, self._day_r = day, 0, 0.0
            self._day_net_pnl_inr = 0.0
            self.features.reset()
        if not self._market_open(ts):
            return []
        self._last_price = price
        if self.atr_source is not None:
            self.atr_source.update(ts, price)
        atr = self.fixed_atr if self.fixed_atr is not None else self.atr_source.value
        if self.tick_recorder is not None:
            try:
                self.tick_recorder.record(ts, price, atr)
            except (OSError, ValueError):
                if not self._recording_failed:
                    logger.exception("PPO futures tick recording failed; retrying on the next tick")
                self._recording_failed = True
            else:
                if self._recording_failed:
                    logger.info("PPO futures tick recording recovered")
                self._recording_failed = False
        self.features.update(price)
        if atr is None or atr <= 0:
            return []

        if self._must_flatten(ts):
            option_price = None
            if self.option_quote_source is not None and self.engine.side != FLAT:
                option_price = self._fresh_option_price()
                if option_price is None:
                    self._discard_unpriced_option_position(ts, "session_end_no_fresh_quote")
                    return []
            events = self.engine.force_close(price, "session_end")
            if self.option_quote_source is not None and events:
                self._decorate_option_events(events, option_price=option_price)
            self._emit(ts, events)
            self._release_option_contract(events)
            return events

        observation = self.features.observation(atr, self.engine)
        action = np.clip(np.asarray(self.policy(observation), dtype=np.float32).reshape(-1), 0.0, 1.0)
        halt_reasons = []
        if self._day_trades >= self.limits.max_trades_per_day:
            halt_reasons.append("max_trades_per_day")
        if self._day_r <= -self.limits.daily_loss_limit_r:
            halt_reasons.append("daily_loss_limit")
        if (
            self.option_quote_source is not None
            and self._day_net_pnl_inr <= -self.limits.daily_loss_limit_inr
        ):
            halt_reasons.append("daily_loss_limit_inr")
        halted = bool(halt_reasons)
        momentum = self.features.momentum_atr(atr)
        intent = map_action(action)[0]
        gate = (
            "not_applicable" if intent == FLAT
            else "pass" if self.engine._momentum_supports(intent, momentum)
            else "blocked"
        )
        if self.engine.side != FLAT:
            entry_block = "position_open"
        elif halted:
            entry_block = ",".join(halt_reasons)
        elif intent == FLAT:
            entry_block = "policy_hold"
        elif gate == "blocked":
            entry_block = "momentum_mismatch"
        else:
            entry_block = "none"
        candidate_contract = None
        candidate_option_price = None
        active_option_price = None
        if self.option_quote_source is not None:
            if self.engine.side != FLAT:
                active_option_price = self._fresh_option_price()
                if active_option_price is None:
                    return []
            elif not halted and intent != FLAT and gate == "pass":
                option_side = "CE" if intent > 0 else "PE"
                candidate_contract, candidate_option_price, quote_reason = (
                    self.option_quote_source.prepare(price, option_side)
                )
                if candidate_contract is None:
                    entry_block = "option_contract_unavailable"
                elif candidate_option_price is None:
                    entry_block = quote_reason or "option_quote_not_ready"
        allow_entry = not halted and (
            self.option_quote_source is None or candidate_option_price is not None
        )
        events = self.engine.step(
            price, atr, action, allow_entry=allow_entry,
            momentum=momentum,
        )
        if self.option_quote_source is not None and events:
            if events[0]["type"] == "entry":
                if candidate_contract is None or candidate_option_price is None:
                    raise RuntimeError("PPO option entry was created without a fresh option quote")
                self._option_contract = candidate_contract
                self._option_entry_price = candidate_option_price
                self._last_option_price = candidate_option_price
                self._last_option_quote_monotonic = time.monotonic()
                self._decorate_option_events(
                    events,
                    option_price=candidate_option_price,
                    contract=candidate_contract,
                )
            else:
                if active_option_price is None:
                    raise RuntimeError("PPO option exit was created without a fresh option quote")
                self._decorate_option_events(events, option_price=active_option_price)
        self._emit(ts, events)
        self._release_option_contract(events)
        return events

    def _fresh_option_price(self) -> float | None:
        if self.option_quote_source is None or self._option_contract is None:
            return None
        price = self.option_quote_source.quote(self._option_contract)
        if price is not None and np.isfinite(price) and price > 0:
            self._last_option_price = float(price)
            self._last_option_quote_monotonic = time.monotonic()
            return float(price)
        return None

    def _decorate_option_events(
        self,
        events: list[dict[str, Any]],
        *,
        option_price: float,
        contract: dict[str, Any] | None = None,
    ) -> None:
        for event in events:
            direction = str(event["side"])
            selected = contract or self._option_contract
            if selected is None or self._option_entry_price is None:
                raise RuntimeError("PPO option event is missing its paper contract or entry quote")

            event.update(
                underlying_direction=direction,
                option_side=str(selected["right"]),
                option_symbol=str(selected["symbol"]),
                option_token=str(selected["token"]),
                strike=float(selected["strike"]),
                expiry=str(selected.get("expiry", "")),
                lot_size=int(selected["lot_size"]),
                lots=1,
                option_stop_loss="",
                option_take_profit="",
            )
            if event["type"] == "entry":
                event.update(
                    underlying_entry_price=float(event["entry_price"]),
                    underlying_stop_loss=float(event["stop_loss"]),
                    underlying_take_profit=float(event["take_profit"]),
                    entry_price=float(option_price),
                    option_entry_price=float(option_price),
                    side=f"BUY_{selected['right']}",
                )
            else:
                underlying_points = float(event["pnl_points"])
                underlying_r = float(event["pnl_r"])
                underlying_entry = float(event["entry_price"])
                underlying_exit = float(event["signal_price"])
                option_points = float(option_price) - self._option_entry_price
                event.update(
                    underlying_pnl_points=underlying_points,
                    underlying_pnl_r=underlying_r,
                    underlying_entry_price=underlying_entry,
                    underlying_exit_price=underlying_exit,
                    underlying_stop_loss=float(event["stop_loss"]),
                    underlying_take_profit=float(event["take_profit"]),
                    entry_price=float(self._option_entry_price),
                    exit_price=float(option_price),
                    option_entry_price=float(self._option_entry_price),
                    option_exit_price=float(option_price),
                    pnl_points=option_points,
                    pnl_r=underlying_r,
                    side=f"BUY_{selected['right']}",
                )

    def _release_option_contract(self, events: list[dict[str, Any]]) -> None:
        if (
            self.option_quote_source is not None
            and events
            and events[0]["type"] == "exit"
            and self._option_contract is not None
        ):
            self._clear_option_contract()

    def _clear_option_contract(self) -> None:
        if self.option_quote_source is not None and self._option_contract is not None:
            self.option_quote_source.release(self._option_contract)
        self._option_contract = None
        self._option_entry_price = None
        self._last_option_price = None
        self._last_option_quote_monotonic = None

    def _discard_unpriced_option_position(self, ts: datetime, reason: str) -> None:
        if self.option_quote_source is None or self.engine.side == FLAT:
            return
        event = self.engine.force_close(self._last_price or 0.0, reason)[0]
        underlying_points = float(event["pnl_points"])
        underlying_r = float(event["pnl_r"])
        contract = self._option_contract or {}
        event.update(
            type="unpriced_exit",
            timestamp=ts.isoformat(),
            reason=reason,
            underlying_direction=str(event["side"]),
            underlying_pnl_points=underlying_points,
            underlying_pnl_r=underlying_r,
            option_side=contract.get("right", ""),
            option_symbol=contract.get("symbol", ""),
            option_token=contract.get("token", ""),
            strike=contract.get("strike", ""),
            expiry=contract.get("expiry", ""),
            option_entry_price=self._option_entry_price,
            option_exit_price=None,
            pnl_points=None,
            pnl_r=None,
            option_pnl_unavailable=True,
        )
        self.writer.log(event)
        logger.error(
            "PPO paper option position %s was flattened without a fresh option quote; "
            "its result is excluded from PnL statistics.",
            contract.get("symbol", "unknown contract"),
        )
        self._clear_option_contract()

    def close_open_position(self, reason: str = "runner_stopped") -> list[dict[str, Any]]:
        if self._last_price is None:
            return []
        if self.option_quote_source is not None and self.engine.side != FLAT:
            quote_max_age = float(
                getattr(self.option_quote_source, "max_age_seconds", 6.0)
            )
            quote_age = (
                time.monotonic() - self._last_option_quote_monotonic
                if self._last_option_quote_monotonic is not None
                else float("inf")
            )
            if self._last_option_price is None or quote_age > quote_max_age:
                self._discard_unpriced_option_position(datetime.now(IST), f"{reason}_no_fresh_quote")
                return []
        events = self.engine.force_close(self._last_price, reason)
        if self.option_quote_source is not None and events:
            self._decorate_option_events(events, option_price=self._last_option_price)
        self._emit(datetime.now(IST), events)
        self._release_option_contract(events)
        return events


def load_policy(model_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"PPO model not found at {model_path}. Commit it to git (models/ is deployed from the repo) "
            "or point the runner at an existing model."
        )
    from numpy_ppo import load_numpy_policy

    model = load_numpy_policy(model_path)
    if model is None:
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path))
    shape = tuple(model.observation_space.shape)
    if shape != (OBSERVATION_SIZE,):
        raise SystemExit(
            f"Model expects observations of shape {shape} but the trading policy produces "
            f"({OBSERVATION_SIZE},). Retrain with train_crude_trading_ppo.py."
        )
    return lambda obs: model.predict(obs, deterministic=True)[0]


def csv_ticks(path: Path) -> Iterator[tuple[datetime, float]]:
    """Replay a CSV with ``timestamp`` and ``price`` columns."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            yield datetime.fromisoformat(row["timestamp"]), float(row["price"])


def angel_ticks(token: str, exchange: str, poll_seconds: float) -> Iterator[tuple[datetime, float]]:
    """Live LTP via the Angel One websocket, independent of the rule engine."""
    from angel_one.smart_stream import get_shared_tick_store

    # Shared with the main bot: reuses its websocket and cached login session.
    store = get_shared_tick_store()
    if store is None:
        raise RuntimeError("Angel One websocket unavailable; paper runner needs live ticks.")
    last_seen = None
    while True:
        store.ensure_subscribed(token, exchange)
        record = store.get(token, max_age=15.0)
        if record is not None and record["received_at"] != last_seen:
            last_seen = record["received_at"]
            yield datetime.now(IST), float(record["ltp"])
        time.sleep(poll_seconds)


class AngelOptionQuoteSource:
    """Resolve and read one ATM crude option contract from the shared websocket."""

    def __init__(
        self,
        store: Any,
        reader: Any,
        *,
        underlying: str = "CRUDEOIL",
        exchange: str = "MCX",
        instrument_type: str = "OPTFUT",
        strike_step: float = 50.0,
        lot_size: int = 100,
        max_age_seconds: float = 6.0,
    ):
        if strike_step <= 0 or lot_size <= 0 or max_age_seconds <= 0:
            raise ValueError("option strike step, lot size, and quote age must be positive")
        self.store = store
        self.reader = reader
        self.underlying = underlying
        self.exchange = exchange
        self.instrument_type = instrument_type
        self.strike_step = float(strike_step)
        self.lot_size = int(lot_size)
        self.max_age_seconds = float(max_age_seconds)
        self._contracts: dict[tuple[float, str], dict[str, Any] | None] = {}
        self._selected_token: str | None = None
        self._contracts_day = datetime.now(IST).date()

    def prepare(
        self, underlying_price: float, right: str
    ) -> tuple[dict[str, Any] | None, float | None, str | None]:
        right = str(right).upper()
        if right not in {"CE", "PE"}:
            raise ValueError("option right must be CE or PE")
        today = datetime.now(IST).date()
        if today != self._contracts_day:
            self._contracts.clear()
            self._contracts_day = today
        strike = float(round(float(underlying_price) / self.strike_step) * self.strike_step)
        cache_key = (strike, right)
        if cache_key not in self._contracts:
            raw_contract = self.reader.find_option_token(
                self.underlying,
                strike,
                right,
                instrument_type=self.instrument_type,
                exchange=self.exchange,
            )
            if not raw_contract or not raw_contract.get("token"):
                self._contracts[cache_key] = None
            else:
                try:
                    contract_lot_size = int(
                        float(raw_contract.get("lotsize") or self.lot_size)
                    )
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid lot size in the option master for %s; using configured size %s.",
                        raw_contract.get("symbol") or raw_contract["token"],
                        self.lot_size,
                    )
                    contract_lot_size = self.lot_size
                self._contracts[cache_key] = {
                    **raw_contract,
                    "token": str(raw_contract["token"]),
                    "symbol": str(raw_contract.get("symbol") or raw_contract["token"]),
                    "right": right,
                    "strike": strike,
                    "expiry": str(raw_contract.get("expiry") or ""),
                    "lot_size": contract_lot_size,
                }

        contract = self._contracts[cache_key]
        if contract is None:
            logger.warning(
                "PPO paper option entry blocked: no %s contract found for %s strike %s.",
                right,
                self.underlying,
                strike,
            )
            self._release_selected()
            return None, None, "option_contract_unavailable"

        token = contract["token"]
        if self._selected_token != token:
            self._release_selected()
            self._selected_token = token
        if not self.store.ensure_subscribed(token, self.exchange):
            logger.warning(
                "PPO paper option entry waiting: subscription failed for %s.",
                contract["symbol"],
            )
            return contract, None, "option_subscription_failed"
        price = self._read_quote(contract)
        if price is None:
            return contract, None, "option_quote_not_ready"
        return contract, price, None

    def quote(self, contract: dict[str, Any]) -> float | None:
        token = str(contract["token"])
        if not self.store.ensure_subscribed(token, self.exchange):
            return None
        return self._read_quote(contract)

    def _read_quote(self, contract: dict[str, Any]) -> float | None:
        token = str(contract["token"])
        record = self.store.get(token, self.max_age_seconds)
        if record is None:
            return None
        try:
            price = float(record["ltp"])
        except (KeyError, TypeError, ValueError):
            logger.warning("PPO paper option quote for %s has no valid LTP.", contract.get("symbol"))
            return None
        return price if np.isfinite(price) and price > 0 else None

    def release(self, contract: dict[str, Any]) -> None:
        token = str(contract["token"])
        if self._selected_token == token:
            self._release_selected()

    def _release_selected(self) -> None:
        # LiveTickStore is shared with the main Crude bot and has no per-consumer
        # subscription counts, so unsubscribing here could disable its option feed.
        self._selected_token = None


def make_live_option_quote_source() -> AngelOptionQuoteSource:
    """Build the paper-only option feed from the already-shared Angel websocket."""
    from angel_one.instrument_reader import InstrumentReader
    from angel_one.smart_stream import get_shared_tick_store
    from config import MCX_CRUDE_OPTION, settings

    store = get_shared_tick_store()
    if store is None:
        raise RuntimeError("Angel One websocket unavailable; PPO option paper mode needs live quotes.")
    return AngelOptionQuoteSource(
        store,
        InstrumentReader(),
        underlying=MCX_CRUDE_OPTION.symbol,
        exchange=MCX_CRUDE_OPTION.exchange,
        instrument_type=MCX_CRUDE_OPTION.option_instrument_type,
        strike_step=MCX_CRUDE_OPTION.strike_step,
        lot_size=MCX_CRUDE_OPTION.lot_size,
        max_age_seconds=settings.ws_tick_max_age_seconds,
    )


def resolve_front_month_token() -> str:
    """MCX_CRUDE_FUTURE_TOKEN override, else the nearest-expiry CRUDEOIL future."""
    import os
    from config import MCX_CRUDE_FUTURE
    from angel_one.instrument_reader import InstrumentReader

    override = os.getenv("MCX_CRUDE_FUTURE_TOKEN", "").strip()
    if override:
        return override
    contract = InstrumentReader().find_nearest_future(
        MCX_CRUDE_FUTURE.symbol,
        instrument_type=MCX_CRUDE_FUTURE.future_instrument_type,
        exchange=MCX_CRUDE_FUTURE.exchange,
    )
    if not contract or not contract.get("token"):
        raise RuntimeError("could not resolve the MCX crude futures token")
    return str(contract["token"])


def historical_atr_bars(
    client: Any, token: str, exchange: str, bar_seconds: int, count: int, now: datetime | None = None
) -> list[tuple[float, float, float]]:
    """Completed (high, low, close) bars aggregated from 1-minute broker candles."""
    now = now or datetime.now(IST)
    now = now.replace(tzinfo=IST) if now.tzinfo is None else now.astimezone(IST)
    lookback = timedelta(seconds=bar_seconds * (count + 2) + 3600)
    response = client.get_candle_data(
        token,
        "ONE_MINUTE",
        (now - lookback).strftime("%Y-%m-%d %H:%M"),
        now.strftime("%Y-%m-%d %H:%M"),
        exchange=exchange,
    )
    rows = response.get("data") if isinstance(response, dict) else None
    current_bucket = int(now.timestamp()) // bar_seconds
    buckets: dict[int, tuple[datetime, float, float, float]] = {}
    for row in rows or []:
        try:
            ts = datetime.fromisoformat(str(row[0]))
            ts = ts.replace(tzinfo=IST) if ts.tzinfo is None else ts
            bucket = int(ts.timestamp()) // bar_seconds
            if bucket >= current_bucket:
                continue
            high, low, close = float(row[2]), float(row[3]), float(row[4])
        except (TypeError, ValueError, IndexError):
            continue
        entry = buckets.get(bucket)
        if entry is None:
            buckets[bucket] = (ts, high, low, close)
        else:
            last_ts, old_high, old_low, old_close = entry
            buckets[bucket] = (
                max(ts, last_ts),
                max(old_high, high),
                min(old_low, low),
                close if ts >= last_ts else old_close,
            )
    ordered = [
        (buckets[key][1], buckets[key][2], buckets[key][3])
        for key in sorted(buckets)
    ]
    return ordered[-count:]


def seed_atr_from_history(atr: BarAtr, token: str, exchange: str) -> bool:
    """Best-effort ATR preload using the shared login session (a new login would drop the feed)."""
    try:
        from angel_one.smart_stream import get_shared_tick_store

        store = get_shared_tick_store()
        client = getattr(store, "_client", None)
        if client is None:
            logger.warning("PPO ATR history unavailable; live bar warm-up will continue.")
            return False
        bars = historical_atr_bars(client, token, exchange, atr.bar_seconds, atr.period + 1)
        atr.seed(bars)
        if atr.value is None:
            logger.warning(
                "PPO ATR history returned %d completed bars; %d required. Retrying while live warm-up continues.",
                len(bars),
                atr.period,
            )
            return False
        logger.info("PPO ATR seeded from %d historical bars; ATR=%s", len(bars), atr.value)
        return True
    except Exception:
        logger.exception("Could not seed PPO ATR from history; falling back to live warm-up.")
        return False


def run_live_forever(
    model_path: Path | None = None,
    retry_seconds: float = 60.0,
    stop_event: threading.Event | None = None,
) -> None:
    """Live PPO paper loop for a background thread: never raises, restarts on failure."""
    model_path = model_path or Path(__file__).resolve().parent / "models" / "crude_trading_ppo_best.zip"
    stop_event = stop_event or threading.Event()
    sink = PpoSheetSink()
    while not stop_event.is_set():
        runner = None
        try:
            token = resolve_front_month_token()
            atr = BarAtr(300)
            seed_atr_from_history(atr, token, "MCX")
            runner = PaperRunner(
                load_policy(model_path),
                ShadowLogWriter(LOG_DIR, sheet_sink=sink),
                engine_config=EngineConfig(0.0, 0.0, trailing_enabled=False),
                atr_source=atr,
                option_quote_source=make_live_option_quote_source(),
                tick_recorder=(
                    FuturesTickRecorder(token)
                    if os.getenv("ENABLE_PPO_FUTURES_RECORDING", "1").strip().lower()
                    not in {"0", "false", "no"}
                    else None
                ),
            )
            if runner.tick_recorder is not None:
                logger.info("PPO futures recording enabled: Google Sheet PPO_Futures_Samples, 10-second sampling, 60-day retention")
            else:
                logger.info("PPO futures tick recording disabled via ENABLE_PPO_FUTURES_RECORDING")
            last_atr_seed_attempt = time.monotonic()
            for timestamp, price in angel_ticks(token, "MCX", 0.25):
                if stop_event.is_set():
                    break
                if atr.value is None and time.monotonic() - last_atr_seed_attempt >= retry_seconds:
                    seed_atr_from_history(atr, token, "MCX")
                    last_atr_seed_attempt = time.monotonic()
                runner.on_tick(timestamp, price)
        except BaseException:  # SystemExit from load_policy must not kill the host process
            logger.exception("PPO paper runner failed; retrying in %.0fs", retry_seconds)
        finally:
            if runner is not None:
                runner.close_open_position()
                if runner.tick_recorder is not None:
                    runner.tick_recorder.close()
        stop_event.wait(retry_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "models" / "crude_trading_ppo_best.zip")
    parser.add_argument("--source", choices=("csv", "angel"), required=True)
    parser.add_argument("--ticks", type=Path, help="CSV with timestamp,price (csv source)")
    parser.add_argument("--token", help="MCX instrument token (angel source)")
    parser.add_argument("--exchange", default="MCX")
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--atr", type=float, help="fixed ATR in price points (default: bar ATR)")
    parser.add_argument("--atr-bar-seconds", type=int, default=300)
    # The Rs 50 per-trade friction (brokerage + slippage) is charged once in paper-trade records.
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument("--no-sheet", action="store_true", help="skip PPO Google Sheet logging")
    parser.add_argument("--cost-points", type=float, default=0.0)
    parser.add_argument("--local-logs", action="store_true", help="write event and summary files locally")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR, help="directory used with --local-logs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.source == "csv" and not args.ticks:
        parser.error("--ticks is required for --source csv")
    if args.source == "angel" and not args.token:
        parser.error("--token is required for --source angel")

    if args.source == "angel":
        # Long-running live mode: restart after any failure instead of exiting.
        while True:
            try:
                run_once(args, args.model)
                return
            except KeyboardInterrupt:
                return
            except BaseException:
                logger.exception("Paper runner crashed; retrying in 60s.")
                time.sleep(60)
    run_once(args, args.model)


def run_once(args, model_path: Path) -> None:
    sink = None if args.no_sheet else PpoSheetSink()
    runner = PaperRunner(
        load_policy(model_path),
        ShadowLogWriter(args.log_dir, sheet_sink=sink, local_logging=args.local_logs),
        engine_config=EngineConfig(
            args.slippage_points, args.cost_points, trailing_enabled=False
        ),
        atr_source=None if args.atr else BarAtr(args.atr_bar_seconds),
        fixed_atr=args.atr,
        option_quote_source=(
            make_live_option_quote_source() if args.source == "angel" else None
        ),
    )
    ticks = csv_ticks(args.ticks) if args.source == "csv" else angel_ticks(args.token, args.exchange, args.poll_seconds)
    try:
        for timestamp, price in ticks:
            for event in runner.on_tick(timestamp, price):
                logger.info("%s %s", event["type"].upper(), {k: round(v, 3) if isinstance(v, float) else v for k, v in event.items()})
    except KeyboardInterrupt:
        pass
    finally:
        runner.close_open_position()
        if sink is not None:
            sink.close()


if __name__ == "__main__":
    main()
