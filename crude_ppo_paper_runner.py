"""Independent PPO paper-trading runner for crude oil.

Runs beside the rule-based engine but shares no state with it: it never imports
the order manager or Sheet 1 loggers, never places broker orders, and writes
to the local ``ppo_shadow_logs`` directory plus the separate ``PPO_Paper_Trades``
Google Sheet tab. Each trade is charged a fixed Rs 50 (0.5 point) friction.

    python crude_ppo_paper_runner.py --source csv --ticks ticks.csv
    python crude_ppo_paper_runner.py --source angel --token <MCX token>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

import numpy as np

from crude_tick_policy import (
    OBSERVATION_SIZE,
    EngineConfig,
    TickFeatureBuilder,
    TradeEngine,
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
    """Append-only JSONL trade log plus a rolling PnL summary, all in ppo_shadow_logs."""

    def __init__(self, directory: Path = LOG_DIR, sheet_sink: PpoSheetSink | None = None):
        self.sheet_sink = sheet_sink
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.trades = 0
        self.wins = 0
        self.pnl_points = 0.0
        self.pnl_r = 0.0
        self.gross_win = 0.0
        self.gross_loss = 0.0
        self.peak = 0.0
        self.max_drawdown = 0.0

    def log(self, event: dict[str, Any]) -> None:
        stamp = event.get("timestamp") or datetime.now(IST).isoformat()
        record = {"timestamp": stamp, "mode": "PPO_PAPER", **event}
        path = self.directory / f"ppo_paper_events_{stamp[:10]}.jsonl"
        with self._lock:
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
            "Symbol": SYMBOL,
            "Action": event["side"],
            "Entry Price": round(event["entry_price"], 2),
            "Exit Price": round(event["exit_price"], 2),
            "Exit Reason": event["reason"],
            "Gross PnL": round(event["gross_pnl_inr"], 2),
            "Charges": round(event["charges_inr"], 2),
            "Net PnL": round(event["net_pnl_inr"], 2),
            "Gross PnL Points": round(event["pnl_points"], 3),
            "Stop Loss": round(event["stop_loss"], 2),
            "Take Profit": round(event["take_profit"], 2),
            "Trail Active": bool(event.get("trail_active")),
            "Hold Ticks": event.get("hold_ticks", ""),
        }

    def _record_exit(self, event: dict[str, Any]) -> None:
        pnl = float(event["pnl_points"])
        self.trades += 1
        self.wins += pnl > 0
        self.pnl_points += pnl
        self.pnl_r += float(event["pnl_r"])
        self.gross_win += max(pnl, 0.0)
        self.gross_loss += max(-pnl, 0.0)
        self.peak = max(self.peak, self.pnl_points)
        self.max_drawdown = max(self.max_drawdown, self.peak - self.pnl_points)
        summary = {
            "trades": self.trades,
            "win_rate": self.wins / self.trades,
            "total_pnl_points": self.pnl_points,
            "total_pnl_r": self.pnl_r,
            "expectancy_r": self.pnl_r / self.trades,
            "profit_factor": (self.gross_win / self.gross_loss) if self.gross_loss else None,
            "max_drawdown_points": self.max_drawdown,
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
        self._prev_close: float | None = None
        self._ranges: list[float] = []
        self.value: float | None = None

    def update(self, timestamp: datetime, price: float) -> None:
        bucket = int(timestamp.timestamp()) // self.bar_seconds
        if self._bucket is None:
            self._bucket, self._high, self._low = bucket, price, price
        elif bucket != self._bucket:
            true_range = self._high - self._low
            if self._prev_close is not None:
                true_range = max(true_range, abs(self._high - self._prev_close), abs(self._low - self._prev_close))
            self._ranges = (self._ranges + [true_range])[-self.period:]
            if len(self._ranges) >= self.period:
                self.value = float(np.mean(self._ranges))
            self._prev_close = price
            self._bucket, self._high, self._low = bucket, price, price
        else:
            self._high, self._low = max(self._high, price), min(self._low, price)
            self._prev_close = self._prev_close if self._prev_close is not None else price


@dataclass(frozen=True)
class RunnerLimits:
    market_start: str = "09:00"
    market_close: str = "23:30"
    flatten_minutes_before_close: int = 5
    max_trades_per_day: int = 20
    daily_loss_limit_r: float = 5.0


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
    ):
        if fixed_atr is None and atr_source is None:
            raise ValueError("provide a BarAtr source or a fixed ATR")
        self.policy, self.writer = policy, writer
        self.limits = limits or RunnerLimits()
        self.engine = TradeEngine(engine_config)
        self.features = TickFeatureBuilder()
        self.atr_source, self.fixed_atr = atr_source, fixed_atr
        self._day: str | None = None
        self._day_trades = 0
        self._day_r = 0.0
        self._last_price: float | None = None
        self._entry_timestamp = ""

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
        for event in events:
            event = {**event, "timestamp": ts.isoformat()}
            if event["type"] == "entry":
                self._day_trades += 1
                self._entry_timestamp = event["timestamp"]
            else:
                gross = float(event["pnl_points"]) * POINT_VALUE_INR
                event.update(
                    entry_timestamp=self._entry_timestamp,
                    gross_pnl_inr=gross,
                    charges_inr=CHARGES_INR_PER_TRADE,
                    net_pnl_inr=gross - CHARGES_INR_PER_TRADE,
                )
                # Daily loss limit is judged on net PnL, in R.
                self._day_r += (
                    float(event["pnl_points"]) - CHARGES_INR_PER_TRADE / POINT_VALUE_INR
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
            self._day, self._day_trades, self._day_r = day, 0, 0.0
            self.features.reset()
        if not self._market_open(ts):
            return []
        self._last_price = price
        if self.atr_source is not None:
            self.atr_source.update(ts, price)
        atr = self.fixed_atr if self.fixed_atr is not None else self.atr_source.value
        self.features.update(price)
        if atr is None or atr <= 0:
            return []

        if self._must_flatten(ts):
            events = self.engine.force_close(price, "session_end")
            self._emit(ts, events)
            return events

        observation = self.features.observation(atr, self.engine)
        action = np.clip(np.asarray(self.policy(observation), dtype=np.float32).reshape(-1), 0.0, 1.0)
        halted = (
            self._day_trades >= self.limits.max_trades_per_day
            or self._day_r <= -self.limits.daily_loss_limit_r
        )
        events = self.engine.step(
            price, atr, action, allow_entry=not halted,
            momentum=self.features.momentum_atr(atr),
        )
        self._emit(ts, events)
        return events

    def close_open_position(self, reason: str = "runner_stopped") -> list[dict[str, Any]]:
        if self._last_price is None:
            return []
        events = self.engine.force_close(self._last_price, reason)
        self._emit(datetime.now(IST), events)
        return events


def load_policy(model_path: Path) -> Callable[[np.ndarray], np.ndarray]:
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
    from angel_one.login import AngelOneLogin
    from angel_one.smart_stream import LiveTickStore

    store = LiveTickStore(AngelOneLogin.connect_from_env(), correlation_id="ppo_paper")
    if not store.start():
        raise SystemExit("Angel One websocket could not start; paper runner needs live ticks.")
    last_seen = None
    while True:
        store.ensure_subscribed(token, exchange)
        record = store.get(token, max_age=15.0)
        if record is not None and record["received_at"] != last_seen:
            last_seen = record["received_at"]
            yield datetime.now(IST), float(record["ltp"])
        time.sleep(poll_seconds)


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
            runner = PaperRunner(
                load_policy(model_path),
                ShadowLogWriter(LOG_DIR, sheet_sink=sink),
                engine_config=EngineConfig(0.0, 0.0),
                atr_source=BarAtr(300),
            )
            for timestamp, price in angel_ticks(token, "MCX", 0.25):
                if stop_event.is_set():
                    break
                runner.on_tick(timestamp, price)
        except BaseException:  # SystemExit from load_policy must not kill the host process
            logger.exception("PPO paper runner failed; retrying in %.0fs", retry_seconds)
        finally:
            if runner is not None:
                runner.close_open_position()
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
    # The Rs 50 per-trade friction (brokerage + slippage) is charged once in the logs.
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument("--no-sheet", action="store_true", help="skip PPO Google Sheet logging")
    parser.add_argument("--cost-points", type=float, default=0.0)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.source == "csv" and not args.ticks:
        parser.error("--ticks is required for --source csv")
    if args.source == "angel" and not args.token:
        parser.error("--token is required for --source angel")

    sink = None if args.no_sheet else PpoSheetSink()
    runner = PaperRunner(
        load_policy(args.model),
        ShadowLogWriter(args.log_dir, sheet_sink=sink),
        engine_config=EngineConfig(args.slippage_points, args.cost_points),
        atr_source=None if args.atr else BarAtr(args.atr_bar_seconds),
        fixed_atr=args.atr,
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
