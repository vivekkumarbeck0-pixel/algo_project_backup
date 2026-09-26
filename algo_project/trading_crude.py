"""MCX Crude Oil option-buying engine.

This script is intentionally isolated and does not import or depend on any
Nifty module, engine, or shared strategy logic.

Core idea:
- Trade only the MCX Crude Oil FUTCOM contract.
- Use Angel One SmartStream WebSocket for real-time futures ticks (LTP, OI, volume).
- Use yfinance CL=F 1-minute trend as a global lead filter.
- Buy ATM CE or PE only when the OI + price + volume + NYMEX scenario matches.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import shutil
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from config import MCX_CRUDE_FUTURE, settings
from engine.adaptive_config import AdaptiveConfigReloader
from sheets_logger import log_trade


IST = ZoneInfo("Asia/Kolkata")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

_status_line_width = 0
_startup_candle_rate_lock = threading.Lock()
_last_startup_candle_request = 0.0


def _now_ist() -> datetime:
    return datetime.now(IST)


def _as_ist(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=IST)
    return value.astimezone(IST)


def _format_ist_timestamp(value: Optional[datetime] = None) -> str:
    return _as_ist(value or _now_ist()).strftime(TIMESTAMP_FORMAT)


def _is_rate_limited(exc: BaseException) -> bool:
    """Recognize Angel One AB1004/access-rate and HTTP 429 failures."""
    from angel_one.market_data import is_rate_limit_error

    status_code = getattr(exc, "status_code", getattr(exc, "status", None))
    return str(status_code) == "429" or is_rate_limit_error(exc)


def _fetch_startup_candles_with_retry(client, token: str, from_time: datetime, to_time: datetime):
    """Fetch startup candles with a serialized request gap and bounded backoff."""
    global _last_startup_candle_request
    attempts = max(1, int(settings.candle_fetch_retries))

    for attempt in range(1, attempts + 1):
        with _startup_candle_rate_lock:
            gap = float(settings.candle_min_request_interval_seconds)
            wait = gap - (time.monotonic() - _last_startup_candle_request)
            if wait > 0:
                time.sleep(wait)
            _last_startup_candle_request = time.monotonic()

        try:
            return client.get_candle_data(
                token,
                "ONE_MINUTE",
                from_time.strftime("%Y-%m-%d %H:%M"),
                to_time.strftime("%Y-%m-%d %H:%M"),
                exchange=MCX_CRUDE_FUTURE.exchange,
            )
        except Exception as exc:
            if not _is_rate_limited(exc) or attempt >= attempts:
                raise
            delay = min(
                float(settings.candle_cooldown_max_wait_seconds),
                float(settings.candle_retry_backoff_seconds) * (2 ** (attempt - 1)),
            )
            logger.warning(
                "Startup MCX candle request throttled; retrying in %.1fs (%d/%d): %s",
                delay, attempt, attempts, exc,
            )
            time.sleep(delay)


def _clear_status_line() -> None:
    """Wipe the in-place status line so real log output never lands on top of it."""
    global _status_line_width
    if _status_line_width:
        sys.stdout.write("\r" + " " * _status_line_width + "\r")
        sys.stdout.flush()
        _status_line_width = 0


class _StatusAwareHandler(logging.StreamHandler):
    def emit(self, record):
        _clear_status_line()
        super().emit(record)


logger = logging.getLogger("crude_option_buyer")
if not logger.handlers:
    handler = _StatusAwareHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

from flask import Flask

_flask_app = Flask(__name__)

@_flask_app.route('/')
@_flask_app.route('/health')
def _health():
    return "MCX Crude Algo Engine Alive", 200

def _run_health_server():
    port = int(os.environ.get("PORT", 10000))
    _flask_app.run(host="0.0.0.0", port=port, use_reloader=False)

threading.Thread(target=_run_health_server, daemon=True).start()
logger.info("Render Health Check server started on Port 10000.")


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    oi: float = 0.0


@dataclass
class Position:
    side: str  # "CE" or "PE"
    strike: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    target_price: float
    trailing_stop: float
    atr_value: float
    futures_entry: float
    futures_entry_oi: float
    scenario: str
    is_real: bool = False
    option_token: str = ""
    option_symbol: str = ""
    quantity: int = 0
    entry_order_id: str = ""
    entry_volume: Optional[float] = None
    entry_oi_change: Optional[float] = None
    pivot_level: Optional[float] = None
    nymex_trend: str = "NEUTRAL"
    entry_index_value: Optional[float] = None
    entry_nearest_pivot: Optional[str] = None
    entry_pivot_number: Optional[int] = None
    entry_pivot_price: Optional[float] = None
    entry_candle_open: Optional[float] = None
    entry_candle_high: Optional[float] = None
    entry_candle_low: Optional[float] = None
    entry_candle_close: Optional[float] = None
    futures_target: Optional[float] = None
    target_source: str = "ATR"
    entry_market_regime: Optional[str] = None
    entry_momentum_strength: float = 0.0
    intratrade_option_prices: list[float] = field(default_factory=list)


class AngelSmartWebSocketClient:
    """Thin wrapper around Angel One SmartWebSocketV2 for live MCX Crude stream."""

    EXCHANGE_TYPE_MAP = {
        "NSE": 1,
        "NFO": 2,
        "BSE": 3,
        "BFO": 4,
        "MCX": 5,
        "NCX": 7,
        "CDS": 13,
    }
    SNAP_QUOTE_MODE = 3

    def __init__(self, api_key: str, client_code: str, feed_token: str, jwt_token: str = ""):
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self.jwt_token = jwt_token
        self.ws = None
        self.connected = False
        self.queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._open_event = threading.Event()
        self._connect_thread: Optional[threading.Thread] = None

        self._SmartWebSocketV2 = None

    @staticmethod
    def _import_smart_websocket_v2():
        """Import SmartWebSocketV2 across SDK naming variants."""
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2

            return SmartWebSocketV2
        except Exception:
            pass

        try:
            from smartapi.smartWebSocketV2 import SmartWebSocketV2

            return SmartWebSocketV2
        except Exception:
            pass

        try:
            import SmartAPI  # type: ignore

            # Mirror package name variants used by different SDK releases.
            sys.modules.setdefault("smartapi", SmartAPI)
            sys.modules.setdefault("SmartApi", SmartAPI)
            from SmartAPI.smartWebSocketV2 import SmartWebSocketV2  # type: ignore

            return SmartWebSocketV2
        except Exception:
            return None

    def connect(self, timeout: float = 10.0) -> bool:
        if self._SmartWebSocketV2 is None:
            self._SmartWebSocketV2 = self._import_smart_websocket_v2()

        if self._SmartWebSocketV2 is None:
            logger.error(
                "Angel One SmartWebSocketV2 is not importable. Install smartapi-python in this Python environment."
            )
            return False

        try:
            self.ws = self._SmartWebSocketV2(
                self.jwt_token,
                self.api_key,
                self.client_code,
                self.feed_token,
            )
            self.ws.on_open = self._on_open
            self.ws.on_data = self._on_message
            self.ws.on_message = self._on_message
            self.ws.on_error = self._on_error
            self.ws.on_close = self._on_close
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to initialize Angel One websocket client: %s", exc)
            self.connected = False
            return False

        # SmartWebSocketV2.connect() runs a blocking run_forever() loop, so it
        # must be driven from a background thread; we only wait here for the
        # on_open callback (or timeout) before returning control to the caller.
        self._open_event.clear()
        self._connect_thread = threading.Thread(target=self._run_forever, daemon=True)
        self._connect_thread.start()

        opened = self._open_event.wait(timeout)
        if not opened:
            logger.error("Angel One websocket did not open within %.0fs.", timeout)
            return False

        logger.info("Angel One SmartStream websocket connected for MCX Crude stream.")
        return True

    def _run_forever(self):
        try:
            self.ws.connect()
        except Exception as exc:  # pragma: no cover
            logger.info("Angel One websocket connection closed; reconnecting in the background.")
            self.connected = False
            self._open_event.clear()

    def _on_open(self, *args, **kwargs):
        self.connected = True
        self._open_event.set()
        logger.info("Angel One SmartStream socket opened.")

    def _on_message(self, *args):
        # SmartWebSocketV2 invokes on_data/on_message with (wsapp, message);
        # message is always the final positional argument.
        message = args[-1] if args else None
        if isinstance(message, str):
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                payload = {"raw": message}
        else:
            payload = message

        if payload is not None:
            self.queue.put(payload)

    def _on_error(self, *args):
        logger.info("Angel One websocket error; reconnect will resume the stream.")
        self.connected = False
        self._open_event.clear()

    def _on_close(self, *args, **kwargs):
        logger.info("Angel One websocket closed; reconnect will resume the stream.")
        self.connected = False
        self._open_event.clear()

    def subscribe_futures(self, token: str, exchange: str = "MCX") -> None:
        if self.ws is None:
            logger.warning("Websocket not initialized yet; cannot subscribe to futures feed.")
            return

        exchange_type = self.EXCHANGE_TYPE_MAP.get(exchange.upper())
        if exchange_type is None:
            logger.error("Unknown exchange '%s' for websocket subscription.", exchange)
            return

        try:
            token_list = [{"exchangeType": exchange_type, "tokens": [str(token)]}]
            self.ws.subscribe("crude1", self.SNAP_QUOTE_MODE, token_list)
            logger.info("Subscribed to MCX Crude futures token %s (exchangeType=%s).", token, exchange_type)
        except Exception as exc:  # pragma: no cover
            self.connected = False
            self._open_event.clear()
            logger.info("MCX Crude websocket subscription closed; reconnect will resume the stream.")

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close_connection()
            except Exception:  # pragma: no cover
                pass
        self.connected = False
        self._open_event.clear()


class YFinanceLeadFilter:
    """1-minute NYMEX WTI CL=F trend filter. This is not a broker feed and avoids MCX rate-limit issues."""

    def __init__(self, refresh_seconds: float = 60.0):
        self.refresh_seconds = refresh_seconds
        self._lock = threading.Lock()
        self._bars: Deque[Bar] = deque(maxlen=settings.max_nymex_history_bars)
        self._last_update: Optional[datetime] = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self.refresh()
            except Exception as exc:  # pragma: no cover
                logger.exception("YFinance CL=F refresh failed: %s", exc)
            self._stop_event.wait(self.refresh_seconds)

    def refresh(self):
        ticker = yf.Ticker("CL=F")
        history = ticker.history(period="7d", interval="1m", auto_adjust=False)
        if history.empty:
            return

        history = history.reset_index()
        if "Datetime" in history.columns:
            time_col = "Datetime"
        elif "Date" in history.columns:
            time_col = "Date"
        elif "index" in history.columns:
            time_col = "index"
        else:
            time_col = history.columns[0]

        with self._lock:
            self._bars.clear()
            for _, row in history.iterrows():
                try:
                    ts = pd.to_datetime(row[time_col])
                    bar = Bar(
                        timestamp=ts.to_pydatetime(),
                        open=float(row.get("Open", 0.0) or 0.0),
                        high=float(row.get("High", 0.0) or 0.0),
                        low=float(row.get("Low", 0.0) or 0.0),
                        close=float(row.get("Close", 0.0) or 0.0),
                        volume=float(row.get("Volume", 0.0) or 0.0),
                    )
                    self._bars.append(bar)
                except Exception:
                    continue
            self._last_update = datetime.now()

    @property
    def latest_bar(self) -> Optional[Bar]:
        with self._lock:
            return None if not self._bars else list(self._bars)[-1]

    @property
    def trend(self) -> str:
        with self._lock:
            if len(self._bars) < 2:
                return "NEUTRAL"
            recent = list(self._bars)[-2:]
            previous_close = recent[0].close
            current_close = recent[-1].close
            if current_close > previous_close:
                return "GREEN"
            if current_close < previous_close:
                return "RED"
            return "NEUTRAL"


class CrudeOptionBuyer:
    """MCX Crude FUTCOM + ATM option buying engine."""

    def __init__(self):
        self.settings = settings
        self.instrument = MCX_CRUDE_FUTURE
        self.option_type = None
        self.current_price: Optional[float] = None
        self.current_oi: Optional[float] = None
        self.current_volume: Optional[float] = None
        self.current_timestamp: Optional[datetime] = None

        self.paper_log_path = Path(self.settings.paper_trade_log)
        self.paper_summary_path = Path(self.settings.paper_trade_summary)
        self.state_path = Path(self.settings.crude_state_file)
        self.paper_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.paper_summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.paper_trade_count = 0
        self.paper_trade_history: list[dict[str, Any]] = []
        self.paper_realized_pnl = 0.0
        self.paper_wins = 0
        self.paper_losses = 0
        self._trail_distance = 3.0
        self._last_status_print = 0.0
        self._status_interval = 1.0
        self._square_off_done = False
        self._option_ltp: Optional[float] = None
        self._option_ltp_token = ""
        self._option_ltp_time = 0.0
        self._option_ltp_max_age = 5.0
        self._last_option_poll = 0.0
        self._option_poll_interval = 2.0
        self._option_ltp_warning_interval = 60.0
        # A subscribe-triggered snap-quote tick can arrive a moment after entry carrying
        # a stale/last-close premium; ignore it until real prints settle.
        self._entry_settle_seconds = 2.0
        self._option_ltp_warnings: dict[str, float] = {}

        self.futures_bars: Deque[Bar] = deque(maxlen=self.settings.max_futures_history_bars)
        self._last_tick_time: Optional[datetime] = None
        self._stale_feed_since: Optional[datetime] = None
        self._stale_log_interval = 60.0
        self._last_oi: Optional[float] = None
        self._last_close: Optional[float] = None
        self._last_cum_volume: Optional[float] = None
        self._last_signal_bar: Optional[datetime] = None
        self._last_no_signal_reason = ""
        self._last_no_signal_log = 0.0
        self._no_signal_log_interval = 30.0
        self._bar_lock = threading.Lock()
        self._trade_lock = threading.Lock()

        self._position: Optional[Position] = None
        self._adaptive_config = AdaptiveConfigReloader(
            Path(self.settings.adaptive_crude_config_file)
        )
        self._live_signal_queue: queue.Queue = queue.Queue()
        self._rest_client = None
        self._rest_fallback = False
        self._last_rest_poll = 0.0
        self._rest_poll_interval = 2.0
        self._next_websocket_reconnect = 0.0
        self._historical_bars_loaded = False
        self._hybrid_model = None
        self._hybrid_model_load_attempted = False
        self._hybrid_model_path = Path(__file__).with_name("crude_hybrid_model.pkl")

        self._smart_stream = AngelSmartWebSocketClient(
            api_key=self.settings.angel_api_key,
            client_code=self.settings.angel_client_code,
            feed_token=self.settings.angel_feed_token,
            jwt_token=self.settings.angel_jwt_token,
        )
        self.nymex_filter = YFinanceLeadFilter(refresh_seconds=self.settings.yfinance_refresh_seconds)
        self._futures_token_cache: Optional[str] = None
        self._restore_state()
        self._refresh_adaptive_config()

    def _refresh_adaptive_config(self) -> None:
        try:
            if self._adaptive_config.refresh(
                self.settings,
                position_is_open=self._position is not None,
            ):
                logger.info("Applied approved adaptive Crude config from %s.", self._adaptive_config.path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.error("Adaptive Crude config rejected; keeping current settings: %s", exc)

    def _set_no_signal_reason(self, reason: str) -> None:
        self._last_no_signal_reason = reason
        now = time.monotonic()
        last_log = getattr(self, "_last_no_signal_log", 0.0)
        interval = getattr(self, "_no_signal_log_interval", 30.0)
        if now - last_log >= interval:
            self._last_no_signal_log = now
            logger.info("Crude signal blocked: %s", reason)

    def _get_hybrid_model(self):
        if self._hybrid_model_load_attempted:
            return self._hybrid_model

        self._hybrid_model_load_attempted = True
        try:
            from engine.crude_hybrid_ml import load_model

            self._hybrid_model = load_model(self._hybrid_model_path)
            logger.info("Loaded Crude hybrid model lazily for signal monitoring: %s", self._hybrid_model_path)
        except Exception as exc:
            logger.warning("Crude hybrid model unavailable; rule-based entries remain enabled: %s", exc)
        return self._hybrid_model

    def start(self):
        logger.info("Starting MCX Crude option buying engine in PAPER mode.")
        logger.info("Paper mode is safe: no live orders are sent. Signals are logged only.")
        self.nymex_filter.start()
        self._connect_futures_stream()

        while True:
            try:
                self._process_live_cycle()
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user.")
                break
            except Exception as exc:  # pragma: no cover
                logger.exception("Fatal engine loop error: %s", exc)
                time.sleep(2.0)

        self.stop()

    def stop(self):
        if self._position is not None:
            self._persist_state()
            logger.info("Engine stopping with an open position; it will resume on the next same-day start.")
        self._smart_stream.close()
        self.nymex_filter.stop()
        _clear_status_line()
        self.print_daily_summary()

    def print_daily_summary(self) -> None:
        """Print all of today's persisted Crude paper trades at shutdown."""
        today = datetime.now(IST).date().isoformat()
        entries: dict[str, list[dict[str, Any]]] = {}
        completed: list[tuple[dict[str, Any], dict[str, Any]]] = []

        try:
            paper_log_path = getattr(self, "paper_log_path", None)
            lines = paper_log_path.read_text(encoding="utf-8").splitlines() if paper_log_path else []
        except OSError:
            lines = []

        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or not str(record.get("timestamp", "")).startswith(today):
                continue

            symbol = str(record.get("symbol") or "")
            if record.get("status") == "entry":
                entries.setdefault(symbol, []).append(record)
            elif record.get("status") == "exit" and entries.get(symbol):
                completed.append((entries[symbol].pop(0), record))

        wins = sum(1 for _, exit_record in completed if float(exit_record.get("pnl_amount") or 0.0) > 0)
        losses = sum(1 for _, exit_record in completed if float(exit_record.get("pnl_amount") or 0.0) < 0)

        print()
        print("=" * 132)
        print(f"DAILY MCX CRUDE TRADING SUMMARY - {today}")
        print("=" * 132)
        print(f"Completed Trades : {len(completed)}")
        print(f"Win / Loss Count : {wins} / {losses}")
        print("-" * 132)
        header = (
            f"{'#':<4}{'Symbol':<27}{'Entry Time':<20}{'Exit Time':<20}"
            f"{'Entry':<12}{'Exit':<12}{'Realized P&L':<16}{'Exit Reason'}"
        )
        print(header)
        print("-" * 132)

        if not completed:
            print("No completed Crude trades today.")
        else:
            for number, (entry, exit_record) in enumerate(completed, start=1):
                print(
                    f"{number:<4}{str(exit_record.get('symbol') or entry.get('symbol') or '--'):<27}"
                    f"{str(entry.get('timestamp', '--')):<20}{str(exit_record.get('timestamp', '--')):<20}"
                    f"{float(entry.get('entry_premium') or 0.0):<12.2f}"
                    f"{float(exit_record.get('exit_premium') or 0.0):<12.2f}"
                    f"{float(exit_record.get('pnl_amount') or 0.0):<+16.2f}"
                    f"{exit_record.get('reason') or '--'}"
                )

        if self._position is not None:
            position = self._position
            print("-" * 132)
            print(
                f"OPEN {position.option_symbol:<22}{position.entry_time.strftime('%Y-%m-%d %H:%M:%S'):<20}"
                f"{'--':<20}{position.entry_price:<12.2f}{'--':<12}{'--':<16}OPEN"
            )

        print("-" * 132)
        print(f"Net Realized P&L Today : {self.paper_realized_pnl:+,.2f}")
        print("=" * 132)

    def _persist_state(self) -> None:
        position = asdict(self._position) if self._position is not None else None
        if position is not None:
            position["entry_time"] = self._position.entry_time.isoformat()

        payload = {
            "date": datetime.now(IST).date().isoformat(),
            "position": position,
            "trail_distance": self._trail_distance,
            "paper_realized_pnl": self.paper_realized_pnl,
            "paper_wins": self.paper_wins,
            "paper_losses": self.paper_losses,
        }
        temporary_path = self.state_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary_path.replace(self.state_path)

    def _restore_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Unable to load Crude trading state from %s: %s", self.state_path, exc)
            return

        today = datetime.now(IST).date().isoformat()
        state_date = payload.get("date")
        is_rollover = state_date != today

        self.paper_realized_pnl = 0.0 if is_rollover else float(payload.get("paper_realized_pnl", 0.0))
        self.paper_wins = 0 if is_rollover else int(payload.get("paper_wins", 0))
        self.paper_losses = 0 if is_rollover else int(payload.get("paper_losses", 0))
        self._trail_distance = float(payload.get("trail_distance", self._trail_distance))
        position = payload.get("position")
        if not isinstance(position, dict):
            if is_rollover:
                self._persist_state()
            return

        try:
            values = dict(position)
            values["entry_time"] = datetime.fromisoformat(values["entry_time"])
            self._position = Position(**values)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Unable to restore saved Crude position: %s", exc)
            return

        if is_rollover:
            self._persist_state()
            logger.info(
                "Rolled Crude state from %s to %s while preserving the open %s position.",
                state_date,
                today,
                self._position.side,
            )
        logger.info(
            "Restored open %s %s position from today's Crude state; position management resumed.",
            self._position.side,
            self._position.option_symbol,
        )

    def _connect_futures_stream(self):
        if not self.settings.angel_jwt_token or not self.settings.angel_feed_token:
            if not self._auto_login():
                level = logger.info if self.settings.execution_mode.upper() == "PAPER" else logger.warning
                level("Angel One auto-login failed; running in simulation mode.")
                return

        if not self.settings.angel_api_key or not self.settings.angel_client_code:
            logger.warning("Angel One API key/client code missing; running in simulation mode.")
            return

        token = self._resolve_futures_token()
        if token:
            self._bootstrap_futures_bars(token)
        else:
            logger.warning("Unable to resolve the MCX Crude futures token for startup candles.")

        connected = self._smart_stream.connect()
        if not connected:
            self._enable_rest_fallback("websocket connection failed")
            return

        if token:
            self._smart_stream.subscribe_futures(str(token), exchange=self.instrument.exchange)
        else:
            logger.warning(
                "Unable to resolve the MCX Crude futures token automatically; switching to REST ticker fallback."
            )
            self._enable_rest_fallback("futures token could not be resolved")

        if self._position is not None and self._position.option_token:
            self._smart_stream.subscribe_futures(
                self._position.option_token, exchange=self.settings.option_exchange
            )

    def _bootstrap_futures_bars(self, token: str) -> None:
        """Load completed MCX one-minute candles so startup has no live warm-up wait."""
        if self._historical_bars_loaded or not self._ensure_rest_client():
            return

        now = datetime.now(IST).replace(second=0, microsecond=0)
        from_time = now - timedelta(minutes=35)
        try:
            response = _fetch_startup_candles_with_retry(self._rest_client, token, from_time, now)
        except Exception as exc:
            logger.warning("Could not load startup MCX Crude candles; using live stream: %s", exc)
            return

        rows = response.get("data") if isinstance(response, dict) else None
        if not isinstance(rows, list):
            logger.warning("Startup MCX Crude candle response contained no usable data.")
            return

        bars: list[Bar] = []
        for row in rows:
            if isinstance(row, dict):
                row = (
                    row.get("timestamp") or row.get("time"),
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("volume"),
                    row.get("oi") or row.get("open_interest"),
                )
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                continue
            try:
                timestamp = pd.Timestamp(row[0])
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.tz_convert(IST).tz_localize(None)
                timestamp = timestamp.to_pydatetime().replace(second=0, microsecond=0)
                if timestamp >= now.replace(tzinfo=None):
                    continue
                bars.append(
                    Bar(
                        timestamp=timestamp,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5] or 0.0),
                        oi=float(row[6] or 0.0) if len(row) > 6 else 0.0,
                    )
                )
            except (TypeError, ValueError, OverflowError):
                continue

        if len(bars) < 22:
            logger.warning("Only %d startup MCX Crude candles available; live warm-up will continue.", len(bars))
            return

        with self._bar_lock:
            self.futures_bars.clear()
            self.futures_bars.extend(bars[-30:])
            self._last_close = self.futures_bars[-1].close
        self._historical_bars_loaded = True
        logger.info("Loaded %d completed MCX Crude 1-minute candles; scanning is ready on the first live tick.", len(self.futures_bars))

    def _auto_login(self) -> bool:
        """Log in to Angel One via SmartConnect (client id/password/TOTP) and
        populate the JWT + feed token automatically, without any manual .env token."""
        try:
            from angel_one.login import AngelOneLogin

            client = AngelOneLogin.connect_from_env()
        except Exception as exc:
            logger.error("Angel One auto-login failed: %s", exc)
            return False

        self._rest_client = client
        self.settings.angel_jwt_token = client.access_token
        self.settings.angel_feed_token = client.feed_token
        if client.user_id:
            self.settings.angel_client_code = client.user_id

        self._smart_stream.jwt_token = self.settings.angel_jwt_token
        self._smart_stream.feed_token = self.settings.angel_feed_token
        self._smart_stream.client_code = self.settings.angel_client_code

        logger.info("Angel One auto-login succeeded; JWT and feed token acquired.")
        return True

    def _enable_rest_fallback(self, reason: str) -> None:
        """Prepare REST polling without allowing broker errors to stop the engine."""
        self._rest_fallback = True
        logger.warning("Using Angel One REST ticker fallback: %s.", reason)

    def _ensure_rest_client(self) -> bool:
        if self._rest_client is not None:
            return True

        return self._auto_login()

    def _poll_rest_ticker(self) -> None:
        token = self._resolve_futures_token()
        if not token or not self._ensure_rest_client():
            return

        if self._poll_rest_full_quote(token):
            return
        self._poll_rest_ltp(token)

    def _poll_rest_full_quote(self, token: str) -> bool:
        """FULL market data carries OI and traded volume, which ltpData does not."""
        try:
            response = self._rest_client.get_market_data(
                "FULL",
                {self.instrument.exchange: [str(token)]},
            )
        except Exception as exc:
            logger.debug("FULL market-data poll unavailable, falling back to LTP: %s", exc)
            return False

        data = response.get("data") if isinstance(response, dict) else None
        fetched = data.get("fetched") if isinstance(data, dict) else None
        if not isinstance(fetched, list) or not fetched:
            return False

        quote = fetched[0]
        if not isinstance(quote, dict):
            return False

        quote = {**quote, "token": str(quote.get("token") or quote.get("symbolToken") or token)}
        self._handle_tick({"data": quote})
        return True

    def _poll_rest_ltp(self, token: str) -> None:
        symbol = os.getenv("MCX_CRUDE_FUTURE_SYMBOL", self.instrument.symbol).strip()
        try:
            response = self._rest_client.get_ltp(
                self.instrument.exchange,
                symbol,
                token,
            )
            data = response.get("data") if isinstance(response, dict) else response
            if isinstance(data, dict):
                self._handle_tick({"data": {**data, "token": str(data.get("token") or data.get("symbolToken") or token)}})
        except Exception as exc:
            logger.error("REST ticker fallback failed: %s", exc)
            self._rest_client = None

    def _resolve_futures_token(self) -> Optional[str]:
        """Resolve the active MCX CRUDEOIL FUTCOM token dynamically.

        Honors an explicit MCX_CRUDE_FUTURE_TOKEN override in the environment
        (useful for testing), otherwise looks up the nearest-expiry CRUDEOIL
        futures contract from the Angel One instrument master so no manual
        token needs to be maintained in .env.
        """
        override = os.getenv("MCX_CRUDE_FUTURE_TOKEN", "").strip()
        if override:
            self._futures_token_cache = override
            return override

        if self._futures_token_cache:
            return self._futures_token_cache

        try:
            from angel_one.instrument_reader import InstrumentReader

            reader = InstrumentReader()
            contract = reader.find_nearest_future(
                self.instrument.symbol,
                instrument_type=self.instrument.future_instrument_type,
                exchange=self.instrument.exchange,
            )
        except Exception as exc:
            logger.error("Failed to auto-fetch MCX Crude futures token: %s", exc)
            return None

        if not contract or not contract.get("token"):
            logger.error(
                "No active %s %s contract found in the Angel One instrument master.",
                self.instrument.symbol,
                self.instrument.future_instrument_type,
            )
            return None

        self._futures_token_cache = str(contract["token"])
        logger.info(
            "Auto-resolved MCX Crude futures token: symbol=%s token=%s expiry=%s",
            contract.get("symbol"),
            self._futures_token_cache,
            contract.get("expiry"),
        )
        return self._futures_token_cache

    def _handle_tick(self, tick_payload: Dict[str, Any]):
        try:
            if not isinstance(tick_payload, dict):
                return

            raw_data = tick_payload.get("data") or tick_payload.get("tokenData") or tick_payload.get("tick")
            if isinstance(raw_data, list):
                raw_data = raw_data[0] if raw_data else {}
            if not isinstance(raw_data, dict):
                raw_data = tick_payload if "last_traded_price" in tick_payload else {}

            if "last_traded_price" in raw_data:
                # SmartWebSocketV2 SNAP_QUOTE payload: prices are paisa-scaled integers.
                ltp = float(raw_data.get("last_traded_price") or 0.0) / 100.0
                oi = float(raw_data.get("open_interest") or 0.0)
                cumulative_volume = float(raw_data.get("volume_trade_for_the_day") or 0.0)
            else:
                ltp = float(raw_data.get("last_price") or raw_data.get("ltp") or raw_data.get("LTP") or 0.0)
                oi = float(raw_data.get("oi") or raw_data.get("OI") or raw_data.get("opnInterest") or raw_data.get("open_interest") or 0.0)
                cumulative_volume = float(
                    raw_data.get("tradeVolume")
                    or raw_data.get("volume")
                    or raw_data.get("totalVolume")
                    or 0.0
                )
            if ltp <= 0:
                return

            # The futures and option legs share one SmartStream connection. Only the
            # active MCX futures token is allowed to update the scanning price.
            token = str(raw_data.get("token") or raw_data.get("symbolToken") or "").strip()
            futures_token = self._resolve_futures_token()
            if not token or not futures_token or token != str(futures_token):
                position = self._position
                if position is not None and token and token == position.option_token:
                    # Guard against the first subscribe-triggered snap-quote right after entry:
                    # it can carry a stale/last-close premium wildly off the fill price, which
                    # would otherwise slam the stop loss within the same second as the entry.
                    since_entry = (_now_ist() - position.entry_time).total_seconds()
                    if (
                        since_entry < self._entry_settle_seconds
                        and position.entry_price
                        and abs(ltp - position.entry_price) > 0.5 * position.entry_price
                    ):
                        logger.warning(
                            "Ignoring implausible option tick %.2f (entry=%.2f) within settle window for %s.",
                            ltp, position.entry_price, position.option_symbol,
                        )
                        return
                    self._option_ltp = ltp
                    self._option_ltp_token = token
                    self._option_ltp_time = time.monotonic()
                return

            now = datetime.now()
            with self._bar_lock:
                volume_delta = self._volume_delta(cumulative_volume)
                self.current_price = ltp
                self.current_oi = oi
                self.current_volume = cumulative_volume
                self.current_timestamp = now
                self._last_tick_time = now

                self._add_tick_to_bar(now, ltp, oi, volume_delta)

            signal = self._evaluate_strategy()
            if signal:
                self._live_signal_queue.put(signal)
        except Exception as exc:  # pragma: no cover
            logger.exception("Tick processing failed: %s", exc)

    def _volume_delta(self, cumulative_volume: float) -> float:
        """Feeds report volume traded for the whole day, so bars need the increment."""
        if cumulative_volume <= 0:
            return 0.0

        previous = self._last_cum_volume
        self._last_cum_volume = cumulative_volume
        if previous is None or cumulative_volume < previous:
            return 0.0
        return cumulative_volume - previous

    def _add_tick_to_bar(self, now: datetime, ltp: float, oi: float, volume: float):
        if not self.futures_bars:
            bar = Bar(timestamp=now, open=ltp, high=ltp, low=ltp, close=ltp, volume=volume, oi=oi)
            self.futures_bars.append(bar)
            return

        current_bar = self.futures_bars[-1]
        bar_time = current_bar.timestamp
        if bar_time.minute == now.minute and bar_time.hour == now.hour and bar_time.date() == now.date():
            current_bar.high = max(current_bar.high, ltp)
            current_bar.low = min(current_bar.low, ltp)
            current_bar.close = ltp
            current_bar.volume += volume
            current_bar.oi = oi
            return

        new_bar = Bar(timestamp=now, open=ltp, high=ltp, low=ltp, close=ltp, volume=volume, oi=oi)
        self.futures_bars.append(new_bar)

    def _evaluate_strategy(self) -> Optional[Dict[str, Any]]:
        # Only one position may be open at a time, so do not build a backlog of signals.
        if self._position is not None:
            self._set_no_signal_reason("position already open")
            return None

        # The newest bar is still forming, so its partial volume would corrupt the spike test.
        completed_bars = list(self.futures_bars)[:-1]
        if len(completed_bars) < 22:
            self._set_no_signal_reason(f"warming up {len(completed_bars)}/22 completed bars")
            return None

        recent = completed_bars[-25:]
        current_bar = recent[-1]
        previous_bar = recent[-2]

        if self._last_signal_bar == current_bar.timestamp:
            self._set_no_signal_reason("current completed bar was already evaluated")
            return None

        current_price = current_bar.close
        previous_close = previous_bar.close
        ois = [bar.oi for bar in recent]
        oi_delta = current_bar.oi - (ois[-2] if len(ois) >= 2 else current_bar.oi)
        history_volumes = [max(float(bar.volume), 0.0) for bar in recent[:-1]]
        ma_volume = float(np.mean(history_volumes[-self.settings.ma_volume_period :]))
        volume_spike = (current_bar.volume > self.settings.volume_spike_factor * ma_volume) and (current_bar.volume > 0)

        futures_up = current_price > previous_close
        futures_down = current_price < previous_close
        oi_up = current_bar.oi > (ois[-2] if len(ois) >= 2 else current_bar.oi)
        oi_down = current_bar.oi < (ois[-2] if len(ois) >= 2 else current_bar.oi)

        # Use the full completed futures history for exact OHLC-derived pivots;
        # do not align or round levels to the option strike grid.
        pivot_bars = completed_bars[-self.settings.pivot_lookback_bars :]
        pivots = self._calculate_daily_pivots(pivot_bars)
        atr_value = self._calculate_atr(recent)
        market_context = self._detect_market_context(recent, atr_value)
        next_pivot = self._next_pivot_for_target(current_price, pivots)

        nymex_trend = self.nymex_filter.trend
        nymex_green = nymex_trend == "GREEN"
        nymex_red = nymex_trend == "RED"

        if not volume_spike:
            required_volume = self.settings.volume_spike_factor * ma_volume
            self._set_no_signal_reason(
                f"volume not spiking: current={current_bar.volume:.0f}, required>{required_volume:.0f}, ma={ma_volume:.0f}"
            )
            return None

        minimum_volume = float(getattr(self.settings, "crude_min_entry_volume", 0.0))
        if current_bar.volume < minimum_volume:
            self._set_no_signal_reason(
                f"adaptive volume gate: current={current_bar.volume:.0f}, minimum={minimum_volume:.0f}"
            )
            return None

        minimum_oi_change = float(getattr(self.settings, "crude_min_abs_oi_change", 0.0))
        if abs(oi_delta) < minimum_oi_change:
            self._set_no_signal_reason(
                f"adaptive OI gate: absolute change={abs(oi_delta):.0f}, minimum={minimum_oi_change:.0f}"
            )
            return None

        minimum_atr = float(getattr(self.settings, "crude_min_entry_atr", 0.0))
        maximum_atr = float(getattr(self.settings, "crude_max_entry_atr", 1_000_000.0))
        if not minimum_atr <= atr_value <= maximum_atr:
            self._set_no_signal_reason(
                f"adaptive ATR gate: current={atr_value:.2f}, range={minimum_atr:.2f}-{maximum_atr:.2f}"
            )
            return None

        allowed_regimes = tuple(
            str(regime).upper()
            for regime in getattr(
                self.settings,
                "crude_allowed_regimes",
                ("TRENDING", "SIDEWAYS", "UNKNOWN"),
            )
        )
        market_regime = str(market_context.get("regime") or "UNKNOWN").upper()
        if market_regime not in allowed_regimes:
            self._set_no_signal_reason(
                f"adaptive regime gate: {market_regime} not in {allowed_regimes}"
            )
            return None

        scenario = None
        side = None

        if futures_up and oi_up and volume_spike and nymex_green:
            scenario = "Long Buildup"
            side = "CE"
        elif futures_down and oi_up and volume_spike and nymex_red:
            scenario = "Short Buildup"
            side = "PE"
        elif futures_up and oi_down and volume_spike and nymex_green:
            scenario = "Short Covering"
            side = "CE"
        elif futures_down and oi_down and volume_spike and nymex_red:
            scenario = "Long Unwinding"
            side = "PE"

        if scenario == "Short Covering":
            self._set_no_signal_reason("Short Covering scenario is disabled")
            return None

        if side is None or scenario is None:
            price_delta = current_price - previous_close
            self._set_no_signal_reason(
                f"direction setup mismatch: price_delta={price_delta:+.2f}, oi_delta={oi_delta:+.0f}, nymex={nymex_trend}"
            )
            return None

        if market_context.get("regime") == "SIDEWAYS" and not self._sideways_entry_ok(current_bar, pivots, side):
            self._set_no_signal_reason(
                f"sideways {side} needs a candle extreme or pivot retest: {market_context.get('reason', '')}"
            )
            return None

        self._last_no_signal_reason = ""

        ai_evaluation = self.ai_dynamic_trade_evaluation(
            current_bar=current_bar,
            previous_bar=previous_bar,
            pivots=pivots,
            atr=atr_value,
            side=side,
        )
        if ai_evaluation is None:
            logger.warning(
                "%s %s proceeding with baseline risk settings; dynamic AI filter did not confirm the setup.",
                scenario,
                side,
            )

        hybrid_evaluation = self._evaluate_hybrid_model(completed_bars, side)
        hybrid_prediction = hybrid_evaluation.action if hybrid_evaluation else "UNAVAILABLE"
        if hybrid_evaluation is None or hybrid_evaluation.action != side:
            logger.warning(
                "%s %s proceeding despite hybrid model warning: prediction=%s, momentum_probability=%s, volatility_probability=%s",
                scenario,
                side,
                hybrid_prediction,
                hybrid_evaluation.momentum_probability if hybrid_evaluation else None,
                hybrid_evaluation.volatility_probability if hybrid_evaluation else None,
            )

        confirmed, pivot_name, pivot_level = self._pivot_breakout(current_price, pivots, side)
        if not confirmed:
            logger.warning(
                "%s %s proceeding without pivot confirmation: price=%.2f, pivot=%s (%.2f).",
                scenario,
                side,
                current_price,
                pivot_name or "n/a",
                pivot_level,
            )

        self._last_signal_bar = current_bar.timestamp
        strike = self._nearest_atm_strike(current_price)
        oi_change = oi_delta if len(ois) >= 2 else None
        return {
            "scenario": scenario,
            "side": side,
            "strike": strike,
            "futures_price": current_price,
            "oi": current_bar.oi,
            "oi_change": oi_change,
            "volume": current_bar.volume,
            "atr": atr_value,
            "pivot": pivots,
            "pivot_level_name": pivot_name,
            "pivot_level": pivot_level,
            "buffer_points": self.settings.buffer_points,
            "next_pivot": next_pivot,
            "nymex_trend": nymex_trend,
            "market_regime": market_context.get("regime"),
            "market_reversal": market_context.get("reversal"),
            "market_context": market_context,
            "ai_evaluation": ai_evaluation,
            "hybrid_prediction": hybrid_prediction,
            "hybrid_momentum_probability": hybrid_evaluation.momentum_probability if hybrid_evaluation else None,
            "hybrid_volatility_probability": hybrid_evaluation.volatility_probability if hybrid_evaluation else None,
            "timestamp": current_bar.timestamp,
        }

    def _evaluate_hybrid_model(self, bars: list[Bar], side: str | None = None):
        if len(bars) < 22:
            return None

        model = self._get_hybrid_model()
        if model is None:
            return None

        import pandas as pd

        frame = pd.DataFrame([asdict(bar) for bar in bars])
        frame["trade_side"] = 1.0 if side == "CE" else -1.0 if side == "PE" else 0.0
        from engine.crude_hybrid_ml import build_features

        features = build_features(frame).dropna()
        if features.empty:
            return None
        try:
            return model.decide(features.iloc[[-1]])
        except Exception as exc:
            logger.warning("Crude hybrid model prediction failed; rule-based signal will continue: %s", exc)
            return None

    def _calculate_daily_pivots(self, bars):
        if not bars:
            return {"PP": 0.0, "R1": 0.0, "R2": 0.0, "S1": 0.0, "S2": 0.0}

        high = max(bar.high for bar in bars)
        low = min(bar.low for bar in bars)
        close = bars[-1].close
        pp = (high + low + close) / 3.0
        r1 = (2 * pp) - low
        s1 = (2 * pp) - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        return {"PP": pp, "R1": r1, "R2": r2, "S1": s1, "S2": s2}

    def _candle_entry_zone_ok(self, candle: Bar, side: str) -> bool:
        """Prefer CE entries near candle lows and PE entries near candle highs."""
        candle_range = float(candle.high) - float(candle.low)
        if candle_range <= 0:
            return False
        zone = min(max(float(getattr(self.settings, "crude_entry_candle_zone", 0.35)), 0.0), 0.5)
        close_position = (float(candle.close) - float(candle.low)) / candle_range
        if side == "CE":
            return close_position <= zone
        return close_position >= 1.0 - zone

    def _sideways_entry_ok(self, candle: Bar, pivots: Dict[str, float], side: str) -> bool:
        """Allow range entries only from a candle extreme or a confirmed S/R retest."""
        if self._candle_entry_zone_ok(candle, side):
            return True

        buffer = float(self.settings.buffer_points)
        close = float(candle.close)
        if side == "CE":
            supports = [
                float(level) for name, level in pivots.items()
                if name in {"S2", "S1", "PP"} and level is not None and float(level) <= close
            ]
            return bool(supports) and float(candle.low) <= max(supports) + buffer

        resistances = [
            float(level) for name, level in pivots.items()
            if name in {"R1", "R2", "PP"} and level is not None and float(level) >= close
        ]
        return bool(resistances) and float(candle.high) >= min(resistances) - buffer

    def ai_dynamic_trade_evaluation(
        self,
        current_bar: Bar,
        previous_bar: Bar,
        pivots: Dict[str, float],
        atr: float,
        side: str,
    ) -> Optional[Dict[str, float | str]]:
        """Evaluate Crude entry quality using momentum, ATR and S/R proximity."""
        atr_value = float(atr or 0.0)
        if atr_value <= 0.0 or current_bar.high <= current_bar.low:
            return None

        close = float(current_bar.close)
        previous_close = float(previous_bar.close)
        momentum = (close - previous_close) / atr_value
        momentum_limit = float(self.settings.crude_ai_max_momentum)
        momentum_score = max(-momentum_limit, min(momentum_limit, momentum))
        minimum_momentum = float(self.settings.crude_ai_min_momentum)

        if side == "CE" and momentum_score < minimum_momentum:
            return None
        if side == "PE" and momentum_score > -minimum_momentum:
            return None
        if not self._candle_entry_zone_ok(current_bar, side):
            return None

        if side == "CE":
            candidates = [
                float(level) for name, level in pivots.items()
                if name in {"S2", "S1", "PP"} and level is not None and float(level) <= close
            ]
            reference_level = max(candidates) if candidates else None
            distance = float(current_bar.low) - reference_level if reference_level is not None else None
        else:
            candidates = [
                float(level) for name, level in pivots.items()
                if name in {"R1", "R2", "PP"} and level is not None and float(level) >= close
            ]
            reference_level = min(candidates) if candidates else None
            distance = reference_level - float(current_bar.high) if reference_level is not None else None

        tolerance = float(self.settings.crude_ai_proximity_tolerance)
        proximity = abs(distance) / max(abs(reference_level), 1.0) if reference_level is not None and distance is not None else None
        if reference_level is None or proximity is None or proximity > tolerance:
            return None

        momentum_strength = min(abs(momentum_score), momentum_limit) / max(momentum_limit, 1.0)
        dynamic_multiplier = float(self.settings.crude_atr_multiplier) * (1.0 + 0.25 * momentum_strength)
        trail_multiplier = dynamic_multiplier * (1.0 + 0.15 * momentum_strength)
        return {
            "momentum": momentum_score,
            "momentum_strength": momentum_strength,
            "reference_level": reference_level,
            "proximity": proximity,
            "tolerance": tolerance,
            "atr_multiplier": dynamic_multiplier,
            "trail_multiplier": trail_multiplier,
            "side": side,
        }

    def _calculate_atr(self, bars, period: int = 14) -> float:
        if len(bars) < 2:
            return 0.0

        tr_values = []
        for idx in range(1, len(bars)):
            prev = bars[idx - 1]
            curr = bars[idx]
            tr = max(
                curr.high - curr.low,
                abs(curr.high - prev.close),
                abs(curr.low - prev.close),
            )
            tr_values.append(tr)

        if len(tr_values) < period:
            return float(np.mean(tr_values)) if tr_values else 0.0
        return float(np.mean(tr_values[-period:]))

    def _detect_market_context(self, bars: list[Bar], atr: float) -> Dict[str, Any]:
        lookback = int(getattr(self.settings, "crude_market_context_lookback_bars", 20))
        swing_lookback = int(getattr(self.settings, "crude_reversal_swing_lookback_bars", 5))
        window = list(bars[-max(lookback, swing_lookback + 1):])
        if len(window) < max(6, swing_lookback + 1):
            return {"regime": "UNKNOWN", "reversal": None, "side": None, "reason": "not enough bars"}

        high = max(float(bar.high) for bar in window)
        low = min(float(bar.low) for bar in window)
        price_range = high - low
        net_move = abs(float(window[-1].close) - float(window[0].open))
        efficiency = net_move / price_range if price_range > 0 else 0.0
        atr_value = max(float(atr or 0.0), 0.01)
        range_multiplier = float(getattr(self.settings, "crude_sideways_range_atr_multiplier", 3.0))
        efficiency_threshold = float(getattr(self.settings, "crude_sideways_efficiency_threshold", 0.25))
        sideways = price_range <= range_multiplier * atr_value or efficiency <= efficiency_threshold

        latest = window[-1]
        swing = window[-swing_lookback - 1:-1]
        swing_low = min(float(bar.low) for bar in swing)
        swing_high = max(float(bar.high) for bar in swing)
        body = abs(float(latest.close) - float(latest.open))
        candle_range = float(latest.high) - float(latest.low)
        lower_wick = min(float(latest.open), float(latest.close)) - float(latest.low)
        upper_wick = float(latest.high) - max(float(latest.open), float(latest.close))
        min_wick = max(body, candle_range * 0.35)

        bullish_reversal = (
            float(latest.low) < swing_low
            and float(latest.close) > swing_low
            and float(latest.close) > float(latest.open)
            and lower_wick >= min_wick
        )
        bearish_reversal = (
            float(latest.high) > swing_high
            and float(latest.close) < swing_high
            and float(latest.close) < float(latest.open)
            and upper_wick >= min_wick
        )

        side = "CE" if bullish_reversal else "PE" if bearish_reversal else None
        reversal = "BULLISH" if bullish_reversal else "BEARISH" if bearish_reversal else None
        return {
            "regime": "SIDEWAYS" if sideways else "TRENDING",
            "reversal": reversal,
            "side": side,
            "range_points": round(price_range, 2),
            "efficiency": round(efficiency, 3),
            "swing_low": round(swing_low, 2),
            "swing_high": round(swing_high, 2),
            "reason": (
                f"range={price_range:.2f}, atr={atr_value:.2f}, efficiency={efficiency:.2f}, "
                f"reversal={reversal or 'NONE'}"
            ),
        }

    def _next_pivot_for_target(self, price: float, pivots: Dict[str, float]) -> float:
        targets = [pivots["R1"], pivots["R2"], pivots["S1"], pivots["S2"]]
        if price >= pivots["PP"]:
            bucket = [pivots["R1"], pivots["R2"]]
        else:
            bucket = [pivots["S1"], pivots["S2"]]
        return min(bucket, key=lambda x: abs(x - price)) if bucket else pivots["PP"]

    @staticmethod
    def _reachable_structure_target(
        futures_price: float,
        pivots: Dict[str, float],
        side: str,
        minimum_distance: float,
        maximum_distance: float,
    ) -> Optional[float]:
        """Return the nearest directional S/R target only when ATR says it is reachable."""
        if side == "CE":
            candidates = [float(level) for name, level in pivots.items()
                          if name in {"R1", "R2"} and level is not None and float(level) > futures_price]
            target = min(candidates, default=None)
            distance = target - futures_price if target is not None else None
        else:
            candidates = [float(level) for name, level in pivots.items()
                          if name in {"S1", "S2"} and level is not None and float(level) < futures_price]
            target = max(candidates, default=None)
            distance = futures_price - target if target is not None else None

        if distance is None or distance < minimum_distance or distance > maximum_distance:
            return None
        return target

    @staticmethod
    def _nearest_pivot_info(price: float, pivots: Dict[str, float]) -> tuple[Optional[str], Optional[int], Optional[float]]:
        numbers = {"S2": -2, "S1": -1, "PP": 0, "R1": 1, "R2": 2}
        candidates = [
            (name, numbers[name], float(level))
            for name, level in pivots.items()
            if name in numbers and level is not None
        ]
        if not candidates:
            return None, None, None
        return min(candidates, key=lambda item: abs(item[2] - float(price)))

    def _pivot_breakout(self, price: float, pivots: Dict[str, float], side: str) -> tuple[bool, Optional[str], float]:
        """Require price to remain inside an exact OHLC S/R zone."""

        support_candidates = [
            (name, float(level)) for name, level in pivots.items()
            if name in {"S2", "S1"} and level is not None and float(level) < float(price)
        ]
        resistance_candidates = [
            (name, float(level)) for name, level in pivots.items()
            if name in {"R1", "R2"} and level is not None and float(level) > float(price)
        ]
        if not support_candidates or not resistance_candidates:
            return False, None, 0.0

        support_name, support = max(support_candidates, key=lambda item: item[1])
        resistance_name, resistance = min(resistance_candidates, key=lambda item: item[1])
        market_inside_zone = support < float(price) < resistance
        if not market_inside_zone:
            logger.debug(
                "Crude S/R zone rejected: support=%.4f resistance=%.4f price=%.4f",
                support, resistance, price,
            )
            return False, None, 0.0

        # All five exact pivot levels may be the entry reference when the
        # scenario and AI gate agree.
        if side == "CE":
            eligible = [
                (name, float(level)) for name, level in pivots.items()
                if name in {"S2", "S1", "PP"} and level is not None and float(level) <= float(price)
            ]
            if not eligible:
                return False, None, 0.0
            name, level = max(eligible, key=lambda item: item[1])
            return True, name, level

        eligible = [
            (name, float(level)) for name, level in pivots.items()
            if name in {"PP", "R1", "R2"} and level is not None and float(level) >= float(price)
        ]
        if not eligible:
            return False, None, 0.0
        name, level = min(eligible, key=lambda item: item[1])
        return True, name, level

    def _nearest_atm_strike(self, futures_price: float) -> int:
        """Choose the closest currently listed MCX Crude option strike.

        The instrument master stores MCX strikes multiplied by 100.  Rounding
        a bad or differently-scaled futures tick directly can otherwise create
        non-existent strikes such as 350, which have no option LTP.
        """
        reference_price = float(futures_price)
        previous_close = getattr(self, "_last_close", None)
        if reference_price < 1000 and previous_close is not None:
            try:
                previous_close = float(previous_close)
                if previous_close >= 1000:
                    reference_price = previous_close
            except (TypeError, ValueError):
                pass
        try:
            from angel_one.instrument_reader import InstrumentReader

            reader = InstrumentReader()
            reader.load()
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            strikes: set[int] = set()
            for contract in reader.instruments:
                if (
                    str(contract.get("name", "")).upper() != self.instrument.symbol.upper()
                    or str(contract.get("instrumenttype", "")).upper()
                    != self.instrument.option_instrument_type.upper()
                    or str(contract.get("exch_seg", "")).upper()
                    != self.settings.option_exchange.upper()
                    or not str(contract.get("symbol", "")).upper().endswith(("CE", "PE"))
                ):
                    continue
                try:
                    expiry = datetime.strptime(str(contract.get("expiry", "")), "%d%b%Y")
                    strike = int(round(float(contract.get("strike", 0)) / 100.0))
                except (TypeError, ValueError):
                    continue
                if expiry >= today and strike >= 1000:
                    strikes.add(strike)

            if strikes:
                if reference_price < 1000:
                    # No credible futures level yet: prefer the center of the
                    # active listed chain over manufacturing a 3-digit strike.
                    ordered_strikes = sorted(strikes)
                    reference_price = ordered_strikes[len(ordered_strikes) // 2]
                    logger.warning(
                        "Invalid MCX Crude futures price %.2f for strike selection; using listed-chain center %.0f.",
                        futures_price,
                        reference_price,
                    )
                return min(strikes, key=lambda strike: abs(strike - reference_price))
        except Exception as exc:
            logger.debug("Could not load listed MCX Crude strikes; using price grid: %s", exc)

        step = self.instrument.strike_step
        rounded = round(reference_price / step) * step
        return int(rounded)

    def _warn_option_ltp_unavailable(self, contract: Dict[str, Any], message: str, *args: Any) -> None:
        """Log one temporary option-LTP warning per contract each minute."""
        key = str(contract.get("token") or contract.get("symbol") or "unknown-option")
        now = time.monotonic()
        warnings = getattr(self, "_option_ltp_warnings", {})
        interval = getattr(self, "_option_ltp_warning_interval", 60.0)
        if now - warnings.get(key, float("-inf")) < interval:
            return
        warnings[key] = now
        self._option_ltp_warnings = warnings
        logger.warning(message, *args)

    def _clear_option_ltp_warning(self, contract: Dict[str, Any]) -> None:
        key = str(contract.get("token") or contract.get("symbol") or "unknown-option")
        warnings = getattr(self, "_option_ltp_warnings", None)
        if warnings is not None:
            warnings.pop(key, None)

    def _write_paper_trade_log(self, record: Dict[str, Any]):
        entry = {"timestamp": _format_ist_timestamp(), **record}
        self.paper_trade_history.append(entry)
        with self.paper_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")

        self.paper_trade_count = len(self.paper_trade_history)
        summary = {
            "total_paper_records": self.paper_trade_count,
            "realized_pnl": round(self.paper_realized_pnl, 2),
            "wins": self.paper_wins,
            "losses": self.paper_losses,
            "latest_status": entry.get("status"),
            "latest_scenario": entry.get("scenario"),
            "latest_side": entry.get("side"),
            "latest_strike": entry.get("strike"),
        }
        self.paper_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    @staticmethod
    def _print_block(title: str, rows: list[tuple[str, Any]]):
        _clear_status_line()
        width = 62
        print()
        print("=" * width)
        print(title.center(width))
        print("=" * width)
        for label, value in rows:
            print(f"  {label:<18}: {value}")
        print("=" * width, flush=True)

    def _current_option_price(self, position: Position) -> float:
        """Live broker premium: websocket tick first, then a throttled REST quote."""
        now = time.monotonic()
        if (
            self._option_ltp is not None
            and self._option_ltp_token == position.option_token
            and (now - self._option_ltp_time) <= self._option_ltp_max_age
        ):
            return self._option_ltp

        if position.option_token and (now - self._last_option_poll) >= self._option_poll_interval:
            self._last_option_poll = now
            live = self._fetch_option_ltp(
                {"symbol": position.option_symbol, "token": position.option_token}
            )
            if live:
                self._option_ltp = live
                self._option_ltp_token = position.option_token
                self._option_ltp_time = now
                return live

        # Hold only this contract's last real quote rather than modelling a price the market never printed.
        if self._option_ltp is not None and self._option_ltp_token == position.option_token:
            return self._option_ltp
        return position.entry_price

    @staticmethod
    def _parse_ist_time(value: str, fallback: dt_time) -> dt_time:
        try:
            hour, minute = value.split(":")
            return dt_time(int(hour), int(minute))
        except Exception:
            return fallback

    def _square_off_time(self) -> dt_time:
        close = self._parse_ist_time(self.settings.market_close_time_ist, dt_time(23, 30))
        anchor = datetime.combine(datetime.now(IST).date(), close)
        return (anchor - timedelta(minutes=self.settings.square_off_buffer_minutes)).time()

    def _entry_cutoff_time(self) -> dt_time:
        """Entries also stop at the square-off boundary, whichever comes first."""
        trade_end = self._parse_ist_time(self.settings.trade_end_time_ist, dt_time(23, 15))
        return min(trade_end, self._square_off_time())

    def _is_square_off_window(self) -> bool:
        return datetime.now(IST).time() >= self._square_off_time()

    def _enforce_session_square_off(self):
        """No position is ever carried overnight; flatten inside the closing window."""
        if not self._is_square_off_window():
            self._square_off_done = False
            return

        if self._position is not None:
            option_price = self._current_option_price(self._position)
            self._close_position(option_price, "DAY CLOSE SQUARE-OFF")

        if not self._square_off_done:
            self._square_off_done = True
            logger.info(
                "Square-off window reached (%s IST); all positions flat and new entries blocked for today.",
                self._square_off_time().strftime("%H:%M"),
            )

        # Drop signals raised just before the cutoff so they cannot fire later.
        while not self._live_signal_queue.empty():
            self._live_signal_queue.get()

    def _resolve_option_contract(self, strike: int, side: str) -> Optional[Dict[str, Any]]:
        try:
            from angel_one.instrument_reader import InstrumentReader

            reader = InstrumentReader()
            return reader.find_option_token(
                self.instrument.symbol,
                strike,
                side,
                instrument_type=self.instrument.option_instrument_type,
                exchange=self.settings.option_exchange,
            )
        except Exception as exc:
            logger.error("Failed to resolve %s %s option contract: %s", strike, side, exc)
            return None

    def _submit_real_order(self, transaction_type: str, contract: Dict[str, Any], quantity: int) -> Optional[str]:
        if not self._ensure_rest_client():
            logger.error("Cannot submit %s order: Angel One session unavailable.", transaction_type)
            return None

        params = {
            "variety": "NORMAL",
            "tradingsymbol": contract.get("symbol"),
            "symboltoken": str(contract.get("token")),
            "transactiontype": transaction_type,
            "exchange": self.settings.option_exchange,
            "ordertype": "MARKET",
            "producttype": "CARRYFORWARD",
            "duration": "DAY",
            "price": "0",
            "squareoff": "0",
            "stoploss": "0",
            "quantity": str(quantity),
        }
        try:
            response = self._rest_client.get_api().placeOrder(params)
        except Exception as exc:
            logger.exception("Angel One %s order failed: %s", transaction_type, exc)
            return None

        order_id = response.get("data", {}).get("orderid") if isinstance(response, dict) else response
        if not order_id:
            logger.error("Angel One %s order returned no order id: %s", transaction_type, response)
            return None

        logger.info("Angel One %s MARKET order accepted: id=%s symbol=%s qty=%s",
                    transaction_type, order_id, contract.get("symbol"), quantity)
        return str(order_id)

    def _risk_levels(
        self,
        entry_price: float,
        atr: float,
        dynamic_evaluation: Optional[Dict[str, float | str]] = None,
    ) -> Optional[Dict[str, float]]:
        """Stop keeps the full ATR+buffer cushion; a premium too thin to hold it is skipped."""
        atr_multiplier = float(
            (dynamic_evaluation or {}).get(
                "atr_multiplier", self.settings.crude_atr_multiplier
            )
        )
        trail_multiplier = float(
            (dynamic_evaluation or {}).get("trail_multiplier", atr_multiplier)
        )
        stop_distance = atr_multiplier * atr + self.settings.buffer_points
        trail_distance = trail_multiplier * atr + 5.0
        min_premium = stop_distance / self.settings.max_stop_premium_fraction

        if entry_price < min_premium:
            logger.info(
                "Entry skipped: premium %.2f cannot hold a %.2f point stop (needs >= %.2f).",
                entry_price,
                stop_distance,
                min_premium,
            )
            return None

        # Buffer protects the stop from noise; it must not enlarge the target.
        risk_reward_ratio = float(getattr(self.settings, "crude_risk_reward_ratio", 2.0))
        atr_target_distance = risk_reward_ratio * atr_multiplier * atr

        return {
            "stop_distance": stop_distance,
            "trail_distance": trail_distance,
            "stop_loss": entry_price - stop_distance,
            "target_price": entry_price + atr_target_distance,
            "minimum_target_distance": 1.5 * atr_multiplier * atr,
            "maximum_target_distance": atr_target_distance,
        }

    def _place_entry_order(self, signal: Dict[str, Any]):
        if self._position is not None:
            return

        strike = int(signal["strike"])
        option_type = str(signal["side"]).upper()
        if option_type not in {"CE", "PE"}:
            logger.error("Entry skipped: unsupported option side %r.", signal["side"])
            return
        price = float(signal["futures_price"])
        atr = float(signal.get("atr") or 0.0)

        now_ist = datetime.now(IST).time()
        if now_ist >= self._entry_cutoff_time():
            logger.info(
                "Entry skipped: %s IST is past the %s IST cutoff; nothing is carried to the next session.",
                now_ist.strftime("%H:%M"),
                self._entry_cutoff_time().strftime("%H:%M"),
            )
            return

        if self.settings.execution_mode.upper() == "REAL" and not self.settings.allow_real_trading:
            logger.warning(
                "REAL mode is disabled in config. Paper-only mode remains active for safety. "
                "Set allow_real_trading=True only after verifying the environment."
            )
            return

        is_real = self.settings.execution_mode.upper() == "REAL" and self.settings.allow_real_trading
        order_id = ""

        # Paper mode prices the same live contract as real mode, so the premium the
        # engine reports always matches what the broker terminal shows.
        contract = self._resolve_option_contract(strike, option_type) or {}
        if is_real and (not self.settings.angel_api_key or not contract.get("token")):
            logger.error(
                "REAL entry aborted: %s.",
                "credentials missing" if not self.settings.angel_api_key
                else f"no {self.instrument.symbol} {strike} {option_type} contract in the instrument master",
            )
            return

        quantity = int(float(contract.get("lotsize") or self.settings.option_lot_size))
        entry_option_price = self._entry_option_ltp(contract) if contract.get("token") else None
        if not entry_option_price:
            # Paper results are only meaningful at a real tradable premium, so never invent one.
            self._warn_option_ltp_unavailable(
                contract,
                "Entry skipped: no broker LTP for %s %s %s.",
                self.instrument.symbol, strike, option_type,
            )
            return

        levels = self._risk_levels(
            entry_option_price,
            atr,
            dynamic_evaluation=signal.get("ai_evaluation"),
        )
        if levels is None:
            return

        stop_loss = levels["stop_loss"]
        target_price = levels["target_price"]
        trail_distance = levels["trail_distance"]
        entry_index_value = float(self.current_price or price)
        entry_pivots = signal.get("pivot") or {}
        minimum_target_distance = levels.get("minimum_target_distance")
        maximum_target_distance = levels.get("maximum_target_distance")
        futures_target = (
            self._reachable_structure_target(
                entry_index_value,
                entry_pivots,
                option_type,
                float(minimum_target_distance),
                float(maximum_target_distance),
            )
            if minimum_target_distance is not None and maximum_target_distance is not None
            else None
        )
        target_source = "S&R" if futures_target is not None else "ATR"
        entry_nearest_pivot, entry_pivot_number, entry_pivot_price = self._nearest_pivot_info(
            entry_index_value,
            entry_pivots,
        )
        entry_candle = signal.get("entry_candle")
        if entry_candle is None and getattr(self, "futures_bars", None):
            entry_candle = self.futures_bars[-1]

        if is_real:
            order_id = self._submit_real_order("BUY", contract, quantity) or ""
            if not order_id:
                logger.error("REAL entry aborted: broker rejected the BUY order.")
                return

        self._position = Position(
            side=option_type,
            strike=strike,
            entry_price=entry_option_price,
            entry_time=_now_ist(),
            stop_loss=stop_loss,
            target_price=target_price,
            trailing_stop=stop_loss,
            atr_value=atr,
            futures_entry=price,
            futures_entry_oi=float(signal.get("oi") or 0.0),
            scenario=str(signal["scenario"]),
            is_real=is_real,
            option_token=str(contract.get("token", "")),
            option_symbol=str(contract.get("symbol", f"{self.instrument.symbol} {strike} {option_type}")),
            quantity=quantity,
            entry_order_id=order_id,
            entry_volume=signal.get("volume"),
            entry_oi_change=signal.get("oi_change"),
            pivot_level=signal.get("pivot_level"),
            nymex_trend=str(signal.get("nymex_trend") or "NEUTRAL"),
            entry_index_value=entry_index_value,
            entry_nearest_pivot=entry_nearest_pivot or signal.get("entry_nearest_pivot"),
            entry_pivot_number=entry_pivot_number if entry_pivot_number is not None else signal.get("entry_pivot_number"),
            entry_pivot_price=entry_pivot_price if entry_pivot_price is not None else signal.get("entry_pivot_price"),
            entry_candle_open=getattr(entry_candle, "open", None),
            entry_candle_high=getattr(entry_candle, "high", None),
            entry_candle_low=getattr(entry_candle, "low", None),
            entry_candle_close=getattr(entry_candle, "close", None),
            futures_target=futures_target,
            target_source=target_source,
            entry_market_regime=signal.get("market_regime"),
            entry_momentum_strength=float(
                (signal.get("ai_evaluation") or {}).get("momentum_strength", 0.0)
            ),
            intratrade_option_prices=[entry_option_price],
        )
        self._trail_distance = trail_distance
        self._option_ltp = entry_option_price
        self._option_ltp_token = self._position.option_token
        self._option_ltp_time = time.monotonic()
        if self._position.option_token:
            self._smart_stream.subscribe_futures(
                self._position.option_token, exchange=self.settings.option_exchange
            )

        mode = "REAL" if is_real else "PAPER"
        self._print_block(
            f"{mode} BUY {option_type}  |  {self._position.option_symbol}",
            [
                ("Time", _format_ist_timestamp(self._position.entry_time)),
                ("Scenario", signal["scenario"]),
                ("Side", option_type),
                ("Strike", strike),
                ("Option expiry", contract.get("expiry", "-")),
                ("Futures LTP", f"{price:.2f}"),
                ("Entry premium", f"{entry_option_price:.2f}"),
                ("Premium source", "broker LTP"),
                ("Stop loss", f"{stop_loss:.2f}"),
                ("Target", f"{futures_target:.2f} futures S&R" if futures_target is not None else f"{target_price:.2f} option ATR"),
                ("Quantity", quantity),
                ("Order id", order_id or "-"),
                ("ATR", f"{atr:.2f}"),
                ("Open interest", f"{signal.get('oi', 0):,.0f}"),
                ("Bar volume", f"{signal.get('volume', 0):,.0f}"),
                ("NYMEX trend", signal.get("nymex_trend")),
                ("Market regime", signal.get("market_regime", "UNKNOWN")),
                ("Reversal", signal.get("market_reversal") or "NONE"),
                ("Pivot cleared", f"{signal.get('pivot_level_name', '-')} {signal.get('pivot_level', 0):.2f} "
                                  f"(+/-{signal.get('buffer_points', 0):.1f} buffer)"),
                ("Next pivot", f"{signal.get('next_pivot', 0):.2f}"),
                ("Square-off at", f"{self._square_off_time().strftime('%H:%M')} IST"),
            ],
        )
        logger.info(
            "%s ENTRY %s %s @ %.2f | scenario=%s futures=%.2f stop=%.2f target=%s",
            mode,
            option_type,
            strike,
            entry_option_price,
            signal["scenario"],
            price,
            stop_loss,
            f"{futures_target:.2f} S&R" if futures_target is not None else f"{target_price:.2f} ATR",
        )
        self._write_paper_trade_log(
            {
                "status": "entry",
                "mode": mode,
                "scenario": signal.get("scenario"),
                "side": option_type,
                "strike": strike,
                "symbol": self._position.option_symbol,
                "order_id": order_id,
                "futures_price": price,
                "entry_premium": round(entry_option_price, 2),
                "stop_loss": round(stop_loss, 2),
                "target": round(target_price, 2),
                "futures_target": futures_target,
                "target_source": target_source,
                "qty": quantity,
                "oi": signal.get("oi"),
                "volume": signal.get("volume"),
                "nymex_trend": signal.get("nymex_trend"),
                "market_regime": signal.get("market_regime"),
                "market_reversal": signal.get("market_reversal"),
                "atr": signal.get("atr"),
                "pivot_level_name": signal.get("pivot_level_name"),
                "pivot_level": signal.get("pivot_level"),
                "buffer_points": signal.get("buffer_points"),
                "next_pivot": signal.get("next_pivot"),
            }
        )
        self._persist_state()

    def _entry_option_ltp(self, contract: Dict[str, Any]) -> Optional[float]:
        """Use a fresh matching option tick, then immediately fall back to SmartAPI ltpData."""
        token = str(contract.get("token") or "")
        if not token:
            return None

        now = time.monotonic()
        # A websocket value of 0 is treated as missing so the REST fallback runs.
        if (
            self._option_ltp
            and self._option_ltp_token == token
            and (now - self._option_ltp_time) <= self._option_ltp_max_age
        ):
            return self._option_ltp

        # Subscribe before the REST call so subsequent price management uses ticks.
        self._smart_stream.subscribe_futures(token, exchange=self.settings.option_exchange)
        live = self._fetch_option_ltp(contract)
        if live is not None:
            self._option_ltp = live
            self._option_ltp_token = token
            self._option_ltp_time = now
            self._clear_option_ltp_warning(contract)
        return live

    def _fetch_option_ltp(self, contract: Dict[str, Any]) -> Optional[float]:
        """REST fallback chain for the option premium: ltpData, then the FULL quote.

        ltpData occasionally throttles (AB1004) or returns an empty body while other
        pollers are active, so transient failures are retried before the alternate
        marketData endpoint is tried. Only a real traded price is accepted; a 0
        means the strike has not printed and the entry still stays skipped.
        """
        if not self._ensure_rest_client():
            return None

        from angel_one.market_data import is_rate_limit_error

        exchange = self.settings.option_exchange
        symbol = contract.get("symbol")
        token = str(contract.get("token"))

        for attempt in range(3):
            try:
                if hasattr(self._rest_client, "get_ltp"):
                    response = self._rest_client.get_ltp(exchange, symbol, token)
                else:
                    response = self._rest_client.get_api().ltpData(exchange, symbol, token)
                ltp = self._parse_ltp_response(response)
                if ltp is not None:
                    return ltp
            except Exception as exc:
                self._warn_option_ltp_unavailable(
                    contract,
                    "Could not fetch option LTP for %s (attempt %d/3): %s",
                    symbol, attempt + 1, exc,
                )
                if not is_rate_limit_error(exc):
                    break
            if attempt < 2:
                time.sleep(0.75 * (attempt + 1))

        ltp = self._fetch_option_quote(exchange, symbol, token)
        if ltp is not None:
            self._clear_option_ltp_warning(contract)
        return ltp

    @staticmethod
    def _parse_ltp_response(response: Any) -> Optional[float]:
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, dict):
            return None
        ltp = float(data.get("ltp") or data.get("last_traded_price") or 0.0)
        if data.get("last_traded_price") is not None and data.get("ltp") is None:
            ltp /= 100.0
        return ltp if ltp > 0 else None

    def _fetch_option_quote(self, exchange: str, symbol: Any, token: str) -> Optional[float]:
        """FULL market-data quote: a separate endpoint that often still carries the
        last traded price when ltpData is throttled or momentarily empty."""
        try:
            response = self._rest_client.get_market_data("FULL", {exchange: [token]})
        except Exception as exc:
            self._warn_option_ltp_unavailable(
                {"symbol": symbol, "token": token},
                "Could not fetch FULL option quote for %s: %s",
                symbol,
                exc,
            )
            return None
        data = response.get("data") if isinstance(response, dict) else None
        fetched = data.get("fetched") if isinstance(data, dict) else None
        if not isinstance(fetched, list) or not fetched or not isinstance(fetched[0], dict):
            return None
        return self._parse_ltp_response({"data": fetched[0]})

    def _close_position(self, option_price: float, reason: str):
        position = self._position
        if position is None:
            return

        exit_order_id = ""
        if position.is_real:
            contract = {"symbol": position.option_symbol, "token": position.option_token}
            exit_order_id = self._submit_real_order("SELL", contract, position.quantity) or ""
            if not exit_order_id:
                logger.error(
                    "EXIT ORDER REJECTED for %s (%s). Position stays open and will be retried; "
                    "square off manually if this repeats near market close.",
                    position.option_symbol,
                    reason,
                )
                return
            option_price = self._fetch_option_ltp(contract) or option_price

        qty = position.quantity or self.settings.option_lot_size
        pnl_points = option_price - position.entry_price
        pnl_amount = pnl_points * qty
        self.paper_realized_pnl += pnl_amount
        if pnl_amount >= 0:
            self.paper_wins += 1
        else:
            self.paper_losses += 1

        mode = "REAL" if position.is_real else "PAPER"
        exit_time = _now_ist()
        held = exit_time - _as_ist(position.entry_time)
        self._print_block(
            f"{mode} EXIT {position.side}  |  {reason}",
            [
                ("Time", _format_ist_timestamp(exit_time)),
                ("Scenario", position.scenario),
                ("Symbol", position.option_symbol),
                ("Side / Strike", f"{position.side} {position.strike}"),
                ("Entry premium", f"{position.entry_price:.2f}"),
                ("Exit premium", f"{option_price:.2f}"),
                ("Points", f"{pnl_points:+.2f}"),
                ("Quantity", qty),
                ("P&L", f"{pnl_amount:+,.2f}"),
                ("Order id", exit_order_id or "-"),
                ("Held for", str(held).split(".")[0]),
                ("Total P&L", f"{self.paper_realized_pnl:+,.2f}"),
                ("Win / Loss", f"{self.paper_wins} / {self.paper_losses}"),
            ],
        )
        logger.info(
            "%s EXIT %s %s @ %.2f (%s) | pnl=%+.2f | total=%+.2f",
            mode,
            position.side,
            position.strike,
            option_price,
            reason,
            pnl_amount,
            self.paper_realized_pnl,
        )
        self._write_paper_trade_log(
            {
                "status": "exit",
                "mode": mode,
                "reason": reason,
                "scenario": position.scenario,
                "side": position.side,
                "strike": position.strike,
                "symbol": position.option_symbol,
                "order_id": exit_order_id,
                "entry_premium": round(position.entry_price, 2),
                "exit_premium": round(option_price, 2),
                "futures_price": self.current_price,
                "qty": qty,
                "pnl_points": round(pnl_points, 2),
                "pnl_amount": round(pnl_amount, 2),
                "total_pnl": round(self.paper_realized_pnl, 2),
            }
        )
        current_timestamp_str = _format_ist_timestamp(exit_time)
        option_type = str(position.side).upper()
        symbol = position.option_symbol or f"{self.instrument.symbol} {position.strike} {option_type}"
        entry_price = position.entry_price
        exit_price = option_price
        pnl = pnl_amount
        current_oi = self.current_oi
        exit_oi_change = (
            current_oi - position.futures_entry_oi
            if current_oi is not None
            else None
        )
        bar_volume = self.futures_bars[-1].volume if getattr(self, "futures_bars", None) else None
        exit_bar = self.futures_bars[-1] if getattr(self, "futures_bars", None) else None
        exit_atr = (
            self._calculate_atr(list(self.futures_bars))
            if exit_bar and len(self.futures_bars) >= 2
            else None
        )
        exit_market_context = (
            self._detect_market_context(list(self.futures_bars), exit_atr)
            if exit_bar and exit_atr is not None
            else {}
        )
        exit_index_value = self.current_price
        exit_pivots = self._calculate_daily_pivots(list(self.futures_bars)) if exit_bar else {}
        exit_nearest_pivot, exit_pivot_number, exit_pivot_price = (
            self._nearest_pivot_info(exit_index_value, exit_pivots)
            if exit_index_value is not None and exit_pivots
            else (None, None, None)
        )
        exit_nymex_trend = getattr(getattr(self, "nymex_filter", None), "trend", None)
        trade_info = {
            "Entry Timestamp": _format_ist_timestamp(position.entry_time),
            "Exit Timestamp": current_timestamp_str,
            "Symbol": symbol,
            "Action": option_type,
            "Entry Price": entry_price,
            "Exit Price": exit_price,
            "SL": position.stop_loss,
            "TP": position.target_price,
            "PnL": pnl,
            "Entry Scenario": position.scenario,
            "Exit Scenario": reason,
            "Entry OI": position.futures_entry_oi,
            "Exit OI": current_oi,
            "Entry OI_Change": position.entry_oi_change,
            "Exit OI_Change": exit_oi_change,
            "Entry NYMEX_Trend": position.nymex_trend,
            "Exit NYMEX_Trend": exit_nymex_trend,
            "Entry Volume": position.entry_volume,
            "Exit Volume": bar_volume,
            "Entry Index Value": position.entry_index_value,
            "Entry Nearest Pivot": position.entry_nearest_pivot,
            "Entry Pivot Number": position.entry_pivot_number,
            "Entry Pivot Price": position.entry_pivot_price,
            "Entry Candle Open": position.entry_candle_open,
            "Entry Candle High": position.entry_candle_high,
            "Entry Candle Low": position.entry_candle_low,
            "Entry Candle Close": position.entry_candle_close,
            "Exit Index Value": exit_index_value,
            "Exit Nearest Pivot": exit_nearest_pivot,
            "Exit Pivot Number": exit_pivot_number,
            "Exit Pivot Price": exit_pivot_price,
            "Exit Candle Open": getattr(exit_bar, "open", None),
            "Exit Candle High": getattr(exit_bar, "high", None),
            "Exit Candle Low": getattr(exit_bar, "low", None),
            "Exit Candle Close": getattr(exit_bar, "close", None),
            "Entry ATR": position.atr_value,
            "Exit ATR": exit_atr,
            "Entry Market Regime": position.entry_market_regime,
            "Exit Market Regime": exit_market_context.get("regime"),
            "Entry Momentum Strength": position.entry_momentum_strength,
            "Intratrade Option Prices": json.dumps(position.intratrade_option_prices),
        }
        # Clear and checkpoint the local position before the external Sheet call.
        # A slow or failed network logger must not leave a closed trade restorable.
        self._position = None
        self._option_ltp = None
        self._option_ltp_token = ""
        self._option_ltp_time = 0.0
        self._persist_state()
        try:
            log_trade(trade_info)
        except Exception:
            logger.exception("Failed to log crude trade exit to Google Sheets.")

    def _update_position_management(self):
        if self._position is None or self.current_price is None:
            return

        position = self._position
        # Never square off within the entry settle window: broker snap-quotes and
        # throttled REST polls right after entry can momentarily report a stale
        # premium far from the real fill, which must not be mistaken for a stop hit.
        settle_seconds = float(getattr(self, "_entry_settle_seconds", 0.0))
        if (_now_ist() - _as_ist(position.entry_time)).total_seconds() < settle_seconds:
            return
        option_price = self._current_option_price(position)
        if (
            len(position.intratrade_option_prices) < 5_000
            and (
                not position.intratrade_option_prices
                or position.intratrade_option_prices[-1] != option_price
            )
        ):
            position.intratrade_option_prices.append(option_price)

        activation_points = float(getattr(self.settings, "crude_trailing_activation_points", 8.0))
        activation_reached = option_price >= position.entry_price + activation_points
        if activation_reached:
            previous_trailing_stop = position.trailing_stop
            buffer_points = max(0.0, float(self.settings.crude_trailing_breakeven_buffer_points))
            position.trailing_stop = max(
                position.trailing_stop,
                min(position.entry_price + buffer_points, option_price),
                option_price - self._trail_distance,
            )
            if position.trailing_stop != previous_trailing_stop:
                self._persist_state()

        trailing_active = position.trailing_stop > position.stop_loss

        futures_target = position.futures_target
        if futures_target is not None and (
            (position.side == "CE" and self.current_price >= futures_target)
            or (position.side == "PE" and self.current_price <= futures_target)
        ):
            self._close_position(option_price, "S&R TARGET HIT")
            return

        if futures_target is None and option_price >= position.target_price:
            self._close_position(option_price, "TARGET HIT")
            return

        if option_price <= position.stop_loss:
            self._close_position(option_price, "STOP LOSS")
            return

        if trailing_active and option_price <= position.trailing_stop:
            self._close_position(option_price, "TRAILING STOP")
            return

    def _process_live_cycle(self):
        self._refresh_adaptive_config()
        if not self._smart_stream.connected and not self._rest_fallback:
            self._enable_rest_fallback("websocket disconnected")

        if not self._smart_stream.connected:
            self._reconnect_websocket_if_due()

        if not self._smart_stream.queue.empty():
            payload = self._smart_stream.queue.get()
            self._handle_tick(payload)

        if self._rest_fallback:
            now = time.monotonic()
            if now - self._last_rest_poll >= self._rest_poll_interval:
                self._last_rest_poll = now
                self._poll_rest_ticker()

        self._enforce_session_square_off()

        while not self._live_signal_queue.empty():
            signal = self._live_signal_queue.get()
            # One position at a time: remaining signals are discarded, not queued.
            if self._position is None:
                self._place_entry_order(signal)

        self._update_position_management()

        self._heartbeat_monitor()
        self._print_status()

        time.sleep(0.1)

    def _print_status(self):
        now = time.monotonic()
        if now - self._last_status_print < self._status_interval:
            return
        self._last_status_print = now

        if self.current_price is None:
            state = "waiting for first tick"
            price_text = "--"
        else:
            price_text = f"{self.current_price:.2f}"
            completed = max(len(self.futures_bars) - 1, 0)
            if completed < 22:
                state = f"warming up {completed}/22 bars"
            elif self._position is not None:
                position = self._position
                option_price = self._current_option_price(position)
                state = (
                    f"IN {position.side} {position.strike} @ {position.entry_price:.2f} "
                    f"now {option_price:.2f} ({option_price - position.entry_price:+.2f}) "
                    f"sl {max(position.stop_loss, position.trailing_stop):.2f} "
                    f"tgt {position.target_price:.2f}"
                )
            else:
                last_reason = getattr(self, "_last_no_signal_reason", "")
                state = "scanning for CE/PE setup"
                if last_reason:
                    state += f" | last block: {last_reason}"

        line = (
            f"[{datetime.now(IST).strftime('%H:%M:%S')}] {self.instrument.symbol} {price_text} | "
            f"{state} | P&L {self.paper_realized_pnl:+,.2f} | "
            f"OI {self.current_oi or 0:,.0f} | vol {self.current_volume or 0:,.0f} | "
            f"NYMEX {self.nymex_filter.trend}"
        )

        global _status_line_width
        # A wrapped line breaks the carriage-return rewrite, so keep it inside the terminal width.
        max_width = max(shutil.get_terminal_size(fallback=(120, 24)).columns - 1, 40)
        if len(line) > max_width:
            line = line[: max_width - 1] + "\u2026"

        padding = max(_status_line_width - len(line), 0)
        sys.stdout.write("\r" + line + " " * padding)
        sys.stdout.flush()
        _status_line_width = len(line)

    def _heartbeat_monitor(self):
        if self._last_tick_time is None:
            self._last_tick_time = datetime.now()
            return

        stale_seconds = (datetime.now() - self._last_tick_time).total_seconds()
        if stale_seconds <= self.settings.futures_tick_timeout_seconds:
            self._stale_feed_since = None
            return

        if self._stale_feed_since is None:
            self._stale_feed_since = datetime.now()
            logger.warning(
                "No MCX crude futures tick for %.0fs; attempting feed recovery.", stale_seconds
            )
            self._recover_futures_feed()
            return

        # Already recovering: only re-log periodically so the loop does not spam.
        if (datetime.now() - self._stale_feed_since).total_seconds() >= self._stale_log_interval:
            self._stale_feed_since = datetime.now()
            logger.warning(
                "MCX crude futures feed still stale for %.0fs (rest_fallback=%s, ws_connected=%s).",
                stale_seconds,
                self._rest_fallback,
                self._smart_stream.connected,
            )
            self._recover_futures_feed()

    def _recover_futures_feed(self) -> None:
        """Re-subscribe only the active MCX Crude futures token after a timeout."""
        token = self._resolve_futures_token()

        if self._smart_stream.connected and token:
            self._smart_stream.subscribe_futures(str(token), exchange=self.instrument.exchange)
            logger.info("Re-subscribed MCX Crude futures token %s after tick timeout.", token)
            return

        if not self._rest_fallback:
            self._enable_rest_fallback("no futures tick within timeout window")

    def _reconnect_websocket_if_due(self) -> None:
        """Reconnect a closed SmartStream socket without interrupting REST fallback scans."""
        now = time.monotonic()
        if now < self._next_websocket_reconnect:
            return

        self._next_websocket_reconnect = now + self.settings.websocket_reconnect_seconds
        try:
            self._connect_futures_stream()
        except Exception as exc:  # pragma: no cover
            logger.debug("MCX Crude websocket reconnect attempt failed: %s", exc)
            return

        if self._smart_stream.connected:
            self._rest_fallback = False
            logger.info("MCX Crude websocket reconnected; live scanning resumed.")


if __name__ == "__main__":
    logger.info("Initializing MCX Crude Oil option-buying engine.")
    engine = CrudeOptionBuyer()
    engine.start()
