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


IST = ZoneInfo("Asia/Kolkata")

_status_line_width = 0


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

        self.futures_bars: Deque[Bar] = deque(maxlen=self.settings.max_futures_history_bars)
        self._last_tick_time: Optional[datetime] = None
        self._stale_feed_since: Optional[datetime] = None
        self._stale_log_interval = 60.0
        self._last_oi: Optional[float] = None
        self._last_close: Optional[float] = None
        self._last_cum_volume: Optional[float] = None
        self._last_signal_bar: Optional[datetime] = None
        self._bar_lock = threading.Lock()
        self._trade_lock = threading.Lock()

        self._position: Optional[Position] = None
        self._live_signal_queue: queue.Queue = queue.Queue()
        self._rest_client = None
        self._rest_fallback = False
        self._last_rest_poll = 0.0
        self._rest_poll_interval = 2.0
        self._next_websocket_reconnect = 0.0
        self._historical_bars_loaded = False

        self._smart_stream = AngelSmartWebSocketClient(
            api_key=self.settings.angel_api_key,
            client_code=self.settings.angel_client_code,
            feed_token=self.settings.angel_feed_token,
            jwt_token=self.settings.angel_jwt_token,
        )
        self.nymex_filter = YFinanceLeadFilter(refresh_seconds=self.settings.yfinance_refresh_seconds)
        self._futures_token_cache: Optional[str] = None
        self._restore_state()

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
        if payload.get("date") != today:
            return

        self.paper_realized_pnl = float(payload.get("paper_realized_pnl", 0.0))
        self.paper_wins = int(payload.get("paper_wins", 0))
        self.paper_losses = int(payload.get("paper_losses", 0))
        self._trail_distance = float(payload.get("trail_distance", self._trail_distance))
        position = payload.get("position")
        if not isinstance(position, dict):
            return

        try:
            values = dict(position)
            values["entry_time"] = datetime.fromisoformat(values["entry_time"])
            self._position = Position(**values)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Unable to restore saved Crude position: %s", exc)
            return

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
            response = self._rest_client.get_candle_data(
                token,
                "ONE_MINUTE",
                from_time.strftime("%Y-%m-%d %H:%M"),
                now.strftime("%Y-%m-%d %H:%M"),
                exchange=self.instrument.exchange,
            )
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
                self._handle_tick({"data": data})
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

            # The option leg shares this stream, so route its ticks away from the futures bars.
            token = str(raw_data.get("token") or raw_data.get("symbolToken") or "").strip()
            position = self._position
            if position is not None and token and token == position.option_token:
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
            return None

        # The newest bar is still forming, so its partial volume would corrupt the spike test.
        completed_bars = list(self.futures_bars)[:-1]
        if len(completed_bars) < 22:
            return None

        recent = completed_bars[-25:]
        current_bar = recent[-1]
        previous_bar = recent[-2]

        if self._last_signal_bar == current_bar.timestamp:
            return None

        current_price = current_bar.close
        previous_close = previous_bar.close
        ois = [bar.oi for bar in recent]
        history_volumes = [max(float(bar.volume), 0.0) for bar in recent[:-1]]
        ma_volume = float(np.mean(history_volumes[-self.settings.ma_volume_period :]))
        volume_spike = (current_bar.volume > self.settings.volume_spike_factor * ma_volume) and (current_bar.volume > 0)

        futures_up = current_price > previous_close
        futures_down = current_price < previous_close
        oi_up = current_bar.oi > (ois[-2] if len(ois) >= 2 else current_bar.oi)
        oi_down = current_bar.oi < (ois[-2] if len(ois) >= 2 else current_bar.oi)

        pivots = self._calculate_daily_pivots(recent)
        atr_value = self._calculate_atr(recent)
        next_pivot = self._next_pivot_for_target(current_price, pivots)

        nymex_trend = self.nymex_filter.trend
        nymex_green = nymex_trend == "GREEN"
        nymex_red = nymex_trend == "RED"

        if not volume_spike:
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

        if side is None or scenario is None:
            return None

        confirmed, pivot_name, pivot_level = self._pivot_breakout(current_price, pivots, side)
        if not confirmed:
            logger.debug(
                "%s %s rejected: %.2f has not cleared pivot %s (%.2f) by the %.1f point buffer.",
                scenario,
                side,
                current_price,
                pivot_name or "n/a",
                pivot_level,
                self.settings.buffer_points,
            )
            return None

        self._last_signal_bar = current_bar.timestamp
        strike = self._nearest_atm_strike(current_price)
        return {
            "scenario": scenario,
            "side": side,
            "strike": strike,
            "futures_price": current_price,
            "oi": current_bar.oi,
            "volume": current_bar.volume,
            "atr": atr_value,
            "pivot": pivots,
            "pivot_level_name": pivot_name,
            "pivot_level": pivot_level,
            "buffer_points": self.settings.buffer_points,
            "next_pivot": next_pivot,
            "nymex_trend": nymex_trend,
            "timestamp": current_bar.timestamp,
        }

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

    def _next_pivot_for_target(self, price: float, pivots: Dict[str, float]) -> float:
        targets = [pivots["R1"], pivots["R2"], pivots["S1"], pivots["S2"]]
        if price >= pivots["PP"]:
            bucket = [pivots["R1"], pivots["R2"]]
        else:
            bucket = [pivots["S1"], pivots["S2"]]
        return min(bucket, key=lambda x: abs(x - price)) if bucket else pivots["PP"]

    def _pivot_breakout(self, price: float, pivots: Dict[str, float], side: str) -> tuple[bool, Optional[str], float]:
        """Require price to clear the crossed pivot by BUFFER_POINTS, not just touch it."""
        buffer_points = self.settings.buffer_points

        if side == "CE":
            crossed = {name: level for name, level in pivots.items() if level <= price}
            if not crossed:
                return False, None, 0.0
            name = max(crossed, key=lambda key: crossed[key])
            level = pivots[name]
            return price >= level + buffer_points, name, level

        crossed = {name: level for name, level in pivots.items() if level >= price}
        if not crossed:
            return False, None, 0.0
        name = min(crossed, key=lambda key: crossed[key])
        level = pivots[name]
        return price <= level - buffer_points, name, level

    def _nearest_atm_strike(self, futures_price: float) -> int:
        step = self.instrument.strike_step
        rounded = round(futures_price / step) * step
        return int(rounded)

    def _write_paper_trade_log(self, record: Dict[str, Any]):
        entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
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

    def _risk_levels(self, entry_price: float, atr: float) -> Optional[Dict[str, float]]:
        """Stop keeps the full ATR+buffer cushion; a premium too thin to hold it is skipped."""
        stop_distance = 0.75 * atr + self.settings.buffer_points
        trail_distance = 1.5 * atr + 5.0
        min_premium = stop_distance / self.settings.max_stop_premium_fraction

        if entry_price < min_premium:
            logger.info(
                "Entry skipped: premium %.2f cannot hold a %.2f point stop (needs >= %.2f).",
                entry_price,
                stop_distance,
                min_premium,
            )
            return None

        return {
            "stop_distance": stop_distance,
            "trail_distance": trail_distance,
            "stop_loss": entry_price - stop_distance,
            "target_price": entry_price + max(1.5 * atr, 2.0 * stop_distance),
        }

    def _place_entry_order(self, signal: Dict[str, Any]):
        if self._position is not None:
            return

        strike = int(signal["strike"])
        option_type = signal["side"]
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
            logger.warning(
                "Entry skipped: no broker LTP for %s %s %s.",
                self.instrument.symbol, strike, option_type,
            )
            return

        levels = self._risk_levels(entry_option_price, atr)
        if levels is None:
            return

        stop_loss = levels["stop_loss"]
        target_price = levels["target_price"]
        trail_distance = levels["trail_distance"]

        if is_real:
            order_id = self._submit_real_order("BUY", contract, quantity) or ""
            if not order_id:
                logger.error("REAL entry aborted: broker rejected the BUY order.")
                return

        self._position = Position(
            side=option_type,
            strike=strike,
            entry_price=entry_option_price,
            entry_time=datetime.now(),
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
                ("Time", self._position.entry_time.strftime("%Y-%m-%d %H:%M:%S")),
                ("Scenario", signal["scenario"]),
                ("Side", option_type),
                ("Strike", strike),
                ("Option expiry", contract.get("expiry", "-")),
                ("Futures LTP", f"{price:.2f}"),
                ("Entry premium", f"{entry_option_price:.2f}"),
                ("Premium source", "broker LTP"),
                ("Stop loss", f"{stop_loss:.2f}"),
                ("Target", f"{target_price:.2f}"),
                ("Quantity", quantity),
                ("Order id", order_id or "-"),
                ("ATR", f"{atr:.2f}"),
                ("Open interest", f"{signal.get('oi', 0):,.0f}"),
                ("Bar volume", f"{signal.get('volume', 0):,.0f}"),
                ("NYMEX trend", signal.get("nymex_trend")),
                ("Pivot cleared", f"{signal.get('pivot_level_name', '-')} {signal.get('pivot_level', 0):.2f} "
                                  f"(+/-{signal.get('buffer_points', 0):.1f} buffer)"),
                ("Next pivot", f"{signal.get('next_pivot', 0):.2f}"),
                ("Square-off at", f"{self._square_off_time().strftime('%H:%M')} IST"),
            ],
        )
        logger.info(
            "%s ENTRY %s %s @ %.2f | scenario=%s futures=%.2f stop=%.2f target=%.2f",
            mode,
            option_type,
            strike,
            entry_option_price,
            signal["scenario"],
            price,
            stop_loss,
            target_price,
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
                "qty": quantity,
                "oi": signal.get("oi"),
                "volume": signal.get("volume"),
                "nymex_trend": signal.get("nymex_trend"),
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
        if (
            self._option_ltp is not None
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
        return live

    def _fetch_option_ltp(self, contract: Dict[str, Any]) -> Optional[float]:
        if not self._ensure_rest_client():
            return None
        try:
            exchange = self.settings.option_exchange
            symbol = contract.get("symbol")
            token = str(contract.get("token"))
            if hasattr(self._rest_client, "get_ltp"):
                response = self._rest_client.get_ltp(exchange, symbol, token)
            else:
                response = self._rest_client.get_api().ltpData(exchange, symbol, token)
            data = response.get("data") if isinstance(response, dict) else None
            if isinstance(data, dict):
                ltp = float(data.get("ltp") or data.get("last_traded_price") or 0.0)
                if data.get("last_traded_price") is not None and data.get("ltp") is None:
                    ltp /= 100.0
                return ltp if ltp > 0 else None
        except Exception as exc:
            logger.warning("Could not fetch option LTP for %s: %s", contract.get("symbol"), exc)
        return None

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
        held = datetime.now() - position.entry_time
        self._print_block(
            f"{mode} EXIT {position.side}  |  {reason}",
            [
                ("Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
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
        self._position = None
        self._option_ltp = None
        self._option_ltp_token = ""
        self._option_ltp_time = 0.0
        self._persist_state()

    def _update_position_management(self):
        if self._position is None or self.current_price is None:
            return

        position = self._position
        option_price = self._current_option_price(position)

        position.trailing_stop = max(position.trailing_stop, option_price - self._trail_distance)

        if option_price >= position.target_price:
            self._close_position(option_price, "TARGET HIT")
            return

        if option_price <= position.stop_loss:
            self._close_position(option_price, "STOP LOSS")
            return

        if option_price <= position.trailing_stop:
            self._close_position(option_price, "TRAILING STOP")
            return

    def _process_live_cycle(self):
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
                state = "scanning for CE/PE setup"

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
        """Re-subscribe the websocket, or fall back to REST polling if it is down."""
        token = self._resolve_futures_token()

        if self._smart_stream.connected and token:
            self._smart_stream.subscribe_futures(str(token), exchange=self.instrument.exchange)
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
