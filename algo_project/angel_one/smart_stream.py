"""Angel One SmartStream (SmartWebSocketV2) live tick feed.

Ticks are pushed by the broker instead of polled, so every quote served from
here is one REST call the process does not have to spend against the Angel One
rate limit.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from datetime import datetime, time as datetime_time
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from logger import get_logger

log = get_logger(__name__)


class AngelSmartWebSocketClient:
    """Thin wrapper around Angel One SmartWebSocketV2."""

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

    def __init__(
        self,
        api_key: str,
        client_code: str,
        feed_token: str,
        jwt_token: str = "",
        correlation_id: str = "algostream",
    ):
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self.jwt_token = jwt_token
        self.correlation_id = correlation_id
        self.ws = None
        self.connected = False
        self.queue: queue.Queue = queue.Queue()
        self._open_event = threading.Event()
        self._connect_thread: Optional[threading.Thread] = None
        self._connect_lock = threading.Lock()
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

            sys.modules.setdefault("smartapi", SmartAPI)
            sys.modules.setdefault("SmartApi", SmartAPI)
            from SmartAPI.smartWebSocketV2 import SmartWebSocketV2  # type: ignore

            return SmartWebSocketV2
        except Exception:
            return None

    def connect(self, timeout: float = 10.0, quiet: bool = False) -> bool:
        with self._connect_lock:
            return self._connect_locked(timeout, quiet)

    def _connect_locked(self, timeout: float, quiet: bool) -> bool:
        if self._connect_thread is not None and self._connect_thread.is_alive():
            if self.connected:
                return True
            if not quiet:
                log.debug("Angel One websocket connection attempt is already running.")
            return False

        if self._SmartWebSocketV2 is None:
            self._SmartWebSocketV2 = self._import_smart_websocket_v2()

        if self._SmartWebSocketV2 is None:
            log.error(
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
        except Exception as exc:
            log.warning("Failed to initialize Angel One websocket client: %s", exc)
            self.connected = False
            return False

        # SmartWebSocketV2.connect() runs a blocking run_forever() loop, so it
        # must be driven from a background thread; we only wait here for the
        # on_open callback (or timeout) before returning control to the caller.
        self._open_event.clear()
        self._connect_thread = threading.Thread(target=self._run_forever, daemon=True)
        self._connect_thread.start()

        if not self._open_event.wait(timeout):
            if not quiet:
                log.warning("Angel One websocket did not open within %.0fs.", timeout)
            return False

        return True

    def _run_forever(self):
        try:
            self.ws.connect()
        except Exception as exc:
            log.debug("Angel One websocket connection loop terminated: %s", exc)
            self.connected = False

    def _on_open(self, *args, **kwargs):
        self.connected = True
        self._open_event.set()
        log.info("Angel One SmartStream socket opened.")

    def _on_message(self, *args):
        # SmartWebSocketV2 invokes on_data/on_message with (wsapp, message);
        # message is always the final positional argument.
        message = args[-1] if args else None
        if isinstance(message, str):
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                return
        else:
            payload = message

        if isinstance(payload, dict):
            self.queue.put(payload)

    def _on_error(self, *args):
        error = args[-1] if args else "unknown"
        log.error(f"RAW WEBSOCKET ERROR: {error}")
        self.connected = False
        self._open_event.clear()

    def _on_close(self, *args, **kwargs):
        error = args[-1] if args else "unknown"
        log.error(f"RAW WEBSOCKET ERROR: {error}")
        self.connected = False
        self._open_event.clear()

    def subscribe(self, token: str, exchange: str) -> bool:
        if self.ws is None or not self.connected:
            return False

        if LiveTickStore._is_after_equity_close() and str(exchange).upper() != "MCX":
            log.info("Skipping non-MCX websocket subscription for %s after 15:30 IST.", token)
            return False

        exchange_type = self.EXCHANGE_TYPE_MAP.get(str(exchange).upper())
        if exchange_type is None:
            log.warning("Unknown exchange '%s' for websocket subscription.", exchange)
            return False

        try:
            token_list = [{"exchangeType": exchange_type, "tokens": [str(token)]}]
            self.ws.subscribe(self.correlation_id, self.SNAP_QUOTE_MODE, token_list)
            return True
        except Exception as exc:
            if self.connected:
                log.debug("Subscription deferred for token %s on %s: %s", token, exchange, exc)
            return False

    def unsubscribe(self, token: str, exchange: str) -> bool:
        if self.ws is None or not self.connected:
            return False

        exchange_type = self.EXCHANGE_TYPE_MAP.get(str(exchange).upper())
        if exchange_type is None:
            return False

        try:
            token_list = [{"exchangeType": exchange_type, "tokens": [str(token)]}]
            self.ws.unsubscribe(self.correlation_id, self.SNAP_QUOTE_MODE, token_list)
            return True
        except Exception as exc:
            log.warning("Unable to unsubscribe token %s on %s: %s", token, exchange, exc)
            return False

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close_connection()
            except Exception:
                pass
        self.connected = False
        self._open_event.clear()


class LiveTickStore:
    """Keeps the newest websocket tick per token so quotes skip the REST API."""

    def __init__(self, client: Any, correlation_id: str = "algostream"):
        self._client = client
        self._correlation_id = correlation_id
        self._stream: Optional[AngelSmartWebSocketClient] = None
        self._ticks: Dict[str, Dict[str, Any]] = {}
        self._subscribed: set[Tuple[str, str]] = set()
        self._lock = threading.Lock()
        self._consumer: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_attempted = False
        self._reconnect_wakeup = threading.Event()
        self._last_session_refresh = 0.0

    @property
    def connected(self) -> bool:
        return bool(self._stream and self._stream.connected)

    def start(self) -> bool:
        """Connect once; a failure degrades to REST rather than raising."""
        if self._start_attempted:
            return self.connected
        self._start_attempted = True

        api_key = getattr(self._client, "api_key", "")
        client_code = getattr(self._client, "user_id", "")
        feed_token = getattr(self._client, "feed_token", "")
        jwt_token = getattr(self._client, "access_token", "")

        if not all([api_key, client_code, feed_token, jwt_token]):
            log.info("SmartStream not started: broker session tokens unavailable; staying on REST.")
            return False

        self._stream = AngelSmartWebSocketClient(
            api_key=api_key,
            client_code=client_code,
            feed_token=feed_token,
            jwt_token=jwt_token,
            correlation_id=self._correlation_id,
        )
        connected = self._stream.connect()

        self._consumer = threading.Thread(target=self._consume_loop, daemon=True)
        self._consumer.start()
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop,
            name="angel-smartstream-reconnect",
            daemon=True,
        )
        self._reconnect_thread.start()
        if connected:
            log.info("SmartStream live tick feed active; REST quote polling is now a fallback.")
        else:
            log.warning("SmartStream is offline; background reconnect has started.")
        return connected

    def _reconnect_loop(self):
        retry_delay = 1.0
        failures = 0
        while not self._stop_event.is_set():
            if self.connected:
                self._filter_after_equity_close()
                retry_delay = 1.0
                failures = 0
                self._reconnect_wakeup.wait(2.0)
                self._reconnect_wakeup.clear()
                continue

            self._reconnect_wakeup.wait(retry_delay)
            self._reconnect_wakeup.clear()
            if self._stop_event.is_set() or self.connected:
                continue

            if failures >= 3:
                self._refresh_session_if_needed()

            if self._stream.connect(timeout=10.0, quiet=True):
                failures = 0
                retry_delay = 1.0
                self._resubscribe_all()
                log.info("SmartStream reconnected; active tokens resubscribed.")
                continue

            failures += 1
            log.warning(
                "SmartStream reconnect attempt failed; retrying in %.0fs.",
                retry_delay,
            )
            retry_delay = min(retry_delay * 2.0, 30.0)

    def _refresh_session_if_needed(self):
        now = time.monotonic()
        if now - self._last_session_refresh < 300.0:
            return
        self._last_session_refresh = now
        try:
            from .login import AngelOneLogin

            refreshed = AngelOneLogin.connect_from_env()
            for name in ("api_key", "user_id", "feed_token", "access_token"):
                value = getattr(refreshed, name, None)
                if value:
                    setattr(self._client, name, value)
            self._stream.api_key = getattr(refreshed, "api_key", self._stream.api_key)
            self._stream.client_code = getattr(refreshed, "user_id", self._stream.client_code)
            self._stream.feed_token = getattr(refreshed, "feed_token", self._stream.feed_token)
            self._stream.jwt_token = getattr(refreshed, "access_token", self._stream.jwt_token)
            log.info("Angel One session refreshed before websocket retry.")
        except Exception as exc:
            log.debug("Angel One session refresh unavailable: %s", exc)

    def _resubscribe_all(self):
        self._filter_after_equity_close()
        with self._lock:
            tokens = tuple(self._subscribed)
        for token, exchange in tokens:
            if self._stop_event.is_set() or not self.connected:
                return
            self._stream.subscribe(token, exchange)

    @staticmethod
    def _is_after_equity_close(now: Optional[datetime] = None) -> bool:
        current = now or datetime.now(ZoneInfo("Asia/Kolkata"))
        if current.tzinfo is None:
            current = current.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        else:
            current = current.astimezone(ZoneInfo("Asia/Kolkata"))
        return current.time() >= datetime_time(15, 30)

    def _filter_after_equity_close(self, now: Optional[datetime] = None):
        if not self._is_after_equity_close(now):
            return

        with self._lock:
            removed = tuple(item for item in self._subscribed if item[1] != "MCX")
            self._subscribed.difference_update(removed)
            removed_tokens = {token for token, _ in removed}
            retained_tokens = {token for token, _ in self._subscribed}
            for token in removed_tokens - retained_tokens:
                self._ticks.pop(token, None)

        for token, exchange in removed:
            self._stream.unsubscribe(token, exchange)
        if removed:
            log.info("Filtered %d non-MCX websocket subscriptions after 15:30 IST.", len(removed))

    def _consume_loop(self):
        while not self._stop_event.is_set():
            try:
                payload = self._stream.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception:
                continue
            self._store(payload)

    def _store(self, payload: Dict[str, Any]) -> None:
        token = payload.get("token") or payload.get("symbolToken")
        if not token:
            return

        raw_ltp = payload.get("last_traded_price")
        if raw_ltp is None:
            return

        # SmartWebSocketV2 reports prices as paisa-scaled integers.
        ltp = float(raw_ltp) / 100.0
        if ltp <= 0:
            return

        record = {
            "ltp": ltp,
            "open_interest": float(payload.get("open_interest") or 0.0),
            "trade_volume": float(payload.get("volume_trade_for_the_day") or 0.0),
            "received_at": time.monotonic(),
        }
        with self._lock:
            self._ticks[str(token).strip()] = record

    def ensure_subscribed(self, token: str, exchange: str) -> None:
        if not self.connected or not token:
            return

        key = (str(token).strip(), str(exchange).upper())
        if self._is_after_equity_close() and key[1] != "MCX":
            self._filter_after_equity_close()
            return
        with self._lock:
            if key in self._subscribed:
                return
            self._subscribed.add(key)

        if not self._stream.subscribe(key[0], key[1]):
            self._reconnect_wakeup.set()

    def get(self, token: str, max_age: float) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._ticks.get(str(token).strip())
        if record is None:
            return None
        if (time.monotonic() - record["received_at"]) > max_age:
            return None
        return record

    def get_price(self, token: str, max_age: float) -> Optional[float]:
        record = self.get(token, max_age)
        return record["ltp"] if record else None

    def stop(self):
        self._stop_event.set()
        self._reconnect_wakeup.set()
        if self._stream is not None:
            self._stream.close()
