from typing import Any, List, Optional
import json
import queue
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

from .login import AngelOneLogin
from config import settings
from logger import get_logger

log = get_logger(__name__)

# Angel One signals throttling either explicitly ("Access denied because of
# exceeding access rate", errorcode AB1004) or implicitly by returning an
# empty body that the SDK then fails to JSON-decode.
RATE_LIMIT_MARKERS = (
    "access denied",
    "exceeding access rate",
    "ab1004",
    "too many requests",
    "429",
    "couldn't parse the json",
    "could not parse the json",
    "expecting value",
    "jsondecodeerror",
    "name resolutionerror",
    "failed to resolve",
    "connection reset",
    "connection aborted",
    "max retries exceeded",
    "temporarily unavailable",
)


def is_rate_limit_error(exc: BaseException) -> bool:
    """True for Angel One throttling / empty-body JSON parse failures."""
    text = str(exc).lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


class _CandlesUnavailable(Exception):
    """Internal signal carrying the cached candles to fall back on, if any."""

    def __init__(self, cause: BaseException, cached: Any):
        super().__init__(str(cause))
        self.cause = cause
        self.cached = cached


class MarketDataFetcher:
    """Fetch candle/market data from Angel One SmartAPI.

    If no credentials are available in the environment, the instance
    will be created with `client=None` and methods will raise a
    RuntimeError when attempting live fetches. This allows using the
    class in demo mode without secrets.
    """

    _throttle_lock = threading.Lock()
    _last_request_ts = 0.0
    _rate_limited_until_global = 0.0
    _candle_throttle_lock = threading.Lock()
    _last_candle_request_ts = 0.0
    _disk_cache_data: Optional[dict] = None
    _request_queue = queue.Queue()
    _request_worker_started = False
    _request_worker_lock = threading.Lock()

    @classmethod
    def _ensure_request_queue_worker(cls) -> None:
        with cls._request_worker_lock:
            if cls._request_worker_started:
                return

            def _worker() -> None:
                while True:
                    task = cls._request_queue.get()
                    if task is None:
                        cls._request_queue.task_done()
                        break
                    fn, args, kwargs, result, event = task
                    try:
                        gap = float(settings.api_min_request_interval_seconds)
                        wait = gap - (time.monotonic() - cls._last_request_ts)
                        if wait > 0:
                            time.sleep(wait)
                        cls._last_request_ts = time.monotonic()
                        result["value"] = fn(*args, **kwargs)
                    except Exception as exc:
                        result["error"] = exc
                    finally:
                        event.set()
                        cls._request_queue.task_done()

            worker = threading.Thread(target=_worker, daemon=True)
            worker.start()
            cls._request_worker_started = True

    @classmethod
    def _queued_call(cls, fn, *args, **kwargs):
        cls._ensure_request_queue_worker()
        result = {}
        event = threading.Event()
        cls._request_queue.put((fn, args, kwargs, result, event))
        event.wait()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def __init__(self, client: Optional[Any] = None, use_env: bool = True):
        self.client = client
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._rate_limited_until = 0.0
        self._option_chain_previous: dict[str, float] = {}
        self._tick_store = None
        if client is None and use_env:
            # Prefer connect_from_env (loads client_id/password/totp and creates session)
            try:
                self.client = AngelOneLogin.connect_from_env()
            except Exception:
                # fall back to from_env (may provide unauthenticated api object)
                try:
                    self.client = AngelOneLogin.from_env()
                except Exception:
                    self.client = None

    # ------------------------------------------------------------
    # LIVE WEBSOCKET FEED (removes most REST quote traffic)
    # ------------------------------------------------------------

    @property
    def ticks(self):
        """Lazily started SmartStream tick store, or None when unavailable."""
        if not settings.use_websocket_feed or not self.client:
            return None
        if self._tick_store is None:
            from .smart_stream import LiveTickStore

            store = LiveTickStore(self.client)
            store.start()
            self._tick_store = store
        return self._tick_store if self._tick_store.connected else None

    def _live_quote(self, token: Optional[str], exchange: Optional[str]):
        """Return a fresh websocket tick for `token`, subscribing on first use."""
        store = self.ticks
        if store is None or not token:
            return None
        store.ensure_subscribed(str(token), exchange or "NSE")
        return store.get(str(token), settings.ws_tick_max_age_seconds)

    # A partially covered chain would mix live and stale strikes, so REST wins unless
    # nearly every leg has a fresh tick.
    LIVE_CHAIN_MIN_COVERAGE = 0.9

    def _live_option_quotes(self, tokens: list, exchange: str) -> Optional[dict]:
        """FULL-quote-shaped rows built from websocket ticks, or None to use REST."""
        store = self.ticks
        if store is None or not tokens:
            return None

        quotes = {}
        for token in tokens:
            store.ensure_subscribed(str(token), exchange)
            record = store.get(str(token), settings.ws_tick_max_age_seconds)
            if record is not None:
                quotes[str(token)] = {
                    "symbolToken": str(token),
                    "ltp": record["ltp"],
                    "opnInterest": record["open_interest"],
                    "tradeVolume": record["trade_volume"],
                }

        if len(quotes) < self.LIVE_CHAIN_MIN_COVERAGE * len(tokens):
            return None
        return quotes

    def _date_range_strings(self, days: int, exchange: Optional[str] = None):
        return AngelOneLogin.market_session_range(days, exchange=exchange)

    # ------------------------------------------------------------
    # RESPONSE CACHE + RATE-LIMIT GUARD
    # ------------------------------------------------------------

    @classmethod
    def _throttle(cls) -> None:
        """Space out broker calls process-wide so the API quota is never burst."""
        with cls._throttle_lock:
            gap = float(settings.api_min_request_interval_seconds)
            wait = gap - (time.monotonic() - cls._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            cls._last_request_ts = time.monotonic()

    @classmethod
    def _throttle_candles(cls) -> None:
        """The historical endpoint has its own, much tighter quota."""
        with cls._candle_throttle_lock:
            gap = float(settings.candle_min_request_interval_seconds)
            wait = gap - (time.monotonic() - cls._last_candle_request_ts)
            if wait > 0:
                time.sleep(wait)
            cls._last_candle_request_ts = time.monotonic()
        cls._throttle()

    # ------------------------------------------------------------
    # CANDLE DISK CACHE (survives restarts and long rate-limit cooldowns)
    # ------------------------------------------------------------

    @classmethod
    def _disk_cache(cls) -> dict:
        if cls._disk_cache_data is None:
            path = Path(settings.candle_disk_cache_file)
            try:
                cls._disk_cache_data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                cls._disk_cache_data = {}
        return cls._disk_cache_data

    @staticmethod
    def _disk_cache_key(key: tuple) -> str:
        return "|".join(str(part) for part in key)

    # Intraday candles go stale fast; serving yesterday's 1-minute bars as if they
    # were live would feed the pattern engine a false trigger.
    CANDLE_DISK_MAX_AGE = {
        "ONE_MINUTE": 300.0,
        "THREE_MINUTE": 600.0,
        "FIVE_MINUTE": 900.0,
        "TEN_MINUTE": 1800.0,
        "FIFTEEN_MINUTE": 1800.0,
        "THIRTY_MINUTE": 3600.0,
        "ONE_HOUR": 7200.0,
    }

    @classmethod
    def _disk_cache_max_age(cls, key: tuple) -> float:
        interval = str(key[3]).upper() if len(key) > 3 else ""
        return cls.CANDLE_DISK_MAX_AGE.get(
            interval, float(settings.candle_disk_cache_max_age_seconds)
        )

    def _disk_cache_get(self, key: tuple):
        entry = self._disk_cache().get(self._disk_cache_key(key))
        if not isinstance(entry, dict):
            return None
        age = time.time() - float(entry.get("saved_at", 0))
        if age > self._disk_cache_max_age(key):
            return None
        return entry.get("data")

    def _disk_cache_set(self, key: tuple, value: Any) -> None:
        if not value:
            return
        cache = self._disk_cache()
        cache[self._disk_cache_key(key)] = {"saved_at": time.time(), "data": value}
        path = Path(settings.candle_disk_cache_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache), encoding="utf-8")
        except Exception as exc:
            log.debug("Could not persist candle cache: %s", exc)

    def _request_candles_with_retry(self, cache_key, symboltoken, interval, from_date, to_date, exchange):
        """Retry a throttled candle request only while there is no cached fallback."""
        attempts = max(1, int(settings.candle_fetch_retries))
        for attempt in range(1, attempts + 1):
            try:
                # Candle history has a separate broker quota from quotes. Apply
                # the process-wide historical throttle before queueing the call.
                type(self)._throttle_candles()
                return self._queued_call(
                    self.client.get_candle_data,
                    symboltoken,
                    interval,
                    from_date,
                    to_date,
                    exchange=exchange,
                )
            except Exception as exc:
                cached = self._cache_get(cache_key)
                if cached is None:
                    cached = self._disk_cache_get(cache_key)

                if cached is not None:
                    self._note_api_failure(exc, f"candles token={symboltoken} interval={interval}")
                    log.info(
                        "Serving last-known candles for token=%s interval=%s after fetch failure.",
                        symboltoken, interval,
                    )
                    raise _CandlesUnavailable(exc, cached) from exc

                if attempt < attempts and is_rate_limit_error(exc):
                    delay = min(30.0, float(settings.candle_retry_backoff_seconds) * (2 ** (attempt - 1)))
                    log.warning(
                        "Candle fetch throttled (token=%s interval=%s); waiting %.0fs and retrying (%d/%d).",
                        symboltoken, interval, delay, attempt, attempts,
                    )
                    time.sleep(3.0)
                    time.sleep(delay)
                    continue

                self._note_api_failure(exc, f"candles token={symboltoken} interval={interval}")
                raise _CandlesUnavailable(exc, None) from exc

    def _request_full_quotes_with_retry(self, exchange: str, tokens: list, context: str) -> dict:
        """FULL quotes for option tokens, retried on transient broker failures."""
        attempts = max(1, int(settings.option_chain_retries))
        for attempt in range(1, attempts + 1):
            try:
                response = self._queued_call(self.client.get_market_data, "FULL", {exchange: tokens})
                data = response.get("data", {}) if isinstance(response, dict) else {}
                fetched = data.get("fetched", []) if isinstance(data, dict) else []
                quotes = {str(row.get("symbolToken")): row for row in fetched if isinstance(row, dict)}
                if quotes:
                    return quotes
                raise RuntimeError("empty FULL quote response")
            except Exception as exc:
                if attempt < attempts:
                    if is_rate_limit_error(exc):
                        delay = min(30.0, 3.0 * (2 ** (attempt - 1)))
                        time.sleep(3.0)
                    else:
                        delay = float(settings.option_chain_retry_backoff_seconds) * attempt
                    log.warning(
                        "Option-chain quotes failed for %s (%s); retrying in %.0fs (%d/%d).",
                        context, str(exc)[:90], delay, attempt, attempts,
                    )
                    time.sleep(delay)
                    continue
                self._note_api_failure(exc, f"option chain {context}")
        return {}

    def _stale_option_chain(self, underlying: str, empty: dict, reason: str) -> dict:
        """Reuse the last good chain so OI-based targets survive a transient outage."""
        cache_key = ("option_chain", str(underlying).upper())
        cached = self._cache_get(cache_key, settings.option_chain_max_stale_seconds)
        if cached is None:
            empty["error"] = f"Option chain unavailable ({reason})"
            log.warning("Option chain unavailable for %s (%s) and nothing cached.", underlying, reason)
            return empty

        log.info("Serving last-known option chain for %s after %s.", underlying, reason)
        stale = dict(cached)
        stale["stale"] = True
        stale["error"] = f"stale: {reason}"
        return stale

    def _cache_get(self, key: tuple, ttl: Optional[float] = None):
        """Return the cached value, or None when missing/older than `ttl`.

        `ttl=None` returns the value at any age (last-known fallback).
        """
        entry = self._cache.get(key)
        if entry is None:
            return None
        cached_at, value = entry
        if ttl is not None and (time.monotonic() - cached_at) > ttl:
            return None
        return value

    def _cache_set(self, key: tuple, value: Any) -> None:
        if value is not None:
            self._cache[key] = (time.monotonic(), value)

    def _cooling_down(self) -> bool:
        return time.monotonic() < max(
            self._rate_limited_until,
            type(self)._rate_limited_until_global,
        )

    def _cooldown_remaining(self) -> float:
        until = max(self._rate_limited_until, type(self)._rate_limited_until_global)
        return max(until - time.monotonic(), 0.0)

    def _note_api_failure(self, exc: BaseException, context: str) -> None:
        if is_rate_limit_error(exc):
            cooldown_until = time.monotonic() + float(settings.api_rate_limit_cooldown_seconds)
            self._rate_limited_until = cooldown_until
            type(self)._rate_limited_until_global = max(
                type(self)._rate_limited_until_global, cooldown_until
            )
            log.warning(
                "Angel One rate limit hit (%s): %s - serving cached values for %.0fs",
                context, exc, float(settings.api_rate_limit_cooldown_seconds),
            )
        else:
            log.warning("Angel One request failed (%s): %s", context, exc)

    def is_connected(self) -> bool:
        """Return whether the broker session can answer a live quote request."""
        if not self.client:
            return False
        try:
            return bool(getattr(self.client, "access_token", True))
        except Exception:
            return False

    def fetch_option_iv(
        self,
        name: str,
        expirydate: str,
        strike: Optional[float] = None,
        option_type: Optional[str] = None,
    ) -> Optional[float]:
        """Implied volatility from the broker's option-greek API, never OCR.

        Angel One expects {"name": "NIFTY", "expirydate": "28AUG2025"} and
        answers with one greek row per strike/optionType, so a single call
        covers both CE and PE. Rows are cached for the metrics refresh
        interval; a throttled/failed call reuses the last-known rows.
        """
        if not self.client or not name or not expirydate:
            return None

        cache_key = ("greeks", str(name).upper(), str(expirydate).upper())
        rows = self._cache_get(cache_key, settings.metrics_refresh_interval_seconds)

        if rows is None and not self._cooling_down():
            rows = self._request_option_greeks(name, expirydate)
            self._cache_set(cache_key, rows)

        if rows is None:
            rows = self._cache_get(cache_key)

        return self._extract_iv(rows, strike, option_type) if rows else None

    def fetch_option_chain(self, underlying: str, center: float, radius: int = 10, expiry: str | None = None) -> dict:
        """Fetch broker FULL quotes for nearest CE/PE strikes around ``center``.

        OI change is computed from the previous successful snapshot in this
        process. The first snapshot therefore reports ``None`` for OI change.
        """
        empty = {"underlying": underlying, "expiry": expiry, "center": center, "rows": [], "by_strike": {}, "error": None}
        try:
            from angel_one.instrument_reader import InstrumentReader
            from config import SYMBOL_REGISTRY

            cfg = SYMBOL_REGISTRY[str(underlying).upper()]
            reader = InstrumentReader()
            reader.load()
            items = [
                item for item in reader.instruments
                if str(item.get("name", "")).upper() == str(underlying).upper()
                and str(item.get("exch_seg", "")).upper() == cfg["exchange"]
                and str(item.get("instrumenttype", "")).upper() == cfg["option_instrumenttype"]
                and str(item.get("symbol", "")).upper().endswith(("CE", "PE"))
            ]
            if not items:
                empty["error"] = "No option contracts found"
                return empty
            if expiry is None:
                from datetime import datetime

                today = datetime.now().date()
                valid_expiries = []
                for item in items:
                    expiry_text = str(item.get("expiry") or "").upper()
                    try:
                        expiry_date = datetime.strptime(expiry_text, "%d%b%Y").date()
                    except ValueError:
                        continue
                    if expiry_date >= today:
                        valid_expiries.append((expiry_date, expiry_text))
                valid_expiries.sort()
                expiry = valid_expiries[0][1] if valid_expiries else None
            items = [item for item in items if not expiry or str(item.get("expiry")) == expiry]
            strikes = sorted({float(item.get("strike", 0)) / 100.0 for item in items})
            nearby = set(sorted(strikes, key=lambda value: abs(value - float(center)))[: max(1, radius * 2 + 1)])
            items = [item for item in items if float(item.get("strike", 0)) / 100.0 in nearby]
            tokens = [str(item.get("token")) for item in items if item.get("token")]
            if not self.client or not tokens:
                empty["error"] = "Broker client or option tokens unavailable"
                return empty
            quotes = self._live_option_quotes(tokens, cfg["exchange"])
            if quotes is None:
                quotes = self._request_full_quotes_with_retry(cfg["exchange"], tokens, underlying)
            if not quotes:
                return self._stale_option_chain(underlying, empty, "no quotes returned")
            rows = []
            grouped = {}
            for item in items:
                strike = float(item.get("strike", 0)) / 100.0
                right = "PE" if str(item.get("symbol", "")).upper().endswith("PE") else "CE"
                token = str(item.get("token"))
                quote = quotes.get(token, {})
                oi = quote.get("opnInterest", quote.get("openInterest"))
                key = f"{underlying}:{expiry}:{strike}:{right}"
                previous = self._option_chain_previous.get(key)
                oi_change = float(oi) - previous if oi is not None and previous is not None else None
                if oi is not None:
                    self._option_chain_previous[key] = float(oi)
                row = {"strike": strike, "right": right, "symbol": item.get("symbol"), "token": token, "ltp": quote.get("ltp"), "open_interest": oi, "oi_change": oi_change, "trade_volume": quote.get("tradeVolume"), "depth": quote.get("depth"), "raw_quote": quote}
                rows.append(row)
                grouped.setdefault(str(strike), {})[right] = row
            empty.update({"expiry": expiry, "rows": rows, "by_strike": grouped})
            self._cache_set(("option_chain", str(underlying).upper()), empty)
            return empty
        except Exception as exc:
            log.warning("Option-chain fetch failed for %s: %s", underlying, exc)
            return self._stale_option_chain(underlying, empty, f"{type(exc).__name__}: {exc}")

    def _request_option_greeks(self, name: str, expirydate: str) -> Optional[list]:
        payload = {"name": str(name).upper(), "expirydate": str(expirydate).upper()}
        targets = [self.client, getattr(self.client, "smart_api", None)]
        for target in targets:
            for method_name in ("get_option_greek", "optionGreek", "getOptionGreek"):
                method = getattr(target, method_name, None) if target is not None else None
                if not callable(method):
                    continue
                try:
                    response = self._queued_call(method, payload)
                except Exception as exc:
                    self._note_api_failure(exc, f"option greeks {payload}")
                    continue

                data = response.get("data") if isinstance(response, dict) else response
                if isinstance(data, dict):
                    data = [data]
                if data:
                    return data
                log.warning("Option greek API returned no data for %s", payload)
        return None

    @staticmethod
    def _extract_iv(rows: list, strike: Optional[float], option_type: Optional[str]) -> Optional[float]:
        values = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if option_type and str(row.get("optionType", "")).upper() != option_type.upper():
                continue
            if strike is not None:
                try:
                    if abs(float(row.get("strikePrice")) - float(strike)) > 0.01:
                        continue
                except (TypeError, ValueError):
                    continue
            for key in ("impliedVolatility", "iv", "implied_volatility"):
                if row.get(key) is not None:
                    try:
                        values.append(float(row[key]))
                    except (TypeError, ValueError):
                        pass
                    break

        values = [value for value in values if value > 0]
        return sum(values) / len(values) if values else None

    def fetch_candles(
        self,
        symboltoken: str,
        interval: str = "ONE_MINUTE",
        days: int = 1,
        exchange: Optional[str] = None,
        underlying_name: Optional[str] = None,
        _allow_future_fallback: bool = True,
    ) -> Any:
        """Fetch candle data for a token using the Angel One client.

        Responses are cached for `settings.api_cache_ttl_seconds` and reused
        (at any age) while the broker is rate-limiting us, so a throttled
        tick degrades to the last-known candles instead of wiping levels.

        Angel One's historical candle API does not return intraday data
        for plain index tokens (e.g. NSE:NIFTY token 26000) even though
        the request succeeds. If `underlying_name` is given and the
        index request comes back empty, this automatically retries once
        against the nearest NFO futures contract for that underlying as
        an index-proxy (logged clearly, never silent).

        Raises RuntimeError if no client is configured, or if the API call
        fails and nothing was ever cached for this request.
        """
        if not self.client:
            raise RuntimeError("No Angel One API client available. Set credentials in .env or pass a client.")

        from_date, to_date = self._date_range_strings(days, exchange=exchange)
        cache_key = (
            "candles",
            str(exchange),
            str(symboltoken),
            str(interval).upper(),
            str(from_date),
            str(to_date),
        )
        fresh = self._cache_get(cache_key, settings.api_cache_ttl_seconds)
        if fresh is not None:
            return fresh

        if self._cooling_down():
            stale = self._cache_get(cache_key)
            if stale is None:
                stale = self._disk_cache_get(cache_key)
            if stale is not None:
                log.debug("Rate-limit cooldown active; serving cached candles for token=%s", symboltoken)
                return stale

            # Nothing was ever cached for this request, so waiting out the cooldown is
            # the only way this token ever gets candles and can trade.
            remaining = self._cooldown_remaining()
            if remaining > float(settings.candle_cooldown_max_wait_seconds):
                raise RuntimeError("Angel One rate limit cooldown active and no cached candles available")
            log.warning(
                "No candles cached for token=%s interval=%s; waiting %.0fs for the rate-limit cooldown.",
                symboltoken, interval, remaining,
            )
            time.sleep(remaining)

        log.info(
            "Fetching candles: exchange=%s token=%s interval=%s from=%s to=%s",
            exchange, symboltoken, interval, from_date, to_date,
        )

        try:
            response = self._request_candles_with_retry(
                cache_key, symboltoken, interval, from_date, to_date, exchange
            )
        except _CandlesUnavailable as exc:
            if exc.cached is not None:
                return exc.cached
            if _allow_future_fallback and underlying_name and str(exchange or "").upper() == "NSE":
                from .instrument_reader import InstrumentReader

                future = InstrumentReader().find_nearest_future(underlying_name)
                if future:
                    log.warning(
                        "Rate-limited NSE candles for %s (token=%s); falling back to nearest "
                        "futures contract %s (token=%s, NFO) as an index proxy",
                        underlying_name, symboltoken, future.get("symbol"), future.get("token"),
                    )
                    proxy = self.fetch_candles(
                        str(future.get("token")),
                        interval=interval,
                        days=days,
                        exchange="NFO",
                        underlying_name=underlying_name,
                        _allow_future_fallback=False,
                    )
                    self._cache_set(cache_key, proxy)
                    self._disk_cache_set(cache_key, proxy)
                    return proxy
            raise RuntimeError(f"Candle fetch failed: {exc.cause}") from exc.cause

        data = response.get("data") if isinstance(response, dict) else response

        if not data and _allow_future_fallback and underlying_name and str(exchange or "").upper() == "NSE":
            from .instrument_reader import InstrumentReader

            future = InstrumentReader().find_nearest_future(underlying_name)
            if future:
                log.warning(
                    "No historical candles for index %s (token=%s); falling back to nearest "
                    "futures contract %s (token=%s, NFO) as an index proxy",
                    underlying_name, symboltoken, future.get("symbol"), future.get("token"),
                )
                proxy = self.fetch_candles(
                    str(future.get("token")),
                    interval=interval,
                    days=days,
                    exchange="NFO",
                    underlying_name=underlying_name,
                    _allow_future_fallback=False,
                )
                self._cache_set(cache_key, proxy)
                self._disk_cache_set(cache_key, proxy)
                return proxy

        if data:
            self._cache_set(cache_key, response)
            self._disk_cache_set(cache_key, response)
        return response


    def fetch_latest_price(
        self,
        symboltoken: str,
        exchange: Optional[str] = None,
        tradingsymbol: Optional[str] = None,
        verbose: bool = False,
    ) -> Optional[float]:
        """Get the latest price for a token via the official SmartAPI `ltpData`
        endpoint (requires the exact tradingsymbol from the instrument master),
        falling back to the last close of recent 1-minute candles and finally
        to the last-known cached price (never returns 0.0 on a rate limit).
        """
        if not self.client:
            raise RuntimeError("No Angel One API client available. Set credentials in .env or pass a client.")

        cache_key = ("ltp", str(exchange), str(symboltoken))
        fresh = self._cache_get(cache_key, settings.ltp_cache_ttl_seconds)
        if fresh is not None:
            return fresh

        live = self._live_quote(symboltoken, exchange)
        if live is not None:
            self._cache_set(cache_key, live["ltp"])
            return live["ltp"]

        if self._cooling_down():
            return self._cache_get(cache_key)

        if tradingsymbol:
            try:
                resp = self._queued_call(self.client.get_ltp, exchange or "NSE", tradingsymbol, symboltoken)
                if verbose:
                    log.debug("ltpData response: %s", resp)
                data = resp.get("data") if isinstance(resp, dict) else None
                if data and data.get("ltp") is not None:
                    price = float(data["ltp"])
                    self._cache_set(cache_key, price)
                    return price
            except Exception as exc:
                self._note_api_failure(exc, f"ltpData {tradingsymbol} token={symboltoken}")

                # During a transient broker/network outage, keep the last
                # known quote and avoid immediately making another candle call.
                stale = self._cache_get(cache_key)
                if stale is not None:
                    return stale

        try:
            candles = self.fetch_candles(symboltoken, interval="ONE_MINUTE", days=1, exchange=exchange)
            price = self.extract_last_close(candles)
            if price is not None:
                self._cache_set(cache_key, price)
                return price
        except Exception as exc:
            log.warning("Candle-based price fallback failed for token=%s: %s", symboltoken, exc)

        return self._cache_get(cache_key)

    @staticmethod
    def extract_last_close(candles: Any) -> Optional[float]:
        """Try to extract the latest close price from various candle formats.

        Supports common list-of-lists formats and dicts with 'close' keys.
        Returns None if no close price can be determined.
        """
        if candles is None:
            return None

        items: List = []

        if isinstance(candles, dict):
            for key in ("candles", "data", "values", "result"):
                if key in candles and candles[key]:
                    items = candles[key]
                    break
            else:
                # try to use first value
                try:
                    first = next(iter(candles.values()))
                    if isinstance(first, list):
                        items = first
                except Exception:
                    items = []
        elif isinstance(candles, list):
            items = candles

        if not items:
            return None

        last = items[-1]

        # list/tuple formats: [ts, open, high, low, close, ...]
        if isinstance(last, (list, tuple)):
            if len(last) >= 5:
                try:
                    return float(last[4])
                except Exception:
                    return None
            # fallback: last element
            try:
                return float(last[-1])
            except Exception:
                return None

        # dict format
        if isinstance(last, dict):
            for k in ("close", "Close", "c", "close_price"):
                if k in last:
                    try:
                        return float(last[k])
                    except Exception:
                        return None

        try:
            return float(last)
        except Exception:
            return None

    # ============================================================
    # CANDLE NORMALIZATION
    # ============================================================

    @staticmethod
    def normalize_candles(candles: Any) -> list[dict]:
        """Convert raw Angel One candle payloads into uniform candle dicts."""
        if candles is None:
            return []

        items = []
        if isinstance(candles, dict):
            for key in ("candles", "data", "values", "result"):
                if key in candles and candles[key]:
                    items = candles[key]
                    break
            else:
                try:
                    first = next(iter(candles.values()))
                    if isinstance(first, list):
                        items = first
                except Exception:
                    items = []
        elif isinstance(candles, list):
            items = candles

        normalized = []
        for item in items:
            if isinstance(item, (list, tuple)):
                if len(item) < 5:
                    continue
                ts = item[0]
                open_price = item[1]
                high_price = item[2]
                low_price = item[3]
                close_price = item[4]
                volume = item[5] if len(item) > 5 else None
            elif isinstance(item, dict):
                ts = item.get("timestamp") or item.get("time") or item.get("dt")
                open_price = item.get("open") or item.get("Open") or item.get("o")
                high_price = item.get("high") or item.get("High") or item.get("h")
                low_price = item.get("low") or item.get("Low") or item.get("l")
                close_price = item.get("close") or item.get("Close") or item.get("c")
                volume = item.get("volume") or item.get("Volume")
            else:
                continue

            try:
                open_price = float(open_price)
            except Exception:
                open_price = None
            try:
                high_price = float(high_price)
            except Exception:
                high_price = None
            try:
                low_price = float(low_price)
            except Exception:
                low_price = None
            try:
                close_price = float(close_price)
            except Exception:
                close_price = None
            try:
                volume = float(volume) if volume is not None else None
            except Exception:
                volume = None

            normalized.append({
                "timestamp": ts,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            })

        return normalized

    # ============================================================
    # SUPPORT / RESISTANCE FROM CANDLES
    # ============================================================

    @staticmethod
    def calculate_trend(candles: list[dict], ma_period: int = 20) -> Optional[str]:
        """Calculate trend based on Simple Moving Average crossover.
        
        Returns "BULLISH" if current close > SMA, "BEARISH" if current close < SMA,
        None if insufficient data.
        """
        if not candles or len(candles) < ma_period:
            return None
        
        try:
            closes = [c.get("close") for c in candles[-ma_period:] if c.get("close") is not None]
            if len(closes) < ma_period:
                return None
            
            sma = statistics.mean(closes)
            current_close = candles[-1].get("close")
            
            if current_close is None:
                return None
            
            if current_close > sma:
                return "BULLISH"
            elif current_close < sma:
                return "BEARISH"
            else:
                return None
        except Exception as e:
            log.debug("Error calculating trend: %s", e)
            return None

    @staticmethod
    def calculate_iv_estimate(candles: list[dict], lookback: int = 20) -> Optional[float]:
        """Estimate Option IV using Bollinger Band width as a volatility proxy.
        
        BB Width = (Upper Band - Lower Band) / SMA
        Returns a percentage (0-100 scale) representing estimated IV.
        """
        if not candles or len(candles) < lookback:
            return None
        
        try:
            recent = candles[-lookback:]
            closes = [c.get("close") for c in recent if c.get("close") is not None]
            highs = [c.get("high") for c in recent if c.get("high") is not None]
            lows = [c.get("low") for c in recent if c.get("low") is not None]
            
            if len(closes) < lookback or not highs or not lows:
                return None
            
            # Calculate standard deviation of closes
            mean_close = statistics.mean(closes)
            variance = sum((x - mean_close) ** 2 for x in closes) / len(closes)
            std_dev = variance ** 0.5
            
            # Estimate IV as (std_dev / mean) * 100 * 2 (scaling factor)
            if mean_close == 0:
                return None
            
            iv_estimate = (std_dev / mean_close) * 100 * 2
            # Cap IV estimate between 0 and 100
            return min(max(iv_estimate, 0.0), 100.0)
        except Exception as e:
            log.debug("Error calculating IV estimate: %s", e)
            return None

    @staticmethod
    def compute_support_resistance(
        candles: list[dict],
        current_price: Optional[float] = None,
        lookback: int = 60,
    ) -> dict:
        candles = [
            candle for candle in candles
            if candle.get("open") is not None
            and candle.get("high") is not None
            and candle.get("low") is not None
            and candle.get("close") is not None
        ][-lookback:]

        if not candles:
            return {
                "support": None,
                "resistance": None,
                "local_supports": [],
                "local_resistances": [],
            }

        if current_price is None:
            current_price = candles[-1]["close"]

        local_supports = []
        local_resistances = []

        for index in range(1, len(candles) - 1):
            prev_candle = candles[index - 1]
            current_candle = candles[index]
            next_candle = candles[index + 1]

            if (
                current_candle["low"] < prev_candle["low"]
                and current_candle["low"] < next_candle["low"]
            ):
                local_supports.append(current_candle["low"])

            if (
                current_candle["high"] > prev_candle["high"]
                and current_candle["high"] > next_candle["high"]
            ):
                local_resistances.append(current_candle["high"])

        support = None
        resistance = None

        if local_supports:
            below = [level for level in local_supports if level <= current_price]
            support = max(below) if below else max(local_supports)

        if local_resistances:
            above = [level for level in local_resistances if level >= current_price]
            resistance = min(above) if above else min(local_resistances)

        if support is None:
            lows = [c["low"] for c in candles if c["low"] is not None and c["low"] <= current_price]
            support = max(lows) if lows else min([c["low"] for c in candles if c["low"] is not None], default=None)

        if resistance is None:
            highs = [c["high"] for c in candles if c["high"] is not None and c["high"] >= current_price]
            resistance = min(highs) if highs else max([c["high"] for c in candles if c["high"] is not None], default=None)

        return {
            "support": support,
            "resistance": resistance,
            "local_supports": local_supports,
            "local_resistances": local_resistances,
        }

    @staticmethod
    def calculate_classic_pivots(candles: list[dict], current_price: Optional[float] = None) -> dict:
        """Calculate exact classic pivot levels from the latest completed session."""
        valid = [
            candle for candle in candles
            if candle.get("high") is not None
            and candle.get("low") is not None
            and candle.get("close") is not None
        ]
        if not valid:
            return {"pp": None, "s1": None, "r1": None, "s2": None, "r2": None}

        # Use the previous session when timestamps are available; fall back to
        # the supplied candle window for brokers that omit timestamps.
        sessions = {}
        for candle in valid:
            timestamp = candle.get("timestamp")
            session_key = str(timestamp)[:10] if timestamp else "current"
            sessions.setdefault(session_key, []).append(candle)
        session_keys = [key for key in sessions if key != "current"]
        source = sessions[sorted(session_keys)[-2]] if len(session_keys) >= 2 else valid

        high = max(float(candle["high"]) for candle in source)
        low = min(float(candle["low"]) for candle in source)
        close = float(source[-1]["close"])
        pp = (high + low + close) / 3.0
        return {
            "pp": pp,
            "s1": 2.0 * pp - high,
            "r1": 2.0 * pp - low,
            "s2": pp - (high - low),
            "r2": pp + (high - low),
            "source_high": high,
            "source_low": low,
            "source_close": close,
        }

    @staticmethod
    def select_pivot_support_resistance(pivots: dict, current_price: Optional[float]) -> tuple[Optional[float], Optional[float]]:
        """Return nearest exact pivot below and above current price."""
        levels = [pivots.get(key) for key in ("s2", "s1", "pp", "r1", "r2")]
        levels = sorted({float(level) for level in levels if level is not None})
        if not levels or current_price is None:
            return None, None
        price = float(current_price)
        below = [level for level in levels if level <= price]
        above = [level for level in levels if level >= price]
        return (max(below) if below else levels[0], min(above) if above else levels[-1])

    # ============================================================
    # ORDER BLOCK DETECTION
    # ============================================================

    @staticmethod
    def detect_order_blocks(candles: list[dict], lookback: int = 40) -> dict:
        candles = [
            candle for candle in candles
            if candle.get("open") is not None
            and candle.get("high") is not None
            and candle.get("low") is not None
            and candle.get("close") is not None
        ][-lookback:]

        bullish_blocks = []
        bearish_blocks = []

        for index in range(len(candles) - 2):
            base = candles[index]
            follow_one = candles[index + 1]
            follow_two = candles[index + 2]

            if base["close"] < base["open"]:
                if (
                    follow_one["close"] > base["high"]
                    and follow_two["close"] > follow_one["close"]
                ):
                    bullish_blocks.append({
                        "zone": {
                            "low": base["low"],
                            "high": base["high"],
                        },
                        "anchor": base,
                        "index": index,
                    })

            if base["close"] > base["open"]:
                if (
                    follow_one["close"] < base["low"]
                    and follow_two["close"] < follow_one["close"]
                ):
                    bearish_blocks.append({
                        "zone": {
                            "low": base["low"],
                            "high": base["high"],
                        },
                        "anchor": base,
                        "index": index,
                    })

        return {
            "bullish": bullish_blocks[-2:],
            "bearish": bearish_blocks[-2:],
        }

    # ============================================================
    # MARKET STRUCTURE HELPER
    # ============================================================

    def fetch_market_structure_levels(
        self,
        symboltoken: str,
        interval: str = "ONE_MINUTE",
        days: int = 1,
        exchange: Optional[str] = None,
        lookback: int = 60,
        underlying_name: Optional[str] = None,
    ) -> dict:
        try:
            candles = self.fetch_candles(
                symboltoken, interval=interval, days=days, exchange=exchange, underlying_name=underlying_name
            )
        except Exception as exc:
            log.warning("Market structure candles unavailable for %s: %s", symboltoken, exc)
            return {
                "current_price": None,
                "support": None,
                "resistance": None,
                "local_supports": [],
                "local_resistances": [],
                "order_blocks": {"bullish": [], "bearish": []},
                "candles": [],
                "trend": None,
                "iv_estimate": None,
                "broker_connected": False,
                "rate_limited": is_rate_limit_error(exc) or self._cooling_down(),
            }
        candle_dicts = self.normalize_candles(candles)
        if not candle_dicts:
            return {
                "current_price": None,
                "support": None,
                "resistance": None,
                "local_supports": [],
                "local_resistances": [],
                "order_blocks": {"bullish": [], "bearish": []},
                "candles": [],
                "trend": None,
                "iv_estimate": None,
                "broker_connected": False,
                "rate_limited": self._cooling_down(),
            }

        current_price = self.extract_last_close(candles)
        levels = self.compute_support_resistance(candle_dicts, current_price=current_price, lookback=lookback)
        pivots = self.calculate_classic_pivots(candle_dicts, current_price=current_price)
        pivot_support, pivot_resistance = self.select_pivot_support_resistance(pivots, current_price)
        order_blocks = self.detect_order_blocks(candle_dicts, lookback=lookback)
        
        # Trend is derived from broker candles. IV is not estimated from candles.
        trend = self.calculate_trend(candle_dicts, ma_period=min(20, len(candle_dicts)))

        return {
            "current_price": current_price,
            "support": levels["support"],
            "resistance": levels["resistance"],
            "pivot_levels": pivots,
            "pivot_support": pivot_support,
            "pivot_resistance": pivot_resistance,
            "local_supports": levels["local_supports"],
            "local_resistances": levels["local_resistances"],
            "order_blocks": order_blocks,
            "candles": candle_dicts,
            "trend": trend,
            "iv_estimate": None,
            "broker_connected": self.is_connected(),
            "rate_limited": False,
        }
