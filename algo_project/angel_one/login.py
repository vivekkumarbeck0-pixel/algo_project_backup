# ============================================================
# ANGEL ONE LOGIN & BROKER API
# ============================================================

import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from logger import get_logger

log = get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)
MCX_MARKET_OPEN = dt_time(9, 0)
MCX_MARKET_CLOSE = dt_time(23, 30)


class AngelOneLogin:

    ENV_API_KEY = "ANGEL_ONE_API_KEY"
    ENV_CLIENT_ID = "ANGEL_ONE_CLIENT_ID"
    ENV_PASSWORD = "ANGEL_ONE_PASSWORD"
    ENV_TOTP = "ANGEL_ONE_TOTP"
    ENV_FILENAME = ".env"

    @classmethod
    def _project_root(cls) -> Path:
        return Path(__file__).resolve().parents[1]

    @classmethod
    def _load_dotenv(cls) -> None:
        dotenv_path = cls._project_root() / cls.ENV_FILENAME
        if not dotenv_path.exists():
            return

        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value

    @classmethod
    def from_env(cls):
        cls._load_dotenv()

        api_key = os.getenv(cls.ENV_API_KEY)
        if not api_key:
            raise EnvironmentError(
                f"{cls.ENV_API_KEY} is not set in the environment or in {cls.ENV_FILENAME}"
            )

        return cls(api_key)

    @classmethod
    def connect_from_env(cls):
        cls._load_dotenv()

        client_id = os.getenv(cls.ENV_CLIENT_ID)
        password = os.getenv(cls.ENV_PASSWORD)
        totp = os.getenv(cls.ENV_TOTP)

        # A second generateSession() on the same account invalidates the first one's
        # tokens, so reuse a live session before creating a new one.
        with cls._session_lock():
            reused = cls._reuse_cached_session(client_id)
            if reused is not None:
                return reused

            totp = cls._current_totp(totp)

            if not client_id or not password or not totp:
                raise EnvironmentError(
                    f"{cls.ENV_CLIENT_ID}, {cls.ENV_PASSWORD}, and {cls.ENV_TOTP} must be set in the environment or {cls.ENV_FILENAME}"
                )

            client = cls.from_env()
            client.connect(client_id, password, totp)
            client._save_session_cache(client_id)
            return client

    @classmethod
    def _current_totp(cls, value):
        if value and len(value) > 6:
            try:
                import pyotp

                return pyotp.TOTP(value).now()
            except Exception:
                pass
        return value

    # ========================================================
    # SHARED SESSION CACHE (lets Nifty + Crude run side by side)
    # ========================================================

    SESSION_CACHE_FILE = "data/broker_session.json"
    SESSION_CACHE_MAX_AGE_SECONDS = 6 * 60 * 60
    SESSION_LOCK_TIMEOUT_SECONDS = 30.0

    @classmethod
    def _session_cache_path(cls) -> Path:
        return cls._project_root() / cls.SESSION_CACHE_FILE

    @staticmethod
    def _session_scope() -> str:
        railway_environment = os.getenv("RAILWAY_ENVIRONMENT_ID")
        railway_service = os.getenv("RAILWAY_SERVICE_ID")
        if railway_environment:
            return f"railway:{railway_environment}:{railway_service or 'unknown-service'}"
        return f"local:{os.getenv('COMPUTERNAME', 'unknown-machine')}"

    @classmethod
    @contextmanager
    def _session_lock(cls):
        """Stop two processes from logging in at the same instant."""
        lock_path = cls._session_cache_path().with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + cls.SESSION_LOCK_TIMEOUT_SECONDS
        handle = None

        while True:
            try:
                handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > cls.SESSION_LOCK_TIMEOUT_SECONDS:
                        lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.time() >= deadline:
                    log.warning("Broker session lock timed out; continuing without it.")
                    break
                time.sleep(0.5)

        try:
            yield
        finally:
            if handle is not None:
                try:
                    os.close(handle)
                except OSError:
                    pass
                lock_path.unlink(missing_ok=True)

    @classmethod
    def _reuse_cached_session(cls, client_id):
        """Rebuild a client from cached tokens, or None when they are unusable."""
        path = cls._session_cache_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        api_key = os.getenv(cls.ENV_API_KEY)
        if data.get("client_id") != client_id or data.get("api_key") != api_key:
            return None
        if data.get("session_scope") != cls._session_scope():
            return None
        if time.time() - float(data.get("saved_at", 0)) > cls.SESSION_CACHE_MAX_AGE_SECONDS:
            return None
        if not data.get("access_token") or not data.get("feed_token"):
            return None

        try:
            client = cls.from_env()
            smart_api = client.smart_api
            smart_api.setAccessToken(data["access_token"])
            smart_api.setRefreshToken(data.get("refresh_token"))
            smart_api.setFeedToken(data["feed_token"])
            smart_api.setUserId(data.get("user_id"))

            profile = smart_api.getProfile(data.get("refresh_token"))
            if not isinstance(profile, dict) or not profile.get("status"):
                return None
        except Exception as exc:
            log.debug("Cached broker session rejected (%s); creating a new one.", exc)
            return None

        client.access_token = data["access_token"]
        client.refresh_token = data.get("refresh_token")
        client.feed_token = data["feed_token"]
        client.user_id = data.get("user_id")

        log.info("Reusing the existing Angel One session (no second login).")
        return client

    def _save_session_cache(self, client_id) -> None:
        payload = {
            "saved_at": time.time(),
            "session_scope": self._session_scope(),
            "client_id": client_id,
            "api_key": self.api_key,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "feed_token": self.feed_token,
            "user_id": self.user_id,
        }
        path = self._session_cache_path()
        temporary = path.with_suffix(".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(path)
        except Exception as exc:
            log.warning("Could not cache the broker session: %s", exc)

    DEFAULT_CLIENT_PUBLIC_IP = "152.58.57.165"
    DEFAULT_CLIENT_LOCAL_IP = "152.58.57.165"

    def __init__(self, api_key, debug=False):
        SmartConnect = self._import_smart_connect()

        self.api_key = api_key
        self.debug = debug
        self.smart_api = SmartConnect(
            api_key=api_key,
            clientPublicIP=self.DEFAULT_CLIENT_PUBLIC_IP,
            clientLocalIP=self.DEFAULT_CLIENT_LOCAL_IP,
            privateKey=api_key,
        )
        self.smart_api.clientPublicIp = self.DEFAULT_CLIENT_PUBLIC_IP
        self.smart_api.clientLocalIp = self.DEFAULT_CLIENT_LOCAL_IP
        self.smart_api.clientPublicIP = self.DEFAULT_CLIENT_PUBLIC_IP
        self.smart_api.clientLocalIP = self.DEFAULT_CLIENT_LOCAL_IP
        self.smart_api.privateKey = api_key
        self.access_token = None
        self.refresh_token = None
        self.feed_token = None
        self.user_id = None

    @staticmethod
    def _import_smart_connect():
        try:
            from SmartApi import SmartConnect
            return SmartConnect
        except (ImportError, ModuleNotFoundError):
            pass

        try:
            import SmartAPI
            sys.modules.setdefault("SmartApi", SmartAPI)
            from SmartApi.smartConnect import SmartConnect
            return SmartConnect
        except (ImportError, ModuleNotFoundError):
            pass

        raise ModuleNotFoundError(
            "SmartApi package is not installed or importable. "
            "Install smartapi-python and ensure the SmartApi/SmartAPI package is available "
            "in the active Python environment."
        )

    # ========================================================
    # CREATE CONNECTION
    # ========================================================

    def connect(self, client_id, password, totp):

        session = self.smart_api.generateSession(
            client_id,
            password,
            totp
        )

        if not session.get("status"):
            raise Exception(
                f"Angel One Login Failed: {session}"
            )

        self.access_token = self.smart_api.access_token
        self.refresh_token = self.smart_api.refresh_token
        self.feed_token = self.smart_api.feed_token
        self.user_id = self.smart_api.userId

        print()
        print("=" * 60)
        print("ANGEL ONE CONNECTION SUCCESSFUL")
        print("=" * 60)

        return session

    # ========================================================
    # GET API OBJECT
    # ========================================================

    def get_api(self):
        return self.smart_api

    def search_scrip(self, exchange, search_text):
        return self.smart_api.searchScrip(exchange, search_text)

    def get_market_data(self, mode, exchange_tokens):
        response = self.smart_api.getMarketData(mode, exchange_tokens)
        if self._is_invalid_token_response(response):
            self._reauthenticate()
            response = self.smart_api.getMarketData(mode, exchange_tokens)
        self._raise_for_api_error(response, "market data")
        return response

    def get_candle_data(self, symboltoken, interval, from_date, to_date, exchange=None):
        exchange = exchange or "NFO"
        params = {
            "exchange": exchange,
            "symboltoken": str(symboltoken),
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date,
        }
        log.debug("getCandleData request: %s", params)
        response = self.smart_api.getCandleData(params)
        if self._is_invalid_token_response(response):
            self._reauthenticate()
            response = self.smart_api.getCandleData(params)
        self._raise_for_api_error(response, f"candle request={params}")
        return response

    def get_option_ohlc(self, symboltoken, interval="ONE_MINUTE", days=1, exchange=None):
        from_date, to_date = self.market_session_range(days, exchange=exchange)
        return self.get_candle_data(
            symboltoken,
            interval,
            from_date,
            to_date,
            exchange=exchange,
        )

    def get_ltp(self, exchange, tradingsymbol, symboltoken):
        """Fetch current LTP/OHLC via SmartAPI's official ltpData endpoint."""
        log.debug("ltpData request: exchange=%s symbol=%s token=%s", exchange, tradingsymbol, symboltoken)
        response = self.smart_api.ltpData(exchange, tradingsymbol, str(symboltoken))
        if self._is_invalid_token_response(response):
            self._reauthenticate()
            response = self.smart_api.ltpData(exchange, tradingsymbol, str(symboltoken))
        self._raise_for_api_error(
            response,
            f"ltpData exchange={exchange} symbol={tradingsymbol} token={symboltoken}",
        )
        return response

    @staticmethod
    def _response_error_code(response):
        if not isinstance(response, dict):
            return None
        return response.get("errorCode") or response.get("errorcode")

    @classmethod
    def _is_invalid_token_response(cls, response) -> bool:
        return (
            isinstance(response, dict)
            and cls._response_error_code(response) == "AG8001"
        )

    @classmethod
    def _raise_for_api_error(cls, response, context: str) -> None:
        if not isinstance(response, dict):
            return
        failed = response.get("status") is False or response.get("success") is False
        if failed:
            raise RuntimeError(
                f"Angel One API error: {response.get('message')} "
                f"(errorcode={cls._response_error_code(response)}) {context}"
            )

    def _reauthenticate(self) -> None:
        """Replace an expired JWT and update the process-local session cache."""
        self._load_dotenv()
        client_id = os.getenv(self.ENV_CLIENT_ID)
        password = os.getenv(self.ENV_PASSWORD)
        totp = self._current_totp(os.getenv(self.ENV_TOTP))
        if not client_id or not password or not totp:
            raise EnvironmentError(
                f"Cannot renew Angel One session: {self.ENV_CLIENT_ID}, "
                f"{self.ENV_PASSWORD}, and {self.ENV_TOTP} must be set"
            )

        log.warning("Angel One token was rejected; creating one fresh session and retrying.")
        with self._session_lock():
            self.connect(client_id, password, totp)
            self._save_session_cache(client_id)

    def get_option_greek(self, params):
        """SmartAPI option-greek data: IV/delta per strike for one expiry.

        Expects {"name": "NIFTY", "expirydate": "28AUG2025"}.
        """
        log.debug("optionGreek request: %s", params)
        response = self.smart_api.optionGreek(params)
        if self._is_invalid_token_response(response):
            self._reauthenticate()
            response = self.smart_api.optionGreek(params)
        self._raise_for_api_error(response, f"optionGreek request={params}")
        return response

    @staticmethod
    def market_session_range(days: int = 1, exchange: str | None = None):
        """Return (from_date, to_date) in Angel One's required 'YYYY-MM-DD HH:MM'
        format, clamped to valid completed NSE market-session times in IST so
        no future/in-progress data is ever requested.

        NOTE: rolls back over weekends only; exchange holidays are not
        accounted for.
        """
        is_mcx = str(exchange or "").upper() == "MCX"
        market_open = MCX_MARKET_OPEN if is_mcx else MARKET_OPEN
        market_close = MCX_MARKET_CLOSE if is_mcx else MARKET_CLOSE
        now_ist = datetime.now(IST)

        if now_ist.time() < market_open:
            to_dt = datetime.combine(now_ist.date() - timedelta(days=1), market_close, tzinfo=IST)
        elif now_ist.time() > market_close:
            to_dt = datetime.combine(now_ist.date(), market_close, tzinfo=IST)
        else:
            to_dt = now_ist.replace(second=0, microsecond=0)

        while to_dt.weekday() >= 5:  # Saturday=5, Sunday=6
            to_dt = datetime.combine(to_dt.date() - timedelta(days=1), market_close, tzinfo=IST)

        from_dt = datetime.combine(to_dt.date() - timedelta(days=max(days, 1) - 1), market_open, tzinfo=IST)
        while from_dt.weekday() >= 5:
            from_dt = datetime.combine(from_dt.date() - timedelta(days=1), market_open, tzinfo=IST)

        return from_dt.strftime("%Y-%m-%d %H:%M"), to_dt.strftime("%Y-%m-%d %H:%M")