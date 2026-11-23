import logging
import time
from threading import Lock
from unittest.mock import Mock, patch
from datetime import datetime

from angel_one.login import AngelOneLogin
from angel_one.market_data import MarketDataFetcher
from angel_one.smart_stream import AngelSmartWebSocketClient, LiveTickStore
from zoneinfo import ZoneInfo

from logger import _SuppressMissingGreekData, _redact_sensitive_text


class FakeCandleClient:
    def __init__(self):
        self.calls = []

    def get_candle_data(self, symboltoken, interval, from_date, to_date, exchange=None):
        self.calls.append((symboltoken, interval, from_date, to_date, exchange))
        return {"status": True, "data": [[from_date, 100, 101, 99, 100, 10]]}


def test_candle_cache_reuses_same_window_and_refetches_new_window():
    client = FakeCandleClient()
    fetcher = MarketDataFetcher(client=client, use_env=False)
    date_window = ["2026-09-07 09:00", "2026-09-07 15:30"]
    fetcher._date_range_strings = lambda days, exchange=None: tuple(date_window)

    with (
        patch.object(MarketDataFetcher, "_throttle_candles"),
        patch.object(fetcher, "_disk_cache_get", return_value=None),
        patch.object(fetcher, "_disk_cache_set"),
    ):
        first = fetcher.fetch_candles("68487", interval="FIVE_MINUTE", days=1, exchange="NSE")
        second = fetcher.fetch_candles("68487", interval="FIVE_MINUTE", days=1, exchange="NSE")

        assert first == second
        assert len(client.calls) == 1

        date_window[1] = "2026-09-08 15:30"
        fetcher.fetch_candles("68487", interval="FIVE_MINUTE", days=1, exchange="NSE")

    assert len(client.calls) == 2


class ExpiringSmartApi:
    def __init__(self):
        self.ltp_calls = 0
        self.candle_calls = 0

    def ltpData(self, exchange, tradingsymbol, symboltoken):
        self.ltp_calls += 1
        if self.ltp_calls == 1:
            return {"success": False, "message": "Invalid Token", "errorCode": "AG8001"}
        return {"status": True, "data": {"ltp": 24500.0}}

    def getCandleData(self, params):
        self.candle_calls += 1
        if self.candle_calls == 1:
            return {"success": False, "message": "Invalid Token", "errorCode": "AG8001"}
        return {"status": True, "data": [[params["fromdate"], 100, 101, 99, 100, 10]]}


def test_invalid_token_reauthenticates_and_retries_ltp_and_candles_once():
    client = object.__new__(AngelOneLogin)
    client.smart_api = ExpiringSmartApi()
    reauthentication_calls = []
    client._reauthenticate = lambda: reauthentication_calls.append(True)

    ltp = client.get_ltp("NSE", "NIFTY", "99926000")
    candles = client.get_candle_data(
        "99926000", "ONE_MINUTE", "2026-09-15 09:15", "2026-09-15 09:30", exchange="NSE"
    )

    assert ltp["data"]["ltp"] == 24500.0
    assert candles["data"]
    assert len(reauthentication_calls) == 2


def test_sensitive_api_headers_are_redacted_from_log_text():
    message = (
        "Headers: {'X-PrivateKey': 'key-value', 'Authorization': 'Bearer token-value', "
        "'X-ClientLocalIP': '192.0.2.1', 'X-MACAddress': 'aa:bb:cc:dd:ee:ff'}"
    )

    redacted = _redact_sensitive_text(message)

    assert "[REDACTED]" in redacted
    assert "key-value" not in redacted
    assert "token-value" not in redacted
    assert "192.0.2.1" not in redacted
    assert "aa:bb:cc:dd:ee:ff" not in redacted


def test_sdk_verbose_option_greek_ab9019_log_is_suppressed():
    record = logging.LogRecord(
        "logzero_default",
        logging.ERROR,
        __file__,
        1,
        "Error on optionGreek request: errorcode=AB9019",
        (),
        None,
    )

    assert not _SuppressMissingGreekData().filter(record)
    assert any(
        isinstance(log_filter, _SuppressMissingGreekData)
        for log_filter in logging.getLogger("logzero_default").filters
    )


def test_option_greek_ab9019_is_cached_without_duplicate_requests():
    class GreekClient:
        def __init__(self):
            self.calls = 0

        def get_option_greek(self, params):
            self.calls += 1
            raise RuntimeError("Angel One API error: No Data Available (errorcode=AB9019)")

    client = GreekClient()
    fetcher = MarketDataFetcher(client=client, use_env=False)
    fetcher._resolve_listed_option_expiry = lambda name, expiry: "27OCT2026"

    assert fetcher.fetch_option_iv("NIFTY", "27OCT2026") is None
    fetcher._cache[("greeks", "NIFTY", "27OCT2026")] = (time.monotonic() - 20, [])
    assert fetcher.fetch_option_iv("NIFTY", "27OCT2026") is None
    assert client.calls == 1


def test_option_greek_expiry_is_resolved_to_master_format_before_request():
    class GreekClient:
        def __init__(self):
            self.params = None

        def get_option_greek(self, params):
            self.params = params
            return {"status": True, "data": []}

    client = GreekClient()
    fetcher = MarketDataFetcher(client=client, use_env=False)
    fetcher._resolve_listed_option_expiry = lambda name, expiry: "27OCT2026"

    fetcher.fetch_option_iv("NIFTY", "2026-10-27")

    assert client.params == {"name": "NIFTY", "expirydate": "27OCT2026"}


def test_option_greek_expiry_must_match_a_live_instrument_master_row(tmp_path):
    master_file = tmp_path / "instruments.json"
    master_file.write_text("[]", encoding="utf-8")

    class FakeInstrumentReader:
        MASTER_FILE = master_file

        def __init__(self):
            self.instruments = []

        def load(self):
            self.instruments = [
                {
                    "name": "NIFTY",
                    "exch_seg": "NFO",
                    "instrumenttype": "OPTIDX",
                    "symbol": "NIFTY27OCT2026CE",
                    "expiry": "27OCT2026",
                },
                {
                    "name": "NIFTY",
                    "exch_seg": "NFO",
                    "instrumenttype": "OPTIDX",
                    "symbol": "NIFTY27OCT2026PE",
                    "expiry": "27OCT2026",
                },
            ]

    class GreekClient:
        def __init__(self):
            self.calls = 0

        def get_option_greek(self, params):
            self.calls += 1
            return {"status": True, "data": []}

    client = GreekClient()
    fetcher = MarketDataFetcher(client=client, use_env=False)
    with patch("angel_one.instrument_reader.InstrumentReader", FakeInstrumentReader):
        assert fetcher._resolve_listed_option_expiry("NIFTY", "2026-10-27") == "27OCT2026"
        assert fetcher._resolve_listed_option_expiry("NIFTY", "2026-10-28") is None
        assert fetcher.fetch_option_iv("NIFTY", "2026-10-28") is None
    assert client.calls == 0


def test_live_stream_filters_nse_tokens_after_equity_close():
    store = LiveTickStore.__new__(LiveTickStore)
    store._lock = Lock()
    store._subscribed = {("26000", "NSE"), ("999", "MCX")}
    store._ticks = {"26000": {"ltp": 25000}, "999": {"ltp": 7000}}
    store._stream = Mock(connected=True)
    after_close = datetime(2026, 10, 1, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata"))

    store._filter_after_equity_close(after_close)

    assert store._subscribed == {("999", "MCX")}
    assert "26000" not in store._ticks
    store._stream.unsubscribe.assert_called_once_with("26000", "NSE")


def test_live_stream_keeps_subscriptions_before_equity_close():
    before_close = datetime(2026, 10, 1, 15, 29, tzinfo=ZoneInfo("Asia/Kolkata"))

    assert not LiveTickStore._is_after_equity_close(before_close)


def test_failed_nifty_spot_subscription_can_be_retried():
    store = LiveTickStore.__new__(LiveTickStore)
    store._lock = Lock()
    store._subscribed = set()
    store._ticks = {}
    store._stream = Mock(connected=True)
    store._stream.subscribe.side_effect = [False, True]
    store._reconnect_wakeup = Mock()
    store._stop_event = Mock()
    store._stop_event.is_set.return_value = False

    store.ensure_subscribed("26000", "NSE")
    assert ("26000", "NSE") not in store._subscribed

    store.ensure_subscribed("26000", "NSE")
    assert ("26000", "NSE") in store._subscribed
    assert [call.args for call in store._stream.subscribe.call_args_list] == [
        ("26000", "NSE"),
        ("26000", "NSE"),
    ]


def test_failed_nifty_spot_resubscription_is_retryable():
    store = LiveTickStore.__new__(LiveTickStore)
    store._lock = Lock()
    store._subscribed = {("26000", "NSE")}
    store._stream = Mock(connected=True)
    store._stream.subscribe.return_value = False
    store._stop_event = Mock()
    store._stop_event.is_set.return_value = False
    store._reconnect_wakeup = Mock()

    store._resubscribe_all()

    assert ("26000", "NSE") not in store._subscribed
    store._stream.subscribe.assert_called_once_with("26000", "NSE")
    store._reconnect_wakeup.set.assert_called_once()


def test_stream_recovery_closes_stale_socket_before_opening_replacement():
    from types import SimpleNamespace
    import angel_one.smart_stream as smart_stream

    events = []

    class FakeStream:
        def __init__(self, *args, **kwargs):
            self.connected = False

        def connect(self, **kwargs):
            events.append("connect")
            self.connected = True
            return True

        def close(self):
            events.append("close")
            self.connected = False

    store = LiveTickStore.__new__(LiveTickStore)
    store._recovery_lock = Lock()
    store._stop_event = Mock()
    store._stop_event.is_set.return_value = False
    store._client = SimpleNamespace()
    store._stream = FakeStream()
    store._correlation_id = "test"
    store._lock = Lock()
    store._subscribed = set()
    store._ticks = {}
    store._reconnect_wakeup = Mock()
    store._last_tick_at = time.monotonic()
    store._next_connect_attempt = 0.0
    refreshed = SimpleNamespace(
        api_key="key",
        user_id="client",
        feed_token="feed",
        access_token="jwt",
    )

    with (
        patch.object(AngelOneLogin, "connect_from_env", return_value=refreshed),
        patch.object(smart_stream, "AngelSmartWebSocketClient", FakeStream),
    ):
        assert store._recover_stream()

    assert events == ["close", "connect"]
    assert store._stream.connected


def test_shared_stream_retries_and_refresh_respect_cooldown_even_on_wakeup():
    store = LiveTickStore(Mock())
    clock = [0.0]
    attempts = []
    refreshes = []
    stopped = [False]

    class StopEvent:
        def is_set(self):
            return stopped[0]

    class Wakeup:
        def __init__(self):
            self.early_wakeup = True

        def wait(self, delay):
            if self.early_wakeup:
                self.early_wakeup = False
                clock[0] += 10.0
            else:
                clock[0] += delay

        def clear(self):
            pass

    def connect(**kwargs):
        attempts.append(clock[0])
        if len(attempts) == 4:
            stopped[0] = True
        return False

    store._stop_event = StopEvent()
    store._reconnect_wakeup = Wakeup()
    store._stream = Mock(connected=False)
    store._stream.connect.side_effect = connect
    store._next_connect_attempt = 90.0
    store._refresh_session_if_needed = Mock(side_effect=lambda: refreshes.append(clock[0]))

    with patch("angel_one.smart_stream.time.monotonic", side_effect=lambda: clock[0]):
        store._reconnect_loop_inner()

    assert attempts == [90.0, 180.0, 270.0, 360.0]
    assert refreshes == [360.0]


def test_watchdog_recovery_waits_between_failed_attempts():
    from types import SimpleNamespace
    import angel_one.smart_stream as smart_stream

    clock = [0.0]
    logins = []
    connects = []
    refreshed = SimpleNamespace(api_key="key", user_id="client", feed_token="feed", access_token="jwt")

    class StopEvent:
        def is_set(self):
            return False

        def wait(self, delay):
            clock[0] += delay
            return False

    class FakeStream:
        def __init__(self, *args, **kwargs):
            self.connected = False

        def connect(self, **kwargs):
            connects.append(clock[0])
            self.connected = len(connects) == 3
            return self.connected

        def close(self):
            self.connected = False

    store = LiveTickStore(SimpleNamespace())
    store._stop_event = StopEvent()
    store._stream = FakeStream()
    store._resubscribe_all = Mock()

    def login():
        logins.append(clock[0])
        return refreshed

    with (
        patch.object(AngelOneLogin, "connect_from_env", side_effect=login),
        patch.object(smart_stream, "AngelSmartWebSocketClient", FakeStream),
        patch("angel_one.smart_stream.time.monotonic", side_effect=lambda: clock[0]),
    ):
        assert store._recover_stream()

    assert logins == [0.0, 90.0, 180.0]
    assert connects == [0.0, 90.0, 180.0]


def test_normal_websocket_close_is_not_logged_as_transport_error(caplog):
    client = AngelSmartWebSocketClient("key", "client", "feed", "jwt")

    with caplog.at_level(logging.INFO, logger="angel_one.smart_stream"):
        client._on_close(object())

    assert "socket closed" in caplog.text
    assert "RAW WEBSOCKET ERROR" not in caplog.text
