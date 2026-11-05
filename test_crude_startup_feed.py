from collections import deque
from datetime import datetime, timedelta
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from trading_crude import AngelSmartWebSocketClient, Bar, CrudeOptionBuyer


IST = ZoneInfo("Asia/Kolkata")


def _engine_for_feed_tests():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine.settings = SimpleNamespace(websocket_reconnect_seconds=3.0)
    engine.instrument = SimpleNamespace(exchange="MCX")
    engine._historical_bars_loaded = False
    engine._bar_lock = threading.Lock()
    engine.futures_bars = deque(maxlen=2000)
    engine._last_close = None
    engine._rest_client = Mock()
    engine._smart_stream = Mock(connected=False)
    engine._rest_fallback = True
    engine._next_websocket_reconnect = 0.0
    return engine


def test_startup_bootstrap_loads_30_completed_crude_candles():
    engine = _engine_for_feed_tests()
    now = datetime.now(IST).replace(second=0, microsecond=0, tzinfo=None)
    engine._rest_client.get_candle_data.return_value = {
        "data": [
            [(now - timedelta(minutes=minute)).strftime("%Y-%m-%dT%H:%M:00+05:30"), 7000, 7010, 6990, 7005, 100]
            for minute in range(31, 0, -1)
        ]
    }

    engine._bootstrap_futures_bars("999")

    assert len(engine.futures_bars) == 30
    assert engine._historical_bars_loaded is True
    assert engine.futures_bars[-1].close == 7005.0
    engine._rest_client.get_candle_data.assert_called_once()


def test_startup_bootstrap_runs_before_a_failed_websocket_connection():
    engine = _engine_for_feed_tests()
    engine.settings.angel_jwt_token = "jwt"
    engine.settings.angel_feed_token = "feed"
    engine.settings.angel_api_key = "key"
    engine.settings.angel_client_code = "client"
    engine.settings.option_exchange = "MCX"
    engine._resolve_futures_token = Mock(return_value="999")
    engine._bootstrap_futures_bars = Mock()
    engine._smart_stream.connect.return_value = False
    engine._enable_rest_fallback = Mock()

    engine._connect_futures_stream()

    engine._bootstrap_futures_bars.assert_called_once_with("999")
    engine._enable_rest_fallback.assert_called_once_with("websocket connection failed")


def test_closed_websocket_reconnects_and_disables_rest_fallback():
    engine = _engine_for_feed_tests()

    def reconnect():
        engine._smart_stream.connected = True

    engine._connect_futures_stream = Mock(side_effect=reconnect)

    engine._reconnect_websocket_if_due()

    engine._connect_futures_stream.assert_called_once()
    assert engine._rest_fallback is False


def _shared_client(store):
    client = AngelSmartWebSocketClient("key", "client", "feed", "jwt")
    client._store = store
    return client


def test_engine_stream_never_opens_its_own_websocket():
    from unittest.mock import patch

    store = Mock(connected=True)
    client = AngelSmartWebSocketClient("key", "client", "feed", "jwt")
    client.client = object()

    with patch("angel_one.smart_stream.get_shared_tick_store", return_value=store) as shared:
        assert client.connect() is True
        assert client.connect() is True

    shared.assert_called_once_with(client.client)
    store.add_listener.assert_called_once_with(client._on_tick)
    assert not hasattr(client, "ws")


def test_connect_reports_unavailable_shared_stream():
    from unittest.mock import patch

    client = AngelSmartWebSocketClient("key", "client", "feed", "jwt")
    with patch("angel_one.smart_stream.get_shared_tick_store", return_value=None):
        assert client.connect() is False
    assert client.connected is False


def test_websocket_subscription_reports_success():
    store = Mock(connected=True)
    store.ensure_subscribed.return_value = True
    client = _shared_client(store)

    assert client.subscribe_futures("OPTION", exchange="MCX") is True
    store.ensure_subscribed.assert_called_once_with("OPTION", "MCX")


def test_websocket_subscription_failure_is_reported():
    store = Mock(connected=True)
    store.ensure_subscribed.return_value = False
    client = _shared_client(store)

    assert client.subscribe_futures("OPTION", exchange="MCX") is False
    assert _shared_client(Mock(connected=False)).subscribe_futures("OPTION") is False


def test_close_detaches_without_stopping_shared_stream():
    store = Mock(connected=True)
    client = _shared_client(store)

    client.close()

    store.remove_listener.assert_called_once_with(client._on_tick)
    store.stop.assert_not_called()
    store.close.assert_not_called()


def test_max_websocket_retries_refresh_session_before_reconnecting():
    engine = _engine_for_feed_tests()
    engine.settings.angel_jwt_token = "stale-jwt"
    engine.settings.angel_feed_token = "stale-feed"
    engine.settings.angel_api_key = "key"
    engine.settings.angel_client_code = "client"
    engine.settings.option_exchange = "MCX"
    engine._websocket_retry_attempts = 3
    engine._resolve_futures_token = Mock(return_value="999")
    engine._bootstrap_futures_bars = Mock()
    engine._enable_rest_fallback = Mock()
    engine._auto_login = Mock(return_value=True)
    engine._smart_stream.connect.return_value = False

    engine._connect_futures_stream()

    engine._auto_login.assert_called_once()
    engine._smart_stream.connect.assert_called_once()
    assert engine._websocket_retry_attempts == 1


def test_websocket_consumer_keeps_latest_tick_per_token_without_backlog():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    stream = _shared_client(Mock(connected=True))
    stream._on_tick({"token": "FUT", "ltp": 7000.0})
    stream._on_tick({"token": "OPT", "ltp": 100.0})
    stream._on_tick({"token": "FUT", "ltp": 7001.0})
    assert len(stream._latest_ticks) == 2
    engine._smart_stream = stream
    engine._handle_tick = Mock()

    engine._process_latest_websocket_ticks()

    handled = [call.args[0]["data"] for call in engine._handle_tick.call_args_list]
    assert handled == [
        {"token": "FUT", "ltp": 7001.0},
        {"token": "OPT", "ltp": 100.0},
    ]
    assert stream.drain_latest_ticks() == []


def test_ltp_only_fallback_reuses_last_valid_oi_and_cumulative_volume():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine._resolve_futures_token = Mock(return_value="FUT")
    engine._position = None
    engine._bar_lock = threading.Lock()
    engine.futures_bars = deque(maxlen=10)
    engine._last_tick_time = None
    engine._last_cum_volume = None
    engine._last_valid_oi = None
    engine._last_valid_cumulative_volume = None
    engine._live_signal_queue = queue.Queue()
    engine._evaluate_strategy = Mock(return_value=None)

    engine._handle_tick({
        "data": {
            "token": "FUT",
            "last_traded_price": 718_000,
            "open_interest": 2_400,
            "volume_trade_for_the_day": 50_000,
        }
    })
    engine._handle_tick({
        "data": {
            "token": "FUT",
            "ltp": 7_181.0,
            "oi": 0,
            "volume": 0,
        }
    })

    assert engine.current_oi == 2_400
    assert engine.current_volume == 50_000
    assert engine._last_valid_oi == 2_400
    assert engine._last_valid_cumulative_volume == 50_000