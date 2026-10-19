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


def test_websocket_does_not_start_parallel_connect_while_previous_thread_runs():
    client = AngelSmartWebSocketClient.__new__(AngelSmartWebSocketClient)
    client._connect_lock = threading.Lock()
    client._connect_thread = Mock(is_alive=Mock(return_value=True))
    client.connected = False

    assert client.connect() is False


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


def test_raw_websocket_callback_errors_are_logged():
    from unittest.mock import patch

    import trading_crude

    client = AngelSmartWebSocketClient.__new__(AngelSmartWebSocketClient)
    client.connected = True
    client._open_event = threading.Event()
    client._lock = threading.Lock()
    client._latest_ticks = {}

    with patch.object(trading_crude.logger, "error") as log_error:
        client._on_error(None, "socket failure")
        client._on_close(None, 1006, "connection reset")

    assert [call.args[0] for call in log_error.call_args_list] == [
        "RAW WEBSOCKET ERROR: socket failure",
        "RAW WEBSOCKET ERROR: connection reset",
    ]


def test_websocket_consumer_keeps_latest_tick_per_token_without_backlog():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    stream = AngelSmartWebSocketClient.__new__(AngelSmartWebSocketClient)
    stream._lock = threading.Lock()
    stream._latest_ticks = {}
    stream._on_message(None, {"data": {"token": "FUT", "ltp": 7000.0}})
    stream._on_message(None, {"data": {"token": "OPT", "ltp": 100.0}})
    stream._on_message(None, {"data": {"token": "FUT", "ltp": 7001.0}})
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