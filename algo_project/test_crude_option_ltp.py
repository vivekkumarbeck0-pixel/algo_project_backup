from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from trading_crude import CrudeOptionBuyer


def _engine_for_option_ltp():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine.settings = SimpleNamespace(option_exchange="MCX")
    engine._option_ltp = None
    engine._option_ltp_token = ""
    engine._option_ltp_time = 0.0
    engine._option_ltp_max_age = 5.0
    engine._smart_stream = Mock()
    engine._rest_client = Mock()
    return engine


def test_entry_option_ltp_falls_back_to_smartapi_quote_without_websocket_tick():
    engine = _engine_for_option_ltp()
    engine._rest_client.get_ltp.return_value = {"data": {"ltp": 83.5}}

    price = engine._entry_option_ltp({"symbol": "CRUDEOIL17SEP268200CE", "token": "123"})

    assert price == 83.5
    engine._smart_stream.subscribe_futures.assert_called_once_with("123", exchange="MCX")
    engine._rest_client.get_ltp.assert_called_once_with("MCX", "CRUDEOIL17SEP268200CE", "123")


def test_entry_option_ltp_uses_matching_fresh_websocket_tick():
    engine = _engine_for_option_ltp()
    engine._option_ltp = 81.25
    engine._option_ltp_token = "123"
    engine._option_ltp_time = __import__("time").monotonic()

    price = engine._entry_option_ltp({"symbol": "CRUDEOIL17SEP268200CE", "token": "123"})

    assert price == 81.25
    engine._smart_stream.subscribe_futures.assert_not_called()
    engine._rest_client.get_ltp.assert_not_called()


def test_open_position_does_not_reuse_an_option_quote_older_than_five_seconds():
    engine = _engine_for_option_ltp()
    position = SimpleNamespace(
        option_token="123",
        option_symbol="CRUDEOIL17SEP268200CE",
        entry_price=80.0,
    )
    engine._position = position
    engine._option_ltp = 81.25
    engine._option_ltp_token = "123"
    engine._option_ltp_time = 1.0
    engine._last_option_poll = 10.0
    engine._option_poll_interval = 2.0

    with patch("trading_crude.time.monotonic", return_value=10.0):
        assert engine._current_option_price(position) is None


def test_option_price_jump_over_thirty_percent_is_rejected_without_refreshing_age():
    engine = _engine_for_option_ltp()
    engine._position = SimpleNamespace(
        option_token="123",
        option_symbol="CRUDEOIL17SEP268200CE",
        entry_price=100.0,
    )
    engine._option_ltp = 100.0
    engine._option_ltp_token = "123"
    engine._option_ltp_time = 5.0

    with patch("trading_crude.time.monotonic", return_value=10.0):
        assert engine._accept_option_price("123", 131.0) is False

    assert engine._option_ltp == 100.0
    assert engine._option_ltp_time == 5.0


def test_exchange_timestamped_option_quote_older_than_five_seconds_is_stale():
    engine = _engine_for_option_ltp()

    now = datetime(2026, 10, 9, 12, 0, 10, tzinfo=ZoneInfo("Asia/Kolkata"))
    with patch("trading_crude._now_ist", return_value=now):
        assert engine._option_quote_is_fresh(
            {"exchFeedTime": "09-Oct-2026 12:00:00"}
        ) is False