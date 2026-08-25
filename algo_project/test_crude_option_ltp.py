from types import SimpleNamespace
from unittest.mock import Mock

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