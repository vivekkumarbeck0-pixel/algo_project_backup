from threading import Lock
from unittest.mock import Mock, patch

from angel_one.login import AngelOneLogin
from angel_one.market_data import MarketDataFetcher
from angel_one.smart_stream import LiveTickStore
from datetime import datetime
from zoneinfo import ZoneInfo


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
