from unittest.mock import patch

from angel_one.login import AngelOneLogin
from angel_one.market_data import MarketDataFetcher


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
