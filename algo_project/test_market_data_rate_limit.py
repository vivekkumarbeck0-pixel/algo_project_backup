from unittest.mock import patch

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

    with patch.object(MarketDataFetcher, "_throttle_candles"):
        first = fetcher.fetch_candles("68487", interval="FIVE_MINUTE", days=1, exchange="NSE")
        second = fetcher.fetch_candles("68487", interval="FIVE_MINUTE", days=1, exchange="NSE")

        assert first == second
        assert len(client.calls) == 1

        date_window[1] = "2026-09-08 15:30"
        fetcher.fetch_candles("68487", interval="FIVE_MINUTE", days=1, exchange="NSE")

    assert len(client.calls) == 2
