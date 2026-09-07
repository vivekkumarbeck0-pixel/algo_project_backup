from datetime import time

from trading_nifty import NiftyTradingSession


def test_nifty_market_hours_start_at_9_am():
    assert NiftyTradingSession._market_hours(NiftyTradingSession.__new__(NiftyTradingSession)) == (
        time(9, 0),
        time(15, 30),
    )