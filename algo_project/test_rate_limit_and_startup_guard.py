import time

from engine.trading_session import LivePaperTradingSession


def test_session_reset_clears_in_memory_caches():
    session = object.__new__(LivePaperTradingSession)
    session._underlying_cache = {"NIFTY": {"token": "123"}}
    session._market_structure_cache = {"NIFTY": {"trend": "BULLISH"}}
    session._pivot_cache = {"NIFTY": {"pivots": {"s1": 1.0}}}
    session._last_metrics = {"india_vix": 12.0, "option_iv": 15.0, "crude_volatility": 10.0}
    session._last_metrics_refresh = 99.0
    session._last_structural_refresh = 88.0
    session._last_option_chain_refresh = 77.0

    session._clear_startup_state()

    assert session._underlying_cache == {}
    assert session._market_structure_cache == {}
    assert session._pivot_cache == {}
    assert session._last_metrics == {"india_vix": None, "option_iv": None, "crude_volatility": None}
    assert session._last_metrics_refresh == 0.0
    assert session._last_structural_refresh == 0.0
    assert session._last_option_chain_refresh == 0.0


def test_trade_gate_requires_fresh_live_breakout():
    session = object.__new__(LivePaperTradingSession)
    session._startup_trade_locked_until = 0.0
    session._last_trade_ready_at = 0.0

    class DummyMarketData:
        def _live_quote(self, token, exchange):
            return {"ltp": 100.0, "time": time.time() - 40}

    session.market_data = DummyMarketData()

    assert session._has_fresh_live_confirmation() is False
