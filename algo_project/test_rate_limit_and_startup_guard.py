import time
from unittest.mock import Mock

from engine.decision_engine import Decision
from engine.position_tracker import PositionTracker
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


def test_trade_gate_accepts_a_fresh_quote_without_price_movement():
    session = object.__new__(LivePaperTradingSession)
    session._startup_trade_locked_until = 0.0
    session._last_live_tick_price = None
    session._last_trade_ready_at = 0.0
    session._active_symbol = "NIFTY"
    session._resolve_underlying = lambda symbol: {"token": "26000"}

    class DummyMarketData:
        def _live_quote(self, token, exchange):
            return {"ltp": 24_500.0, "time": time.time()}

    session.market_data = DummyMarketData()

    assert session._has_fresh_live_confirmation() is True
    assert session._has_fresh_live_confirmation() is True


def test_nifty_trade_gate_falls_back_to_fresh_rest_ltp_when_websocket_tick_is_missing():
    session = object.__new__(LivePaperTradingSession)
    session._startup_trade_locked_until = 0.0
    session._last_live_tick_price = None
    session._last_trade_ready_at = 0.0
    session._active_symbol = "NIFTY"
    session._resolve_underlying = lambda symbol: {"token": "26000", "symbol": "NIFTY"}
    session.market_data = Mock()
    session.market_data._live_quote.return_value = None
    session.market_data.fetch_latest_price.return_value = 24_500.0

    assert session._has_fresh_live_confirmation() is True
    assert session._last_live_tick_price == 24_500.0
    session.market_data.fetch_latest_price.assert_called_once_with(
        "26000", exchange="NSE", tradingsymbol="NIFTY"
    )


def test_nifty_trade_gate_falls_back_when_websocket_tick_is_stale():
    session = object.__new__(LivePaperTradingSession)
    session._startup_trade_locked_until = 0.0
    session._last_live_tick_price = None
    session._last_trade_ready_at = 0.0
    session._active_symbol = "NIFTY"
    session._resolve_underlying = lambda symbol: {"token": "26000", "symbol": "NIFTY"}
    session.market_data = Mock()
    session.market_data._live_quote.return_value = {"ltp": 24_500.0, "time": time.time() - 40}
    session.market_data.fetch_latest_price.return_value = 24_501.0

    assert session._has_fresh_live_confirmation() is True
    assert session._last_live_tick_price == 24_501.0
    session.market_data.fetch_latest_price.assert_called_once()


def test_nifty_trade_gate_rejects_invalid_rest_ltp_fallback():
    session = object.__new__(LivePaperTradingSession)
    session._startup_trade_locked_until = 0.0
    session._last_trade_ready_at = 0.0
    session._active_symbol = "NIFTY"
    session._resolve_underlying = lambda symbol: {"token": "26000", "symbol": "NIFTY"}
    session.market_data = Mock()
    session.market_data._live_quote.return_value = None
    session.market_data.fetch_latest_price.return_value = 0.0

    assert session._has_fresh_live_confirmation() is False


def test_nifty_trade_gate_rejects_stale_cached_rest_ltp_fallback():
    session = object.__new__(LivePaperTradingSession)
    session._startup_trade_locked_until = 0.0
    session._last_trade_ready_at = 0.0
    session._active_symbol = "NIFTY"
    session._resolve_underlying = lambda symbol: {"token": "26000", "symbol": "NIFTY"}
    session.market_data = Mock()
    session.market_data._live_quote.return_value = None
    session.market_data.fetch_latest_price.return_value = 24_500.0
    session.market_data._cache = {
        ("ltp", "NSE", "26000"): (time.monotonic() - 20.0, 24_500.0)
    }

    assert session._has_fresh_live_confirmation() is False


def test_open_trade_refuses_second_position_when_one_is_open():
    session = object.__new__(LivePaperTradingSession)
    session.tracker = PositionTracker()
    session.tracker.open_position("NIFTY", 23500, "CE", "BUY", 65, 100.0)

    decision = Decision(
        action="BUY",
        option_type="PE",
        strike=23450,
        quantity=65,
        confidence="HIGH",
        risk_approved=True,
        underlying="NIFTY",
    )

    session._open_trade(decision, index_price=23480.0)

    open_positions = session.tracker.open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].option_type == "CE"
