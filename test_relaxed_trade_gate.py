from unittest.mock import patch

from engine.decision_engine import DecisionEngine
from engine.position_tracker import PositionTracker
from engine.risk_manager import RiskManager, RiskLimits


def test_relaxed_watchlist_keeps_bullish_trade_valid():
    tracker = PositionTracker()
    risk = RiskManager(tracker, RiskLimits(max_trades_per_day=None, max_consecutive_stop_losses=None))
    engine = DecisionEngine(risk_manager=risk)

    snapshot = {
        "spot": 24157,
        "underlying": "NIFTY",
        "support_strike": 24150,
        "resistance_strike": 24250,
        "market_strike": 24150,
        "aoc_watchlist": {"top": [24280], "bottom": [24110]},
        "aoc_color_available": True,
        "aoc_scenario": "bullish",
    }
    market_structure = {
        "support": 24150,
        "resistance": 24250,
        "trend": "BULLISH",
        "micro_momentum": "BULLISH",
        "entry_confirmed": True,
        "option_chain": {
            "by_strike": {
                "24150": {
                    "PE": {"open_interest": 100000, "oi_change": 500, "trade_volume": 1000},
                },
            },
        },
    }

    with patch.object(engine, "_is_auto_square_off_time", return_value=False):
        decision = engine.decide(snapshot, market_structure)

    assert decision.action == "BUY"
    assert decision.risk_approved is True
