from types import SimpleNamespace
from unittest.mock import Mock, patch

from engine.pivot_ladder import build_connected_ladder
from engine.decision_engine import DecisionEngine
from engine.price_engine import PriceEngine


def test_build_connected_ladder_maps_each_strike_to_nearest_pivots():
    ladder = build_connected_ladder(
        strikes=[24000, 24050, 24100, 24150],
        pivots={"s2": 23950, "s1": 24000, "pp": 24050, "r1": 24100, "r2": 24150},
    )

    assert ladder == {
        24000.0: {"support": 23950.0, "resistance": 24050.0},
        24050.0: {"support": 24000.0, "resistance": 24100.0},
        24100.0: {"support": 24050.0, "resistance": 24150.0},
        24150.0: {"support": 24100.0, "resistance": None},
    }


def test_build_connected_ladder_does_not_reuse_equal_pivot_as_side():
    ladder = build_connected_ladder(
        strikes=[50, 100, 350],
        pivots={"s1": 100, "pp": 200, "r1": 300},
    )

    assert ladder == {
        50.0: {"support": None, "resistance": 100.0},
        100.0: {"support": None, "resistance": 200.0},
        350.0: {"support": 300.0, "resistance": None},
    }


def test_build_connected_ladder_stops_at_farther_oi_or_oi_change_peak():
    strikes = [23900, 23950, 24000, 24050, 24100, 24150, 24200]
    option_chain = {
        "by_strike": {
            "23900": {"PE": {"open_interest": 100, "oi_change": 80}},
            "23950": {"PE": {"open_interest": 200, "oi_change": 20}},
            "24050": {"CE": {"open_interest": 100, "oi_change": 20}},
            "24100": {"CE": {"open_interest": 300, "oi_change": 40}},
            "24150": {"CE": {"open_interest": 200, "oi_change": 100}},
            "24200": {"CE": {"open_interest": 50, "oi_change": 10}},
        }
    }

    ladder = build_connected_ladder(
        strikes,
        {"s2": 23900, "s1": 24000, "pp": 24050, "r1": 24100, "r2": 24200},
        option_chain=option_chain,
        spot=24000,
    )

    assert set(ladder) == {23900.0, 23950.0, 24000.0, 24050.0, 24100.0, 24150.0}
    assert ladder[24150.0] == {"support": 24100.0, "resistance": None}
    assert ladder[23900.0] == {"support": None, "resistance": None}
    assert ladder[23950.0] == {"support": 23900.0, "resistance": None}


def test_classic_pivot_target_cannot_cross_oi_cutoff():
    market_structure = {
        "unified_chain": {"target_up": 24200.0},
        "pivot_levels": {"r1": 24250.0, "r2": 24300.0},
    }
    pivot_ladder = {"24050": {}, "24100": {}}

    assert DecisionEngine._pivot_target(
        market_structure, 24000.0, "CE", pivot_ladder
    ) is None


def test_strike_sr_uses_candle_swings_instead_of_strike_grid():
    candles = [
        {"high": 23370, "low": 23340},
        {"high": 23375, "low": 23334},
        {"high": 23381, "low": 23339},
        {"high": 23376, "low": 23341},
        {"high": 23430, "low": 23390},
        {"high": 23425, "low": 23385},
        {"high": 23482, "low": 23415},
        {"high": 23440, "low": 23410},
        {"high": 23445, "low": 23420},
    ]
    option_chain = {
        "by_strike": {
            "23350": {
                "CE": {"open_interest": 120, "oi_change": 20},
                "PE": {"open_interest": 180, "oi_change": 40},
            },
            "23400": {
                "CE": {"open_interest": 300, "oi_change": 50},
                "PE": {"open_interest": 180, "oi_change": 20},
            },
            "23450": {
                "CE": {"open_interest": 200, "oi_change": 100},
                "PE": {"open_interest": 220, "oi_change": 40},
            },
        }
    }

    ladder = build_connected_ladder(
        [23350, 23400, 23450], {}, option_chain=option_chain, spot=23400, candles=candles
    )

    assert ladder[23350.0]["support"] == 23334.0
    assert ladder[23350.0]["resistance"] == 23381.0
    assert ladder[23400.0]["support"] == 23381.0
    assert ladder[23400.0]["resistance"] == 23430.0
    assert ladder[23450.0]["support"] == 23430.0
    assert ladder[23450.0]["resistance"] == 23482.0
    assert all(
        47.0 <= level["resistance"] - level["support"] <= 53.0
        for level in ladder.values()
    )


def test_candle_sr_skips_nearby_levels_when_no_47_to_53_pair_exists():
    candles = [
        {"high": 23410, "low": 23405},
        {"high": 23423.9, "low": 23395},
        {"high": 23415, "low": 23400},
        {"high": 23420, "low": 23381},
        {"high": 23430, "low": 23410},
        {"high": 23425, "low": 23420},
    ]
    option_chain = {
        "by_strike": {
            "23400": {
                "CE": {"open_interest": 100, "oi_change": 10},
                "PE": {"open_interest": 100, "oi_change": 10},
            }
        }
    }

    ladder = build_connected_ladder(
        [23400], {}, option_chain=option_chain, spot=23400, candles=candles
    )

    assert ladder[23400.0]["support"] == 23381.0
    assert ladder[23400.0]["resistance"] == 23430.0
    assert ladder[23400.0]["resistance"] - ladder[23400.0]["support"] == 49.0


def test_oi_cutoff_is_applied_to_originating_strike_not_candle_price():
    pivot_ladder = {"24400": {"support": 24381.0, "resistance": 24430.0}}
    market_structure = {
        "unified_chain": {"target_up": 24450.0},
        "pivot_levels": {"r1": 24475.0},
    }

    assert DecisionEngine._within_oi_pivot_cutoff(
        24430.0, pivot_ladder, current_price=24350.0, option_type="CE"
    )
    assert DecisionEngine._pivot_target(
        market_structure, 24350.0, "CE", pivot_ladder
    ) == 24430.0


def test_cross_side_entry_uses_outermost_of_oi_and_oi_change_peak_strikes():
    option_chain = {
        "by_strike": {
            "23350": {"PE": {"open_interest": 100, "oi_change": 10},
                      "CE": {"open_interest": 10, "oi_change": 5}},
            "23400": {"PE": {"open_interest": 20, "oi_change": 200},
                      "CE": {"open_interest": 300, "oi_change": 10}},
            "23450": {"PE": {"open_interest": 10, "oi_change": 5},
                      "CE": {"open_interest": 20, "oi_change": 300}},
        }
    }

    assert DecisionEngine._opposite_side_oi_peak_strike(
        option_chain, current_price=23450, trade_option_type="CE"
    ) == (23400.0, "OI-change 100%")
    assert DecisionEngine._opposite_side_oi_peak_strike(
        option_chain, current_price=23350, trade_option_type="PE"
    ) == (23400.0, "OI 100%")


def test_nifty_oi_entry_uses_selected_strikes_own_entry_and_target_sr():
    option_chain = {
        "by_strike": {
            "23350": {"PE": {"open_interest": 100, "oi_change": 10},
                      "CE": {"open_interest": 10, "oi_change": 5}},
            "23400": {"PE": {"open_interest": 20, "oi_change": 200},
                      "CE": {"open_interest": 300, "oi_change": 10}},
            "23450": {"PE": {"open_interest": 10, "oi_change": 5},
                      "CE": {"open_interest": 20, "oi_change": 300}},
        }
    }
    ladder = {
        23350.0: {"support": 23334.0, "resistance": 23381.0},
        23400.0: {"support": 23381.0, "resistance": 23430.0},
        23450.0: {"support": 23430.0, "resistance": 23482.0},
    }

    assert DecisionEngine._nifty_oi_entry(
        option_chain, ladder, current_price=23450, trade_option_type="CE"
    ) == (23400.0, "OI-change 100%", 23381.0, 23430.0)
    assert DecisionEngine._nifty_oi_entry(
        option_chain, ladder, current_price=23350, trade_option_type="PE"
    ) == (23400.0, "OI 100%", 23430.0, 23381.0)


def test_decision_buys_selected_pe_peak_strike_ce_from_support_to_resistance():
    candles = [
        {"high": 23370, "low": 23340},
        {"high": 23375, "low": 23334},
        {"high": 23381, "low": 23339},
        {"high": 23376, "low": 23341},
        {"high": 23430, "low": 23390},
        {"high": 23425, "low": 23385},
        {"high": 23482, "low": 23415},
        {"high": 23440, "low": 23410},
        {"high": 23445, "low": 23420},
    ]
    option_chain = {
        "by_strike": {
            "23350": {
                "PE": {"open_interest": 100, "oi_change": 20, "trade_volume": 1000},
                "CE": {"open_interest": 10, "oi_change": 5, "trade_volume": 100},
            },
            "23400": {
                "CE": {"open_interest": 100, "oi_change": 20, "trade_volume": 1000},
                "PE": {"open_interest": 10, "oi_change": 5, "trade_volume": 100},
            },
            "23450": {
                "CE": {"open_interest": 20, "oi_change": 200, "trade_volume": 1000},
                "PE": {"open_interest": 5, "oi_change": 2, "trade_volume": 100},
            },
        }
    }
    ladder = build_connected_ladder(
        [23350, 23400, 23450], {}, option_chain=option_chain, spot=23341, candles=candles
    )
    risk = Mock()
    risk.evaluate.return_value = SimpleNamespace(approved=True, reasons=[])
    engine = DecisionEngine(risk_manager=risk)
    snapshot = {"spot": 23341, "underlying": "NIFTY", "market_strike": 23350}
    market_structure = {
        "support": ladder[23350.0]["support"],
        "resistance": ladder[23350.0]["resistance"],
        "trend": "BULLISH",
        "micro_momentum": "BULLISH",
        "option_chain": option_chain,
        "pivot_ladder": ladder,
        "candlestick_patterns": {},
    }

    with patch.object(engine, "_is_auto_square_off_time", return_value=False):
        decision = engine.decide(snapshot, market_structure)

    assert decision.action == "BUY"
    assert decision.option_type == "CE"
    assert decision.strike == 23350.0
    assert decision.support == 23334.0
    assert decision.index_target == 23430.0
    assert any("CE same-side OI 100% strike 23400" in reason for reason in decision.reasons)


def test_decision_buys_selected_ce_peak_strike_pe_from_resistance_to_support():
    option_chain = {
        "by_strike": {
            "23400": {
                "CE": {"open_interest": 100, "oi_change": 200, "trade_volume": 1000},
                "PE": {"open_interest": 10, "oi_change": 5, "trade_volume": 100},
            }
        }
    }
    ladder = {"23400": {"support": 23381.0, "resistance": 23430.0}}
    risk = Mock()
    risk.evaluate.return_value = SimpleNamespace(approved=True, reasons=[])
    engine = DecisionEngine(risk_manager=risk)
    snapshot = {"spot": 23422, "underlying": "NIFTY", "market_strike": 23400}
    market_structure = {
        "support": 23381.0,
        "resistance": 23430.0,
        "trend": "BEARISH",
        "micro_momentum": "BEARISH",
        "option_chain": option_chain,
        "pivot_ladder": ladder,
        "candlestick_patterns": {},
    }

    with patch.object(engine, "_is_auto_square_off_time", return_value=False):
        decision = engine.decide(snapshot, market_structure)

    assert decision.action == "BUY"
    assert decision.option_type == "PE"
    assert decision.strike == 23400.0
    assert decision.resistance == 23430.0
    assert decision.index_target == 23381.0


def test_same_strike_ce_pe_oi_rows_include_pivot_sr():
    rows = PriceEngine.build_same_strike_pivot_rows(
        {
            "by_strike": {
                "24400": {
                    "CE": {"open_interest": 120000, "oi_change": 1500},
                    "PE": {"open_interest": 95000, "oi_change": -250},
                }
            }
        },
        {"24400.0": {"support": 24350, "resistance": None}},
    )

    assert rows == [{
        "strike": 24400.0,
        "ce_oi": 120000.0,
        "ce_oi_change": 1500.0,
        "pe_oi": 95000.0,
        "pe_oi_change": -250.0,
        "support": 24350.0,
        "resistance": None,
    }]


def test_decision_uses_configured_nifty_target_without_40_60_band_clamp():
    risk = Mock()
    risk.evaluate.return_value = SimpleNamespace(approved=True, reasons=[])
    engine = DecisionEngine(risk_manager=risk)
    snapshot = {"spot": 25000.0, "underlying": "NIFTY", "market_strike": 25000.0}
    market_structure = {
        "support": 24980.0,
        "resistance": 25020.0,
        "trend": "BULLISH",
        "micro_momentum": "BULLISH",
        "option_chain": {"by_strike": {}},
        "pivot_ladder": {},
        "candlestick_patterns": {"bullish": True},
        "smc": {"order_blocks": {"bullish": [{"zone": {"low": 24990.0, "high": 25008.0}}]}, "fair_value_gaps": {"bullish": []}},
    }

    with patch.object(engine, "_is_auto_square_off_time", return_value=False), \
         patch.object(engine, "_same_side_oi_target", return_value=(None, None, "")), \
         patch.object(engine, "_peak_oi_pivot_target", return_value=(None, None, "")), \
         patch.object(engine, "_option_wall_pivot_target", return_value=(None, None)), \
         patch.object(engine, "_pivot_oi_change_target", return_value=(None, None)):
        decision = engine.decide(snapshot, market_structure)

    assert decision.action == "BUY"
    assert decision.index_target == 25050.0
    assert any("fallback target" in reason.lower() for reason in decision.reasons)


def test_nifty_entry_uses_50_point_fallback_when_dynamic_target_is_missing():
    risk = Mock()
    risk.evaluate.return_value = SimpleNamespace(approved=True, reasons=[])
    engine = DecisionEngine(risk_manager=risk)
    snapshot = {"spot": 25000.0, "underlying": "NIFTY", "market_strike": 25000.0}
    option_chain = {
        "by_strike": {
            "25000": {"PE": {"open_interest": 100, "oi_change": 10}},
            "25050": {"CE": {"open_interest": 100, "oi_change": 10}},
        }
    }
    market_structure = {
        "support": 24992.0,
        "resistance": 25050.0,
        "trend": "BULLISH",
        "micro_momentum": "BULLISH",
        "option_chain": option_chain,
        "pivot_ladder": {
            "25000": {"support": 24992.0, "resistance": None},
            "25050": {"support": None, "resistance": None},
        },
        "candlestick_patterns": {},
    }

    with patch.object(engine, "_is_auto_square_off_time", return_value=False), \
         patch.object(engine, "_peak_oi_pivot_target", return_value=(None, None, "")), \
         patch.object(engine, "_option_wall_pivot_target", return_value=(None, None)), \
         patch.object(engine, "_pivot_oi_change_target", return_value=(None, None)):
        decision = engine.decide(snapshot, market_structure)

    assert decision.action == "BUY"
    assert decision.option_type == "CE"
    assert decision.strike == 25000.0
    assert decision.index_target == 25050.0
    assert decision.index_sl == 24975.0
    assert any("configured target_points=50.0" in reason for reason in decision.reasons)