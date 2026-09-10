from types import SimpleNamespace
from unittest.mock import Mock

from engine.crude_hybrid_ml import HybridDecision
from trading_crude import Bar, CrudeOptionBuyer


def test_hybrid_no_trade_is_warning_only_for_rule_based_signal(caplog):
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine._position = None
    engine._last_signal_bar = None
    engine.settings = SimpleNamespace(
        ma_volume_period=5,
        volume_spike_factor=2.0,
        pivot_lookback_bars=20,
        buffer_points=5.0,
    )
    engine.nymex_filter = SimpleNamespace(trend="GREEN")

    bars = [
        Bar(
            timestamp=index,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
            oi=100.0,
        )
        for index in range(22)
    ]
    bars[-1] = Bar(
        timestamp=22,
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=100.0,
        oi=110.0,
    )
    engine.futures_bars = bars + [bars[-1]]
    engine._calculate_daily_pivots = Mock(return_value={"PP": 100.0, "R1": 102.0, "R2": 104.0, "S1": 98.0, "S2": 96.0})
    engine._calculate_atr = Mock(return_value=1.0)
    engine._next_pivot_for_target = Mock(return_value=102.0)
    engine.ai_dynamic_trade_evaluation = Mock(return_value={"approved": True})
    engine._evaluate_hybrid_model = Mock(
        return_value=HybridDecision("NO_TRADE", 0.58, 0.42, "NORMAL", "momentum confidence is inconclusive")
    )
    engine._pivot_breakout = Mock(return_value=(False, None, 0.0))
    engine._nearest_atm_strike = Mock(return_value=100)

    signal = engine._evaluate_strategy()

    assert signal is not None
    assert signal["side"] == "CE"
    assert signal["hybrid_prediction"] == "NO_TRADE"
    assert signal["hybrid_momentum_probability"] == 0.58
    assert "proceeding despite hybrid model warning" in caplog.text
    assert "prediction=NO_TRADE" in caplog.text


def test_crude_signal_records_low_volume_block_reason():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine._position = None
    engine._last_signal_bar = None
    engine._last_no_signal_reason = ""
    engine._last_no_signal_log = 0.0
    engine._no_signal_log_interval = 30.0
    engine.settings = SimpleNamespace(
        ma_volume_period=5,
        volume_spike_factor=2.0,
        pivot_lookback_bars=20,
    )
    engine.nymex_filter = SimpleNamespace(trend="GREEN")

    bars = [
        Bar(
            timestamp=index,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
            oi=100.0,
        )
        for index in range(23)
    ]
    engine.futures_bars = bars
    engine._calculate_daily_pivots = Mock(return_value={"PP": 100.0, "R1": 102.0, "R2": 104.0, "S1": 98.0, "S2": 96.0})
    engine._calculate_atr = Mock(return_value=1.0)
    engine._next_pivot_for_target = Mock(return_value=102.0)

    signal = engine._evaluate_strategy()

    assert signal is None
    assert engine._last_no_signal_reason.startswith("volume not spiking")


def test_crude_market_context_detects_bullish_reversal_in_sideways_range():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine.settings = SimpleNamespace(
        crude_market_context_lookback_bars=8,
        crude_sideways_range_atr_multiplier=5.0,
        crude_sideways_efficiency_threshold=0.35,
        crude_reversal_swing_lookback_bars=5,
    )
    bars = [
        Bar(timestamp=1, open=100.0, high=102.0, low=99.0, close=101.0, volume=10.0, oi=100.0),
        Bar(timestamp=2, open=101.0, high=102.0, low=99.5, close=100.5, volume=10.0, oi=100.0),
        Bar(timestamp=3, open=100.5, high=101.5, low=99.0, close=100.0, volume=10.0, oi=100.0),
        Bar(timestamp=4, open=100.0, high=101.0, low=98.8, close=99.7, volume=10.0, oi=100.0),
        Bar(timestamp=5, open=99.7, high=101.0, low=99.2, close=100.2, volume=10.0, oi=100.0),
        Bar(timestamp=6, open=99.4, high=101.2, low=97.8, close=100.9, volume=10.0, oi=100.0),
    ]

    context = engine._detect_market_context(bars, atr=1.0)

    assert context["regime"] == "SIDEWAYS"
    assert context["reversal"] == "BULLISH"
    assert context["side"] == "CE"
