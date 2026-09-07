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
