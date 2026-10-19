from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from engine.crude_outcome_filter import (
    OUTCOME_FEATURES,
    build_live_outcome_features,
    build_outcome_features,
    fit_outcome_filter,
    predict_win_probability,
)
from trading_crude import CrudeOptionBuyer


def _outcome_rows(count=150):
    rows = []
    for index in range(count):
        winning = index % 2 == 0
        rows.append({
            "Entry Timestamp": f"2026-01-{1 + index // 10:02d} {index % 10:02d}:00:00",
            "Symbol": "CRUDEOILCE" if index % 4 < 2 else "CRUDEOILPE",
            "PnL": 100.0 if winning else -60.0,
            "Entry Volume": 200.0 if winning else 20.0,
            "Entry OI": 1000.0,
            "Entry OI_Change": 100.0 if winning else 5.0,
            "Entry Index Value": 100.0,
            "Entry Pivot Price": 99.0,
            "Entry Candle Open": 99.0,
            "Entry Candle High": 102.0 if winning else 100.0,
            "Entry Candle Low": 98.0,
            "Entry Candle Close": 101.0 if winning else 99.5,
            "Entry NYMEX_Trend": "GREEN" if winning else "RED",
            "Entry Market Regime": "TRENDING" if winning else "SIDEWAYS",
            "Entry ATR": 1.0,
            "Entry Momentum Strength": 0.8 if winning else 0.1,
        })
    return pd.DataFrame(rows)


def _live_signal():
    return {
        "side": "CE",
        "volume": 200.0,
        "oi": 1000.0,
        "oi_change": 100.0,
        "futures_price": 100.0,
        "entry_pivot_price": 99.0,
        "candle_open": 99.0,
        "candle_high": 102.0,
        "candle_low": 98.0,
        "candle_close": 101.0,
        "nymex_trend": "GREEN",
        "market_regime": "TRENDING",
        "atr": 1.0,
        "ai_evaluation": {"momentum_strength": 0.8},
    }


def test_outcome_filter_backtest_approves_only_predictive_synthetic_holdout():
    artifact, report = fit_outcome_filter(_outcome_rows())
    live_features = build_live_outcome_features(_live_signal())

    assert report["approved_for_veto"] is True
    assert tuple(live_features.columns) == OUTCOME_FEATURES
    assert predict_win_probability(artifact, live_features) > 0.5
    assert report["test_selected"]["total_pnl"] > 0


def test_live_and_sheet_feature_builders_produce_identical_row():
    sheet_row = _outcome_rows(1)
    signal = _live_signal()
    live_features = build_live_outcome_features(signal)
    historical_features = build_outcome_features(sheet_row)

    pd.testing.assert_frame_equal(live_features, historical_features)


def test_unapproved_model_never_vetoes_existing_rule_signal():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine.settings = SimpleNamespace(crude_ai_filter_mode="VETO")
    engine._outcome_filter = {"approved_for_veto": False, "threshold": 0.55}
    engine._set_no_signal_reason = Mock()
    signal = _live_signal()

    with patch("trading_crude.build_live_outcome_features", return_value=pd.DataFrame([{}])), \
         patch("trading_crude.predict_win_probability", return_value=0.1):
        blocked = engine._apply_outcome_filter(signal)

    assert blocked is False
    assert signal["ai_filter_mode"] == "SHADOW_UNAPPROVED"
    engine._set_no_signal_reason.assert_not_called()


def test_holdout_approved_model_vetoes_only_below_threshold():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine.settings = SimpleNamespace(crude_ai_filter_mode="VETO")
    engine._outcome_filter = {"approved_for_veto": True, "threshold": 0.55}
    engine._set_no_signal_reason = Mock()
    signal = _live_signal()

    with patch("trading_crude.build_live_outcome_features", return_value=pd.DataFrame([{}])), \
         patch("trading_crude.predict_win_probability", return_value=0.40):
        blocked = engine._apply_outcome_filter(signal)

    assert blocked is True
    engine._set_no_signal_reason.assert_called_once()


def test_shadow_model_records_probability_without_blocking():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine.settings = SimpleNamespace(crude_ai_filter_mode="SHADOW")
    engine._outcome_filter = {"approved_for_veto": True, "threshold": 0.55}
    signal = _live_signal()

    with patch("trading_crude.build_live_outcome_features", return_value=pd.DataFrame([{}])), \
         patch("trading_crude.predict_win_probability", return_value=0.80):
        blocked = engine._apply_outcome_filter(signal)

    assert blocked is False
    assert signal["ai_win_probability"] == 0.80
    assert signal["ai_filter_mode"] == "SHADOW"