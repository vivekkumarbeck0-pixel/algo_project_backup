from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sheets_logger import SHEET_COLUMNS, log_trade
from trading_crude import Bar, CrudeOptionBuyer


def test_crude_setup_score_uses_entry_data_and_stays_in_range():
    bars = [
        Bar(timestamp=index, open=100.0, high=101.0, low=99.0, close=100.0, volume=10.0, oi=100.0)
        for index in range(21)
    ]
    previous = bars[-1]
    strong = Bar(timestamp=21, open=100.0, high=102.0, low=99.0, close=101.0, volume=100.0, oi=110.0)
    weak = Bar(timestamp=21, open=101.0, high=102.0, low=99.0, close=101.0, volume=21.0, oi=100.1)

    strong_score = CrudeOptionBuyer._score_trade_setup(
        strong, previous, bars + [strong], "CE", 1.0, 20.0, 10.0, True, "TRENDING"
    )
    weak_score = CrudeOptionBuyer._score_trade_setup(
        weak, previous, bars + [weak], "CE", 1.0, 20.0, 0.1, False, "UNKNOWN"
    )

    assert 0 <= weak_score < strong_score <= 10
    assert strong_score == 8.7


def test_crude_sheet_appends_score_to_existing_headers():
    worksheet = Mock()
    worksheet.get_all_values.return_value = [["Entry Timestamp", "PnL"], ["older trade", "100"]]
    client = Mock()
    client.open.return_value.sheet1 = worksheet

    with patch("sheets_logger.get_gspread_client", return_value=client):
        assert log_trade({"Entry Timestamp": "new trade", "PnL": -100, "Entry AI Score": 7.4})

    assert "Entry AI Score" in SHEET_COLUMNS
    assert SHEET_COLUMNS[-2:] == ["Entry AI Win Probability", "Entry AI Filter Mode"]
    worksheet.update.assert_called_once_with("A1", [["Entry Timestamp", "PnL"] + [
        header for header in SHEET_COLUMNS if header not in ("Entry Timestamp", "PnL")
    ]])
    row = worksheet.append_row.call_args.args[0]
    assert row[SHEET_COLUMNS.index("Entry AI Score")] == 7.4


@pytest.mark.parametrize(
    "ai_evaluation",
    [{"momentum_strength": 0.5}, None],
)
def test_crude_signal_keeps_valid_rule_signal_when_ai_evaluation_is_unavailable(ai_evaluation):
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
    engine.ai_dynamic_trade_evaluation = Mock(return_value=ai_evaluation)
    engine._pivot_breakout = Mock(return_value=(False, None, 0.0))
    engine._nearest_atm_strike = Mock(return_value=100)

    signal = engine._evaluate_strategy()

    assert signal is not None
    assert signal["side"] == "CE"
    assert signal["ai_evaluation"] is ai_evaluation
    assert 0 <= signal["entry_ai_score"] <= 10


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


def test_crude_sideways_entry_accepts_a_pivot_retest_away_from_candle_low():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine.settings = SimpleNamespace(crude_entry_candle_zone=0.35, buffer_points=2.0)
    candle = Bar(timestamp=1, open=103.0, high=105.0, low=99.5, close=104.0, volume=10.0, oi=100.0)
    pivots = {"PP": 100.0, "R1": 106.0, "R2": 110.0, "S1": 96.0, "S2": 92.0}

    assert engine._sideways_entry_ok(candle, pivots, "CE")
