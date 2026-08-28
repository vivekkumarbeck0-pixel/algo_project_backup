from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, Mock, patch

from trading_crude import Bar, CrudeOptionBuyer, Position


def _make_engine(state_path):
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine.state_path = Path(state_path)
    engine._position = None
    engine._trail_distance = 3.0
    engine.paper_realized_pnl = 0.0
    engine.paper_wins = 0
    engine.paper_losses = 0
    return engine


def test_crude_state_restores_open_position_on_same_day_restart():
    with TemporaryDirectory() as temporary_directory:
        state_path = f"{temporary_directory}/crude_daily_state.json"
        original = _make_engine(state_path)
        original._trail_distance = 7.5
        original.paper_realized_pnl = 125.0
        original.paper_wins = 2
        original.paper_losses = 1
        original._position = Position(
            side="CE",
            strike=7200,
            entry_price=84.5,
            entry_time=datetime.now(),
            stop_loss=70.0,
            target_price=110.0,
            trailing_stop=78.0,
            atr_value=12.0,
            futures_entry=7180.0,
            futures_entry_oi=1000.0,
            scenario="LONG_BUILDUP",
            option_token="12345",
            option_symbol="CRUDEOIL CE",
            quantity=100,
        )
        original._persist_state()

        restarted = _make_engine(state_path)
        restarted._restore_state()

        assert restarted._position is not None
        assert restarted._position.option_symbol == "CRUDEOIL CE"
        assert restarted._position.trailing_stop == 78.0
        assert restarted._trail_distance == 7.5
        assert restarted.paper_realized_pnl == 125.0


def test_crude_stop_persists_open_position_without_closing_it():
    with TemporaryDirectory() as temporary_directory:
        engine = _make_engine(f"{temporary_directory}/crude_daily_state.json")
        engine._position = Position(
            side="PE",
            strike=7150,
            entry_price=65.0,
            entry_time=datetime.now(),
            stop_loss=52.0,
            target_price=90.0,
            trailing_stop=60.0,
            atr_value=10.0,
            futures_entry=7160.0,
            futures_entry_oi=900.0,
            scenario="SHORT_BUILDUP",
        )
        engine._smart_stream = Mock()
        engine.nymex_filter = Mock()

        engine.stop()

        assert engine._position is not None
        assert engine.state_path.exists()
        engine._smart_stream.close.assert_called_once()
        engine.nymex_filter.stop.assert_called_once()


def test_crude_exit_logs_complete_trade_data_to_google_sheets():
    engine = _make_engine("unused.json")
    engine.settings = Mock(option_lot_size=100)
    engine.current_price = 7_200.0
    engine.current_oi = 1_250.0
    engine.futures_bars = [
        Bar(
            timestamp=datetime.now(),
            open=7_200.0,
            high=7_205.0,
            low=7_195.0,
            close=7_200.0,
            volume=321.0,
        )
    ]
    engine._write_paper_trade_log = Mock()
    engine._persist_state = Mock()
    engine.nymex_filter = Mock(trend="RED")
    engine._position = Position(
        side="CE",
        strike=7200,
        entry_price=80.0,
        entry_time=datetime.now(),
        stop_loss=70.0,
        target_price=100.0,
        trailing_stop=75.0,
        atr_value=10.0,
        futures_entry=7180.0,
        futures_entry_oi=1_000.0,
        scenario="Long Buildup",
        option_symbol="CRUDEOIL17SEP267200CE",
        quantity=100,
        pivot_level=7175.0,
        nymex_trend="GREEN",
        entry_volume=150.0,
        entry_oi_change=50.0,
    )

    with patch("trading_crude.log_trade") as log_trade:
        engine._close_position(90.0, "TARGET HIT")

    log_trade.assert_called_once_with(
        {
            "Entry Timestamp": ANY,
            "Exit Timestamp": ANY,
            "Symbol": "CRUDEOIL17SEP267200CE",
            "Action": "CE",
            "Entry Price": 80.0,
            "Exit Price": 90.0,
            "SL": 70.0,
            "TP": 100.0,
            "PnL": 1_000.0,
            "Entry Scenario": "Long Buildup",
            "Exit Scenario": "TARGET HIT",
            "Entry OI": 1_000.0,
            "Exit OI": 1_250.0,
            "Entry OI_Change": 50.0,
            "Exit OI_Change": 250.0,
            "Entry NYMEX_Trend": "GREEN",
            "Exit NYMEX_Trend": "RED",
            "Entry Volume": 150.0,
            "Exit Volume": 321.0,
        }
    )


def test_crude_pe_exit_keeps_pe_in_combined_google_sheets_record():
    engine = _make_engine("unused.json")
    engine.settings = Mock(option_lot_size=100)
    engine.current_price = 7_100.0
    engine.current_oi = 900.0
    engine.futures_bars = []
    engine._write_paper_trade_log = Mock()
    engine._persist_state = Mock()
    engine.nymex_filter = Mock(trend="NEUTRAL")
    engine._position = Position(
        side="PE",
        strike=7100,
        entry_price=75.0,
        entry_time=datetime.now(),
        stop_loss=60.0,
        target_price=95.0,
        trailing_stop=65.0,
        atr_value=10.0,
        futures_entry=7_120.0,
        futures_entry_oi=950.0,
        scenario="Short Buildup",
        option_symbol="CRUDEOIL17SEP267100PE",
        quantity=100,
        entry_order_id="BUY-PE-1",
    )

    with patch("trading_crude.log_trade") as log_trade:
        engine._close_position(85.0, "TARGET HIT")

    payload = log_trade.call_args.args[0]
    assert payload["Symbol"].endswith("PE")
    assert payload["Action"] == "PE"
    assert payload["Entry Price"] == 75.0
    assert payload["Exit Price"] == 85.0