from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from trading_crude import CrudeOptionBuyer, Position


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