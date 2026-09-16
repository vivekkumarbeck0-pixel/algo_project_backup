from datetime import datetime, time as datetime_time, timezone
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

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

def test_crude_state_rollover_preserves_open_position_and_updates_date():
    with TemporaryDirectory() as temporary_directory:
        state_path = f"{temporary_directory}/crude_daily_state.json"
        original = _make_engine(state_path)
        original._position = Position(
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
        original._persist_state()

        payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
        payload["date"] = "2000-01-01"
        payload["paper_realized_pnl"] = 125.0
        payload["paper_wins"] = 2
        Path(state_path).write_text(json.dumps(payload), encoding="utf-8")

        restarted = _make_engine(state_path)
        restarted._restore_state()

        assert restarted._position is not None
        assert restarted._position.side == "PE"
        assert restarted.paper_realized_pnl == 0.0
        assert json.loads(Path(state_path).read_text(encoding="utf-8"))["date"] == datetime.now(timezone.utc).astimezone().date().isoformat()


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
        entry_candle_open=7_180.0,
        entry_candle_high=7_190.0,
        entry_candle_low=7_175.0,
        entry_candle_close=7_185.0,
    )

    with patch("trading_crude.log_trade") as log_trade:
        engine._close_position(90.0, "TARGET HIT")

    payload = log_trade.call_args.args[0]
    assert payload["Entry Candle Open"] == 7_180.0
    assert payload["Entry Candle High"] == 7_190.0
    assert payload["Entry Candle Low"] == 7_175.0
    assert payload["Entry Candle Close"] == 7_185.0
    assert payload["Exit Candle Open"] == 7_200.0
    assert payload["Exit Candle High"] == 7_205.0
    assert payload["Exit Candle Low"] == 7_195.0
    assert payload["Exit Candle Close"] == 7_200.0
    assert payload["Entry ATR"] == 10.0
    assert payload["Exit ATR"] is None
    assert payload["Entry Market Regime"] is None
    assert payload["Exit Market Regime"] is None


def test_crude_entry_captures_latest_candle_for_google_sheets():
    engine = _make_engine("unused.json")
    engine.settings = Mock(
        execution_mode="PAPER",
        allow_real_trading=False,
        option_lot_size=100,
        option_exchange="MCX",
        buffer_points=5.0,
    )
    engine.instrument = Mock(symbol="CRUDEOIL")
    engine.current_price = 7_185.0
    engine.futures_bars = [
        Bar(
            timestamp=datetime.now(),
            open=7_180.0,
            high=7_190.0,
            low=7_175.0,
            close=7_185.0,
            volume=150.0,
        )
    ]
    engine._entry_cutoff_time = Mock(return_value=datetime_time(23, 59))
    engine._resolve_option_contract = Mock(
        return_value={"token": "12345", "symbol": "CRUDEOIL17SEP267200CE", "lotsize": "100"}
    )
    engine._entry_option_ltp = Mock(return_value=80.0)
    engine._risk_levels = Mock(
        return_value={"stop_loss": 70.0, "target_price": 100.0, "trail_distance": 15.0}
    )
    engine._nearest_pivot_info = Mock(return_value=("PP", 0, 7_180.0))
    engine._square_off_time = Mock(return_value=datetime_time(23, 30))
    engine._print_block = Mock()
    engine._write_paper_trade_log = Mock()
    engine._persist_state = Mock()
    engine._smart_stream = Mock()

    engine._place_entry_order(
        {
            "scenario": "Long Buildup",
            "side": "CE",
            "strike": 7_200,
            "futures_price": 7_185.0,
            "oi": 1_000.0,
            "volume": 150.0,
            "atr": 10.0,
            "pivot_level_name": "PP",
            "pivot_level": 7_180.0,
            "buffer_points": 5.0,
            "next_pivot": 7_200.0,
            "nymex_trend": "GREEN",
        }
    )

    assert engine._position is not None
    assert engine._position.entry_candle_open == 7_180.0
    assert engine._position.entry_candle_high == 7_190.0
    assert engine._position.entry_candle_low == 7_175.0
    assert engine._position.entry_candle_close == 7_185.0


def test_crude_exit_logs_sheet_timestamps_in_ist_format():
    engine = _make_engine("unused.json")
    engine.settings = Mock(option_lot_size=100)
    engine.current_price = 7_200.0
    engine.current_oi = 1_250.0
    engine.futures_bars = []
    engine._write_paper_trade_log = Mock()
    engine._persist_state = Mock()
    engine.nymex_filter = Mock(trend="GREEN")
    engine._position = Position(
        side="CE",
        strike=7200,
        entry_price=80.0,
        entry_time=datetime(2026, 8, 28, 9, 15, tzinfo=timezone.utc),
        stop_loss=70.0,
        target_price=100.0,
        trailing_stop=75.0,
        atr_value=10.0,
        futures_entry=7180.0,
        futures_entry_oi=1_000.0,
        scenario="Long Buildup",
        option_symbol="CRUDEOIL17SEP267200CE",
        quantity=100,
    )

    fixed_exit_ist = datetime(2026, 8, 28, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
    with patch("trading_crude._now_ist", return_value=fixed_exit_ist), patch("trading_crude.log_trade") as log_trade:
        engine._close_position(90.0, "TARGET HIT")

    payload = log_trade.call_args.args[0]
    assert payload["Entry Timestamp"] == "2026-08-28 14:45:00"
    assert payload["Exit Timestamp"] == "2026-08-28 15:30:00"


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


def test_crude_trailing_stop_starts_after_eight_profit_points():
    engine = _make_engine("unused.json")
    engine.settings = Mock(crude_trailing_activation_points=8.0)
    engine.current_price = 7_200.0
    engine._trail_distance = 3.0
    engine._current_option_price = Mock(return_value=107.0)
    engine._persist_state = Mock()
    engine._close_position = Mock()
    engine._position = Position(
        side="CE",
        strike=7200,
        entry_price=100.0,
        entry_time=datetime.now(),
        stop_loss=90.0,
        target_price=120.0,
        trailing_stop=104.0,
        atr_value=5.0,
        futures_entry=7_180.0,
        futures_entry_oi=1_000.0,
        scenario="Long Buildup",
    )

    engine._update_position_management()

    assert engine._position.trailing_stop == 104.0
    engine._persist_state.assert_not_called()
    engine._close_position.assert_not_called()

    engine._current_option_price.return_value = 110.0

    engine._update_position_management()

    assert engine._position.trailing_stop == 107.0
    engine._persist_state.assert_called_once()
    engine._close_position.assert_not_called()

    engine._current_option_price.return_value = 106.0

    engine._update_position_management()

    engine._close_position.assert_called_once_with(106.0, "TRAILING STOP")