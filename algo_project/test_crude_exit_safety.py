from datetime import datetime, time as datetime_time, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

from trading_crude import CrudeOptionBuyer, Position


def _engine_for_exit_tests():
    engine = CrudeOptionBuyer.__new__(CrudeOptionBuyer)
    engine.settings = SimpleNamespace(
        crude_trailing_activation_points=8.0,
        crude_trailing_breakeven_buffer_points=1.0,
        futures_tick_timeout_seconds=5.0,
        execution_mode="PAPER",
    )
    engine.instrument = SimpleNamespace(symbol="CRUDEOIL")
    engine.current_price = 7_200.0
    engine.current_timestamp = datetime.now()
    engine._position = Position(
        side="CE",
        strike=7200,
        entry_price=100.0,
        entry_time=datetime.now() - timedelta(seconds=10),
        stop_loss=90.0,
        target_price=120.0,
        trailing_stop=90.0,
        atr_value=5.0,
        futures_entry=7_180.0,
        futures_entry_oi=1_000.0,
        scenario="Long Buildup",
        option_token="option-token",
        option_symbol="CRUDEOIL7200CE",
    )
    engine._current_option_price = Mock(return_value=89.0)
    engine._option_ltp_time = 1.0
    engine._exit_confirmation = None
    engine._close_position = Mock()
    engine._persist_state = Mock()
    engine._sl_cooldowns = {}
    engine._sl_cooldown_seconds = 180.0
    return engine


def test_sl_exit_requires_two_distinct_breached_ticks_not_repeated_loop_reads():
    engine = _engine_for_exit_tests()

    engine._update_position_management()
    engine._update_position_management()
    engine._close_position.assert_not_called()

    engine._option_ltp_time = 2.0
    engine._update_position_management()

    engine._close_position.assert_called_once_with(89.0, "STOP LOSS")


def test_sl_exit_executes_after_three_seconds_of_a_fresh_sustained_breach():
    engine = _engine_for_exit_tests()

    with patch("trading_crude.time.monotonic", side_effect=[100.0, 103.0]):
        engine._update_position_management()
        engine._update_position_management()

    engine._close_position.assert_called_once_with(89.0, "STOP LOSS")


def test_non_breaching_tick_resets_consecutive_exit_confirmation():
    engine = _engine_for_exit_tests()

    engine._update_position_management()
    engine._current_option_price.return_value = 95.0
    engine._option_ltp_time = 2.0
    engine._update_position_management()
    engine._current_option_price.return_value = 89.0
    engine._option_ltp_time = 3.0
    engine._update_position_management()
    engine._close_position.assert_not_called()

    engine._option_ltp_time = 4.0
    engine._update_position_management()
    engine._close_position.assert_called_once_with(89.0, "STOP LOSS")


def test_sl_cooldown_is_per_contract_and_expires_at_180_seconds():
    engine = _engine_for_exit_tests()
    position = engine._position

    with patch("trading_crude.time.time", return_value=1_000.0):
        engine._record_stop_loss_cooldown(position)

    with patch("trading_crude.time.time", return_value=1_000.0):
        assert engine._sl_cooldown_remaining("CRUDEOIL7200CE") == 180.0
        assert engine._sl_cooldown_remaining("CRUDEOIL7150CE") == 0.0
        assert engine._sl_cooldown_remaining("CRUDEOIL") == 180.0

    with patch("trading_crude.time.time", return_value=1_180.0):
        assert engine._sl_cooldown_remaining("CRUDEOIL7200CE") == 0.0
        assert engine._sl_cooldown_remaining("CRUDEOIL") == 0.0


def test_stop_loss_and_trailing_stop_closes_start_contract_cooldown():
    for reason in ("STOP LOSS", "TRAILING STOP"):
        engine = _engine_for_exit_tests()
        engine.settings = SimpleNamespace(option_lot_size=100)
        engine.paper_realized_pnl = 0.0
        engine.paper_wins = 0
        engine.paper_losses = 0
        engine.current_oi = None
        engine.futures_bars = []
        engine._write_paper_trade_log = Mock()
        engine._print_block = Mock()
        del engine._close_position

        with patch("trading_crude.log_trade"):
            engine._close_position(89.0, reason)

        assert engine._sl_cooldown_remaining("CRUDEOIL7200CE") > 179.0


def test_entry_is_skipped_while_the_same_contract_is_in_sl_cooldown():
    engine = _engine_for_exit_tests()
    engine._position = None
    engine._entry_cutoff_time = Mock(return_value=datetime_time(23, 59))
    engine._resolve_option_contract = Mock(
        return_value={"symbol": "CRUDEOIL7150PE", "token": "different-contract"}
    )
    engine._sl_cooldowns = {"CRUDEOIL": 4_000_000_000.0}
    engine._entry_option_ltp = Mock()

    engine._place_entry_order({"strike": 7150, "side": "PE", "futures_price": 7_180.0})

    engine._entry_option_ltp.assert_not_called()
    assert engine._position is None
