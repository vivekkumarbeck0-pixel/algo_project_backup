from datetime import datetime, time
from unittest.mock import Mock, patch

import pytest

from engine.position_tracker import PositionTracker
from engine.trading_session import IST
from trading_nifty import NiftyTradingSession


def test_nifty_market_hours_start_at_9_am():
    assert NiftyTradingSession._market_hours(NiftyTradingSession.__new__(NiftyTradingSession)) == (
        time(9, 0),
        time(15, 30),
    )


def test_nifty_dashboard_displays_ist_time(capsys):
    session = NiftyTradingSession.__new__(NiftyTradingSession)
    session._refresh_console_view = Mock()
    session.order_manager = Mock()
    session.order_manager.mode.value = "PAPER"
    session._last_metrics = {}
    session._metrics_updated_at = None
    session._pivot_cache = {}
    session.tracker = Mock()
    session.tracker.trades_today_count.return_value = 0
    session.tracker.consecutive_stop_losses.return_value = 0
    session.tracker.daily_realized_pnl.return_value = 0.0
    session.tracker.open_positions.return_value = []
    session.risk_manager = Mock()
    session.risk_manager.limits.max_trades_per_day = None
    session.risk_manager.limits.max_consecutive_stop_losses = None
    with patch("engine.trading_session.datetime") as current_datetime:
        current_datetime.now.return_value = type("Clock", (), {
            "strftime": lambda self, pattern: "2026-09-17 09:17:00",
        })()
        session._print_dashboard({"market_status": "OPEN"}, "NIFTY", None, None)

    current_datetime.now.assert_called_with(IST)
    assert "2026-09-17 09:17:00" in capsys.readouterr().out


@pytest.mark.parametrize("exit_ltp", [278.35, None])
def test_nifty_closed_market_does_not_fill_at_stale_quote(exit_ltp):
    session = NiftyTradingSession.__new__(NiftyTradingSession)
    session.default_symbol = "NIFTY"
    session.tracker = PositionTracker()
    position = session.tracker.open_position("NIFTY", 23450, "PE", "BUY", 65, 195.95)
    session._is_market_open = Mock(return_value=False)
    session._square_off_due = Mock(return_value=True)
    session._option_ltp_lookup = Mock(return_value=exit_ltp)
    session._log_nifty_exit = Mock()
    session._record_training_outcome = Mock()
    session._print_trade_closed = Mock()
    session.print_daily_summary = Mock()
    session._fallback_metrics = Mock(return_value={})
    session._print_dashboard = Mock()
    session._persist_state = Mock()

    session.run_once()

    session._option_ltp_lookup.assert_not_called()
    assert session.tracker.open_positions() == [position]
    assert "Square-off pending" in session._print_dashboard.call_args.kwargs["blocked_reason"]
    session._persist_state.assert_called_once()


def test_nifty_restored_position_without_option_ltp_is_dropped():
    session = NiftyTradingSession.__new__(NiftyTradingSession)
    session.tracker = PositionTracker()
    session.tracker.open_position("NIFTY", 23450, "PE", "BUY", 65, 195.95)
    session.state_store = Mock()
    session.state_store.load.return_value = {"positions": session.tracker.export_state()}
    session._option_ltp_lookup = Mock(return_value=None)
    session._persist_state = Mock()

    session._load_state()

    assert not session.tracker.open_positions()
    dropped_position = session.tracker.closed_positions()[0]
    assert dropped_position.status.value == "CLOSED"
    assert dropped_position.exit_price is None
    assert dropped_position.close_reason == "STARTUP_DROPPED"
    session._option_ltp_lookup.assert_called_once()
    session._persist_state.assert_called_once()


def test_nifty_restored_prior_day_position_is_dropped_without_ltp_lookup():
    session = NiftyTradingSession.__new__(NiftyTradingSession)
    session.tracker = PositionTracker()
    position = session.tracker.open_position("NIFTY", 23450, "PE", "BUY", 65, 195.95)
    position.opened_at = datetime(2000, 1, 1, 9, 30)
    session.state_store = Mock()
    session.state_store.load.return_value = {"positions": session.tracker.export_state()}
    session._option_ltp_lookup = Mock(return_value=195.95)
    session._persist_state = Mock()

    session._load_state()

    assert not session.tracker.open_positions()
    assert session.tracker.closed_positions()[0].close_reason == "STARTUP_DROPPED"
    session._option_ltp_lookup.assert_not_called()
    session._persist_state.assert_called_once()


def test_nifty_restored_position_with_valid_option_ltp_is_retained():
    session = NiftyTradingSession.__new__(NiftyTradingSession)
    session.tracker = PositionTracker()
    session.tracker.open_position("NIFTY", 23450, "PE", "BUY", 65, 195.95)
    session.state_store = Mock()
    session.state_store.load.return_value = {"positions": session.tracker.export_state()}
    session._option_ltp_lookup = Mock(return_value=195.95)
    session._persist_state = Mock()

    session._load_state()

    assert len(session.tracker.open_positions()) == 1
    session._option_ltp_lookup.assert_called_once()
    session._persist_state.assert_not_called()


@pytest.mark.parametrize("exit_ltp", [278.35, None])
@pytest.mark.parametrize("rollover", [True, False])
def test_nifty_cutoff_and_next_session_exits_before_signals(exit_ltp, rollover):
    session = NiftyTradingSession.__new__(NiftyTradingSession)
    session.default_symbol = "NIFTY"
    session.tracker = PositionTracker()
    position = session.tracker.open_position("NIFTY", 23450, "PE", "BUY", 65, 195.95)
    position.opened_at = datetime(2026, 9, 15, 9, 38) if rollover else datetime.now(IST)
    position.entry_metadata["option_instrument"] = {"token": "42", "exch_seg": "NFO", "symbol": "NIFTY23450PE"}
    session._is_market_open = Mock(return_value=True)
    session._square_off_due = Mock(return_value=not rollover)
    session._resolve_underlying = Mock(return_value={"token": "1", "symbol": "NIFTY"})
    session.market_data = Mock()
    session.market_data.fetch_latest_price.return_value = 23442.5
    session._market_structure = Mock(return_value={"trend": "NEUTRAL"})
    session._fallback_metrics = Mock(return_value={"trend": "NEUTRAL", "micro_momentum": "NEUTRAL"})
    session._refresh_metrics = Mock(return_value={"india_vix": None, "option_iv": None})
    session._nearest_strike = Mock(return_value=23450)
    session._check_all_exits = Mock(return_value=[])
    session._check_reversal_exits = Mock(return_value=[])
    session._option_ltp_lookup = Mock(return_value=exit_ltp)
    session._log_nifty_exit = Mock()
    session._record_training_outcome = Mock()
    session._print_trade_closed = Mock()
    session.print_daily_summary = Mock()
    session._print_dashboard = Mock()
    session._persist_state = Mock()
    session.risk_manager = Mock()
    session.risk_manager.limits.max_trades_per_day = 0

    session.run_once()

    if rollover:
        session._option_ltp_lookup.assert_called_once_with(position, live_only=True)
    else:
        session._option_ltp_lookup.assert_called_once_with(position)
    if exit_ltp is None:
        assert session.tracker.open_positions() == [position]
        assert "Square-off pending" in session._print_dashboard.call_args.args[-1]
    else:
        assert not session.tracker.open_positions()
        assert position.exit_price == exit_ltp
        assert position.close_reason == ("SESSION_ROLLOVER" if rollover else "TIME_EXIT")
        session._print_trade_closed.assert_called_once_with(position)


def test_nifty_rollover_rejects_cached_option_price():
    session = NiftyTradingSession.__new__(NiftyTradingSession)
    session.instrument_reader = Mock()
    session.instrument_reader.find_option_token.return_value = {"token": "42", "exch_seg": "NFO"}
    session.market_data = Mock()
    session.market_data._live_quote.return_value = None
    position = PositionTracker().open_position("NIFTY", 23450, "PE", "BUY", 65, 195.95)
    position.entry_metadata["option_instrument"] = {"token": "42", "exch_seg": "NFO"}

    assert session._option_ltp_lookup(position, live_only=True) is None
    session.market_data.fetch_latest_price.assert_not_called()


def test_nifty_legacy_position_does_not_use_a_new_expiry_for_rollover():
    session = NiftyTradingSession.__new__(NiftyTradingSession)
    session.instrument_reader = Mock()
    session.market_data = Mock()
    position = PositionTracker().open_position("NIFTY", 23450, "PE", "BUY", 65, 195.95)
    position.opened_at = datetime(2026, 9, 15, 9, 38)

    assert session._option_ltp_lookup(position, live_only=True) is None
    assert session._option_ltp_lookup(position) is None
    session.instrument_reader.find_option_token.assert_not_called()
    session.market_data._live_quote.assert_not_called()