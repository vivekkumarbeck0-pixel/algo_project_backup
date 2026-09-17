from datetime import time
from unittest.mock import Mock, patch

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