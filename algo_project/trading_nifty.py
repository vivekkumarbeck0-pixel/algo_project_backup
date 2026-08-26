"""Standalone NIFTY trading session.

Run with: ``python trading_nifty.py``
"""

import time
from datetime import datetime, time as dt_time

from config import settings
from engine.trading_session import IST, LivePaperTradingSession


class NiftyTradingSession(LivePaperTradingSession):
    """NIFTY-only session with NSE hours and 1m/3m/5m chart confluence."""

    def __init__(self, image_path: str = "data/aoc.png"):
        self.image_path = image_path
        super().__init__(default_symbol="NIFTY")

    def _session_symbol(self) -> str:
        return "NIFTY"

    def _candle_intervals(self) -> tuple[str, ...]:
        return ("FIVE_MINUTE", "THREE_MINUTE", "ONE_MINUTE")

    def _sr_settings(self) -> dict:
        # NIFTY strikes sit on a 50-point grid, so the hybrid S/R zone is
        # exactly [strike - 25, strike + 25] (e.g. 24200 -> 24175-24225).
        return {"sr_range_offset": 25.0, "sr_range_tolerance": 0.0}

    def _strike_step(self) -> float:
        return 50.0

    def _market_hours(self) -> tuple[dt_time, dt_time]:
        return dt_time(9, 15), dt_time(15, 30)

    def _state_file(self) -> str:
        return "data/nifty_daily_state.json"

    def _dashboard_metric_labels(self) -> tuple[str, str]:
        return "India VIX", "Option IV"

    def print_daily_summary(self) -> None:
        """Print the NIFTY book when the session stops or is interrupted."""
        trades = sorted(self.tracker.trades_today(), key=lambda position: position.opened_at)
        closed_trades = [position for position in trades if position.closed_at is not None]
        wins = sum(1 for position in closed_trades if (position.pnl or 0.0) > 0.0)
        losses = sum(1 for position in closed_trades if (position.pnl or 0.0) < 0.0)

        print()
        print("=" * 124)
        print(f"DAILY NIFTY TRADING SUMMARY - {datetime.now(IST).strftime('%Y-%m-%d')}")
        print("=" * 124)
        print(f"Total Trades Executed Today : {len(trades)}")
        print(f"Win / Loss Count             : {wins} / {losses}")
        print("-" * 124)

        header = (
            f"{'#':<4}{'Symbol':<20}{'Entry Time':<20}{'Exit Time':<20}"
            f"{'Entry Price':<14}{'Exit Price':<14}{'Realized P&L':<16}{'Exit Reason'}"
        )
        print(header)
        print("-" * 124)

        if not trades:
            print("No Nifty trades executed today.")
        else:
            for position in trades:
                symbol = f"{position.symbol} {position.strike:g}{position.option_type}"
                exit_time = position.closed_at.strftime("%Y-%m-%d %H:%M:%S") if position.closed_at else "--"
                exit_price = f"{position.exit_price:.2f}" if position.exit_price is not None else "--"
                realized_pnl = f"{position.pnl:+.2f}" if position.pnl is not None else "--"
                reason = position.close_reason or "OPEN"
                print(
                    f"{position.trade_number:<4}{symbol:<20}"
                    f"{position.opened_at.strftime('%Y-%m-%d %H:%M:%S'):<20}"
                    f"{exit_time:<20}{position.entry_price:<14.2f}{exit_price:<14}"
                    f"{realized_pnl:<16}{reason}"
                )

        print("-" * 124)
        print(f"Net Realized P&L for Nifty Today : {self.tracker.daily_realized_pnl():+.2f}")
        print("=" * 124)

    def run_forever(self, interval_seconds: float | None = None) -> None:
        """Run the Nifty session and always show its daily book on Ctrl+C."""
        interval = interval_seconds if interval_seconds is not None else settings.live_poll_interval_seconds
        interval = max(2.0, min(10.0, interval))

        print(f"Starting live Nifty paper trading session (interval={interval}s)")
        try:
            while True:
                try:
                    self.run_once()
                except Exception as exc:
                    print(f"Nifty trading session tick failed: {exc}")

                max_trades = self.risk_manager.limits.max_trades_per_day
                limit_reached = max_trades is not None and self.tracker.trades_today_count() >= max_trades
                if limit_reached and not self.tracker.open_positions():
                    print("\nDaily trade limit reached and no open positions - stopping session.")
                    break

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nNifty session interrupted by user. Preparing daily summary...")
        finally:
            self.print_daily_summary()


if __name__ == "__main__":
    NiftyTradingSession().run_forever(settings.live_poll_interval_seconds)
