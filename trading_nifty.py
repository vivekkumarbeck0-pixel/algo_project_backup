"""Standalone NIFTY trading session.

Run with: ``python trading_nifty.py``
"""

import math
import os
import time
from datetime import datetime, time as dt_time

from config import settings
from engine.trading_session import IST, LivePaperTradingSession
from logger import get_logger

log = get_logger(__name__)

NIFTY_LIVE_ENABLE_ENV = "NIFTY_REAL_TRADING_ENABLED"
NIFTY_LIVE_CONFIRM_ENV = "NIFTY_REAL_TRADING_CONFIRM"
NIFTY_LIVE_CONFIRM_VALUE = "I_ACCEPT_SOFTWARE_MONITORED_NIFTY_EXITS"
NIFTY_LIVE_MAX_ENTRIES_PER_DAY = 6
NIFTY_LIVE_MAX_DAILY_LOSS = 4000.0


class NiftyTradingSession(LivePaperTradingSession):
    """NIFTY-only session with NSE hours and 1m/3m/5m chart confluence."""

    def __init__(self, image_path: str = "data/aoc.png"):
        self.image_path = image_path
        self._live_recovery_blocked = False
        self._live_entry_enabled = False
        self._live_activation_blocked_reason = None
        super().__init__(default_symbol="NIFTY")
        self._paper_trade_limit = self.risk_manager.limits.max_trades_per_day
        self._paper_daily_loss_limit = self.risk_manager.limits.max_daily_loss
        self._activate_live_from_environment()

    def _initial_execution_mode(self) -> str:
        # NIFTY real execution is opt-in through its own Railway variable only.
        return "PAPER"

    def _activate_live_from_environment(self) -> None:
        enabled = os.environ.get(NIFTY_LIVE_ENABLE_ENV, "").strip().lower() == "true"
        if not enabled:
            log.info("NIFTY real trading is OFF; starting in paper mode.")
            return

        confirmation = os.environ.get(NIFTY_LIVE_CONFIRM_ENV, "").strip()
        if confirmation != NIFTY_LIVE_CONFIRM_VALUE:
            self._live_activation_blocked_reason = (
                f"{NIFTY_LIVE_CONFIRM_ENV} must exactly match the documented confirmation value"
            )
            log.error("NIFTY live trading was requested but remains OFF: %s", self._live_activation_blocked_reason)
            return

        if self._live_recovery_blocked:
            self._live_activation_blocked_reason = (
                "a restored live position requires manual reconciliation in Angel One"
            )
            log.error("NIFTY live trading remains OFF: %s", self._live_activation_blocked_reason)
            return

        if self.tracker.open_positions():
            self._live_activation_blocked_reason = (
                "a local NIFTY position is open; wait until it is closed before enabling live entries"
            )
            log.error("NIFTY live trading remains OFF: %s", self._live_activation_blocked_reason)
            return

        try:
            if self.order_manager.has_existing_nifty_activity():
                self._live_activation_blocked_reason = (
                    "Angel One reports an existing NIFTY position or pending NFO order"
                )
                log.error("NIFTY live trading remains OFF: %s", self._live_activation_blocked_reason)
                return
        except Exception as exc:
            self._live_activation_blocked_reason = f"broker preflight failed: {exc}"
            log.error("NIFTY live trading remains OFF: %s", self._live_activation_blocked_reason)
            return

        try:
            self.risk_manager.limits.max_trades_per_day = NIFTY_LIVE_MAX_ENTRIES_PER_DAY
            self.risk_manager.limits.max_daily_loss = NIFTY_LIVE_MAX_DAILY_LOSS
            self.set_execution_mode("REAL")
        except Exception as exc:
            self.risk_manager.limits.max_trades_per_day = self._paper_trade_limit
            self.risk_manager.limits.max_daily_loss = self._paper_daily_loss_limit
            self._live_activation_blocked_reason = f"could not initialize live execution: {exc}"
            log.error("NIFTY live trading remains OFF: %s", self._live_activation_blocked_reason)
            return

        self._live_entry_enabled = True
        log.warning(
            "NIFTY REAL TRADING ENABLED: market orders, one option lot, max %d entries/day, "
            "gross realized-loss stop ₹%.0f. SL/TP exits depend on this bot remaining online.",
            NIFTY_LIVE_MAX_ENTRIES_PER_DAY,
            NIFTY_LIVE_MAX_DAILY_LOSS,
        )

    def _load_state(self) -> None:
        super()._load_state()
        today = datetime.now().date()
        dropped = 0
        for position in self.tracker.open_positions():
            if str((position.entry_metadata or {}).get("execution_mode", "")).upper() == "REAL":
                self._live_recovery_blocked = True
                log.critical(
                    "Restored live NIFTY position #%s; automation is blocked until the broker position "
                    "is manually reconciled",
                    position.trade_number,
                )
                continue
            if position.opened_at.date() < today:
                self.tracker.drop_position(position, reason="STARTUP_DROPPED")
                dropped += 1
                continue

            try:
                option_ltp = self._to_float(self._option_ltp_lookup(position))
            except Exception as exc:
                log.warning("Unable to validate restored NIFTY option LTP: %s", exc)
                option_ltp = None
            if option_ltp is None or not math.isfinite(option_ltp) or option_ltp <= 0:
                self.tracker.drop_position(position, reason="STARTUP_DROPPED")
                dropped += 1

        if dropped:
            self._persist_state()
            log.info("Dropped %d restored NIFTY paper position(s) at startup", dropped)

    def run_once(self):
        if self._live_recovery_blocked:
            print(
                "NIFTY automation is blocked: a previous live position was restored. "
                "Check Angel One Positions/Order Book and reconcile it manually."
            )
            return
        super().run_once()

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
        return dt_time(9, 0), dt_time(15, 30)

    def _state_file(self) -> str:
        return "data/nifty_daily_state.json"

    def _load_state(self) -> None:
        super()._load_state()
        today = datetime.now().date()
        dropped = 0
        for position in self.tracker.open_positions():
            if position.opened_at.date() < today:
                self.tracker.drop_position(position, reason="STARTUP_DROPPED")
                dropped += 1
                continue

            try:
                option_ltp = self._to_float(self._option_ltp_lookup(position))
            except Exception as exc:
                log.warning("Unable to validate restored NIFTY option LTP: %s", exc)
                option_ltp = None
            if option_ltp is None or not math.isfinite(option_ltp) or option_ltp <= 0:
                self.tracker.drop_position(position, reason="STARTUP_DROPPED")
                dropped += 1

        if dropped:
            self._persist_state()
            log.info("Dropped %d restored NIFTY position(s) at startup", dropped)

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

        mode = "REAL" if self._live_entry_enabled else "PAPER"
        print(f"Starting NIFTY {mode} trading session (interval={interval}s)")
        if self._live_activation_blocked_reason:
            print(f"Live trading is OFF: {self._live_activation_blocked_reason}")
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
