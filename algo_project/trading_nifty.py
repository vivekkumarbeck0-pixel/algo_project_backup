"""Standalone NIFTY trading session.

Run with: ``python trading_nifty.py``
"""

from datetime import time as dt_time

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


if __name__ == "__main__":
    NiftyTradingSession().run_forever(settings.live_poll_interval_seconds)
