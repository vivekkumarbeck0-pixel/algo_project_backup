from datetime import datetime
from tempfile import TemporaryDirectory
import unittest

from engine.daily_state import DailyStateStore
from engine.position_tracker import PositionTracker
from engine.risk_manager import RiskLimits, RiskManager


class DailyStateTests(unittest.TestCase):
    def test_daily_state_restores_trade_count_and_stop_loss_streak(self):
        with TemporaryDirectory() as temporary_directory:
            state_store = DailyStateStore(f"{temporary_directory}/daily_state.json")
            tracker = PositionTracker()

            for index in range(3):
                position = tracker.open_position("NIFTY", 25000 + index * 50, "CE", "BUY", 65, 100.0)
                tracker.close_position(position, 90.0, reason="SL_HIT")

            state_store.save(
                {
                    "date": datetime.now().date().isoformat(),
                    "positions": tracker.export_state(),
                    "trades_today": tracker.trades_today_count(),
                    "consecutive_stop_losses": tracker.consecutive_stop_losses(),
                    "daily_realized_pnl": tracker.daily_realized_pnl(),
                }
            )

            restored_payload = state_store.load(datetime.now().date().isoformat())
            restored_tracker = PositionTracker()
            restored_tracker.restore_state(restored_payload["positions"])
            # The consecutive stop-loss lock has been removed: a losing streak no longer blocks new trades.
            risk_manager = RiskManager(restored_tracker, RiskLimits())

            self.assertEqual(restored_tracker.trades_today_count(), 3)
            self.assertEqual(restored_tracker.consecutive_stop_losses(), 3)
            self.assertEqual(restored_tracker.daily_realized_pnl(), -1950.0)
            self.assertTrue(risk_manager.evaluate(65).approved)

    def test_daily_state_rolls_prior_day_forward(self):
        with TemporaryDirectory() as temporary_directory:
            state_store = DailyStateStore(f"{temporary_directory}/daily_state.json")
            state_store.save({"date": "2000-01-01", "positions": []})

            restored = state_store.load(datetime.now().date().isoformat())

            self.assertEqual(restored["date"], datetime.now().date().isoformat())
            self.assertEqual(restored["positions"], [])
            self.assertEqual(restored["daily_realized_pnl"], 0.0)

    def test_daily_state_rollover_preserves_open_position_and_resets_daily_book(self):
        with TemporaryDirectory() as temporary_directory:
            state_store = DailyStateStore(f"{temporary_directory}/daily_state.json")
            tracker = PositionTracker()
            position = tracker.open_position("NIFTY", 25000, "CE", "BUY", 65, 100.0)
            state_store.save(
                {
                    "date": "2000-01-01",
                    "positions": tracker.export_state(),
                    "trades_today": 1,
                    "consecutive_stop_losses": 1,
                    "daily_realized_pnl": -650.0,
                }
            )

            restored = state_store.load(datetime.now().date().isoformat())
            restored_tracker = PositionTracker()
            restored_tracker.restore_state(restored["positions"])

            self.assertEqual(restored["date"], datetime.now().date().isoformat())
            self.assertEqual(restored["trades_today"], 0)
            self.assertEqual(restored["daily_realized_pnl"], 0.0)
            self.assertEqual(len(restored_tracker.open_positions()), 1)
            self.assertEqual(restored_tracker.open_positions()[0].strike, position.strike)