"""Pre-trade risk checks.

RiskManager approves or rejects a trading decision *before* an order
is sent to the broker. Extend `evaluate()` with additional rules
(margin, volatility, news blackout, etc.) as the strategy matures.
"""

from dataclasses import dataclass, field

from config import settings
from engine.position_tracker import PositionTracker
from logger import get_logger

log = get_logger(__name__)


@dataclass
class RiskLimits:
    max_open_positions: int = 1
    # Generic sanity ceiling - actual trade quantity is sized from contract lot size
    max_quantity_per_trade: int = 500
    max_daily_loss: float = 5000.0
    max_trades_per_day: int = 10
    max_consecutive_stop_losses: int = 3

    def __post_init__(self):
        # Fetch limits safely from settings module with fallback defaults
        if hasattr(settings, "max_trades_per_day"):
            self.max_trades_per_day = getattr(settings, "max_trades_per_day", self.max_trades_per_day)
        if hasattr(settings, "max_daily_loss"):
            self.max_daily_loss = getattr(settings, "max_daily_loss", self.max_daily_loss)
        if hasattr(settings, "max_open_positions"):
            self.max_open_positions = getattr(settings, "max_open_positions", self.max_open_positions)
        if hasattr(settings, "max_consecutive_stop_losses"):
            self.max_consecutive_stop_losses = getattr(
                settings, "max_consecutive_stop_losses", self.max_consecutive_stop_losses
            )


@dataclass
class RiskCheckResult:
    approved: bool
    reasons: list[str]


class RiskManager:
    """Approves/rejects trades against configured risk limits."""

    def __init__(self, tracker: PositionTracker, limits: RiskLimits | None = None):
        self.tracker = tracker
        self.limits = limits or RiskLimits()

    def evaluate(self, quantity: int) -> RiskCheckResult:
        reasons = []

        # 1. Open Positions Check
        active_positions_count = (
            self.tracker.get_open_positions_count()
            if hasattr(self.tracker, "get_open_positions_count")
            else len(self.tracker.open_positions())
        )
        if active_positions_count >= self.limits.max_open_positions:
            reasons.append(
                f"Max open positions reached ({active_positions_count}/{self.limits.max_open_positions})"
            )

        # 2. Daily Max Trade Count Check
        today_trades_count = self.tracker.trades_today_count()
        if self.limits.max_trades_per_day is not None and today_trades_count >= self.limits.max_trades_per_day:
            reasons.append(
                f"Daily trade limit reached ({today_trades_count}/{self.limits.max_trades_per_day} trades/day)"
            )

        # 3. Order Quantity Sanity Check
        if quantity > self.limits.max_quantity_per_trade:
            reasons.append(
                f"Quantity {quantity} exceeds max per trade limit ({self.limits.max_quantity_per_trade})"
            )

        # 4. Daily Drawdown / Loss Limit Check
        daily_pnl = self.tracker.daily_realized_pnl()
        if daily_pnl <= -abs(self.limits.max_daily_loss):
            reasons.append(
                f"Daily loss limit breached (Current Realized P&L: {daily_pnl} <= Limit: -{abs(self.limits.max_daily_loss)})"
            )

        # 5. Consecutive stop-loss lock for the remainder of the session.
        consecutive_stop_losses = self.tracker.consecutive_stop_losses()
        if (
            self.limits.max_consecutive_stop_losses is not None
            and consecutive_stop_losses >= self.limits.max_consecutive_stop_losses
        ):
            reasons.append(
                "Trading locked after "
                f"{consecutive_stop_losses} consecutive stop-loss exits today"
            )

        approved = not reasons
        if not approved:
            log.warning("Risk check REJECTED trade: %s", "; ".join(reasons))
        else:
            log.info("Risk check APPROVED trade for quantity=%s", quantity)

        return RiskCheckResult(approved=approved, reasons=reasons)