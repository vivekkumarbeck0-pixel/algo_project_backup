"""Position tracker with serialization and persistence hooks."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

from logger import get_logger

log = get_logger(__name__)


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Position:
    symbol: str
    strike: float
    option_type: str  # "CE" or "PE"
    side: str  # "BUY" or "SELL"
    quantity: int
    entry_price: float
    trade_number: int = 0
    # Index levels at entry, used to trigger SL/Target
    index_entry: float | None = None
    index_sl: float | None = None
    index_target: float | None = None
    entry_metadata: dict = field(default_factory=dict)
    trailing_stop: float | None = None
    opened_at: datetime = field(default_factory=datetime.now)
    exit_price: float | None = None
    closed_at: datetime | None = None
    close_reason: str | None = None  # "SL_HIT", "TARGET_HIT", "MANUAL", "EOD"
    status: PositionStatus = PositionStatus.OPEN

    @property
    def pnl(self) -> float | None:
        if self.exit_price is None:
            return None
        direction = 1 if self.side == "BUY" else -1
        return direction * (self.exit_price - self.entry_price) * self.quantity


class PositionTracker:
    """Tracks open/closed positions and reports every state mutation."""

    def __init__(self, on_change: Callable[[], None] | None = None):
        self._positions: list[Position] = []
        self._on_change = on_change

    def _notify_change(self) -> None:
        if self._on_change:
            self._on_change()

    @staticmethod
    def _serialize_position(position: Position) -> dict:
        data = asdict(position)
        data["status"] = position.status.value
        data["opened_at"] = position.opened_at.isoformat()
        data["closed_at"] = position.closed_at.isoformat() if position.closed_at else None
        return data

    @staticmethod
    def _deserialize_position(data: dict) -> Position:
        values = dict(data)
        values["opened_at"] = datetime.fromisoformat(values["opened_at"])
        if values.get("closed_at"):
            values["closed_at"] = datetime.fromisoformat(values["closed_at"])
        values["status"] = PositionStatus(values.get("status", PositionStatus.OPEN.value))
        return Position(**values)

    def export_state(self) -> list[dict]:
        return [self._serialize_position(position) for position in self._positions]

    def restore_state(self, positions: list[dict]) -> None:
        self._positions = [self._deserialize_position(position) for position in positions]

    def open_position(
        self,
        symbol,
        strike,
        option_type,
        side,
        quantity,
        entry_price,
        index_entry=None,
        index_sl=None,
        index_target=None,
        entry_metadata=None,
        trailing_stop=None,
    ) -> Position:
        position = Position(
            symbol=symbol,
            strike=strike,
            option_type=option_type,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            trade_number=self.trades_today_count() + 1,
            index_entry=index_entry,
            index_sl=index_sl,
            index_target=index_target,
            entry_metadata=dict(entry_metadata or {}),
            trailing_stop=trailing_stop,
        )
        self._positions.append(position)
        log.info(
            "Opened position #%d: %s %s %s strike=%s qty=%s @ %s (index_entry=%s sl=%s target=%s)",
            position.trade_number, side, symbol, option_type, strike, quantity, entry_price,
            index_entry, index_sl, index_target,
        )
        self._notify_change()
        return position

    def close_position(self, position: Position, exit_price: float, reason: str = "MANUAL") -> Position:
        position.exit_price = exit_price
        position.closed_at = datetime.now()
        position.close_reason = reason
        position.status = PositionStatus.CLOSED
        log.info(
            "Closed position #%d: %s %s pnl=%s reason=%s",
            position.trade_number, position.symbol, position.option_type, position.pnl, reason,
        )
        self._notify_change()
        return position

    def check_exit(self, position: Position, current_index_price: float) -> str | None:
        """Return 'SL_HIT'/'TARGET_HIT' based on option contract type (CE vs PE)
        and live index price level, else None."""
        if current_index_price is None or position.index_sl is None or position.index_target is None:
            return None

        # Call Option (CE) SL/Target logic
        if position.option_type == "CE":
            if current_index_price <= position.index_sl:
                return "SL_HIT"
            if current_index_price >= position.index_target:
                return "TARGET_HIT"

        # Put Option (PE) SL/Target logic
        elif position.option_type == "PE":
            if current_index_price >= position.index_sl:
                return "SL_HIT"
            if current_index_price <= position.index_target:
                return "TARGET_HIT"

        return None

    def open_positions(self) -> list[Position]:
        return [p for p in self._positions if p.status == PositionStatus.OPEN]

    def get_open_positions_count(self) -> int:
        """Returns active open positions count (used by RiskManager)."""
        return len(self.open_positions())

    def closed_positions(self) -> list[Position]:
        return [p for p in self._positions if p.status == PositionStatus.CLOSED]

    def total_realized_pnl(self) -> float:
        return sum(p.pnl or 0 for p in self.closed_positions())

    # ============================================================
    # DAILY TRADE LIMIT / SUMMARY HELPERS
    # ============================================================

    def trades_today(self) -> list[Position]:
        today = datetime.now().date()
        return [p for p in self._positions if p.opened_at.date() == today]

    def trades_today_count(self) -> int:
        return len(self.trades_today())

    def closed_today(self) -> list[Position]:
        today = datetime.now().date()
        return [p for p in self.closed_positions() if p.closed_at and p.closed_at.date() == today]

    def daily_realized_pnl(self) -> float:
        return sum(p.pnl or 0 for p in self.closed_today())

    def consecutive_stop_losses(self) -> int:
        losses = 0
        for position in sorted(self.closed_today(), key=lambda item: item.closed_at or item.opened_at, reverse=True):
            if position.close_reason != "SL_HIT":
                break
            losses += 1
        return losses