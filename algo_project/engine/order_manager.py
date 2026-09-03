"""Executes a `Decision` either in paper mode (simulated fill, fully
implemented) or live mode (real broker order - NOT implemented yet).

Live order placement needs a verified SmartAPI `placeOrder` payload
(exchange, symboltoken, producttype, ordertype, price, squareoff,
stoploss, duration, variety, etc.). Do not enable live mode until that
payload has been tested end-to-end against Angel One's SmartAPI docs.
"""

from enum import Enum

from engine.decision_engine import Decision
from engine.position_tracker import PositionTracker
from logger import get_logger

log = get_logger(__name__)


class ExecutionMode(str, Enum):
    PAPER = "PAPER"
    REAL = "REAL"
    LIVE = "REAL"  # Backward-compatible alias.


class OrderManager:
    """Routes an approved `Decision` to a paper fill or a live broker order."""

    def __init__(
        self,
        tracker: PositionTracker,
        mode: ExecutionMode = ExecutionMode.PAPER,
        broker_client=None,
    ):
        self.tracker = tracker
        self.mode = mode
        self.broker_client = broker_client  # Required for LIVE mode

        if self.mode == ExecutionMode.REAL and self.broker_client is None:
            raise ValueError("broker_client is required for real execution mode")

    def execute(self, decision: Decision, ltp: float, entry_metadata: dict | None = None):
        if decision.action not in ("BUY", "SELL"):
            log.info("No order placed: decision action=%s", decision.action)
            return None

        if self.mode == ExecutionMode.PAPER:
            return self._execute_paper(decision, ltp, entry_metadata)

        return self._execute_live(decision, ltp)

    def _execute_paper(self, decision: Decision, ltp: float, entry_metadata: dict | None = None):
        # Options buying strategy: Always BUY the contract (CE or PE)
        order_side = "BUY"

        position = self.tracker.open_position(
            symbol=decision.underlying,
            strike=decision.strike,
            option_type=decision.option_type,
            side=order_side,
            quantity=decision.quantity,
            entry_price=ltp,
            index_entry=decision.index_entry,
            index_sl=decision.index_sl,
            index_target=decision.index_target,
            entry_metadata=entry_metadata,
            trailing_stop=decision.index_sl,
        )
        log.info("[PAPER] Simulated fill: %s", position)
        return position

    def check_exits(self, symbol: str, current_index_price: float, ltp_lookup) -> list:
        """Check open positions for the specific symbol against index-based SL/Target
        and close any that have been hit using option premium lookup.

        Returns the list of positions closed on this call.
        """
        closed = []
        for position in list(self.tracker.open_positions()):
            # Symbol validation to prevent cross-symbol SL trigger
            pos_symbol = getattr(position, "symbol", getattr(position, "underlying", None))
            if pos_symbol and pos_symbol != symbol:
                continue

            reason = self.tracker.check_exit(position, current_index_price)
            if not reason:
                continue

            exit_price = ltp_lookup(position)
            if exit_price is None:
                log.warning(
                    "Cannot close position #%s (%s hit): no exit LTP available",
                    getattr(position, "trade_number", "N/A"),
                    reason,
                )
                continue

            self.tracker.close_position(position, exit_price, reason=reason)
            closed.append(position)

        return closed

    def _execute_live(self, decision: Decision, ltp: float):
        # Live order placement guard
        raise NotImplementedError(
            "Live order placement is not implemented yet. "
            "Verify the SmartAPI placeOrder payload before enabling live mode."
        )