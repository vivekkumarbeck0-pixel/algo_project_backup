"""Routes approved decisions to paper fills or confirmed Angel One orders."""

import math
import time
from enum import Enum

from engine.decision_engine import Decision
from engine.position_tracker import Position, PositionTracker
from logger import get_logger

log = get_logger(__name__)

ORDER_CONFIRMATION_ATTEMPTS = 10
ORDER_CONFIRMATION_DELAY_SECONDS = 0.5
FILLED_STATUSES = {"COMPLETE", "COMPLETED", "FILLED", "TRADED"}
REJECTED_STATUSES = {"REJECTED", "CANCELLED", "CANCELED"}
PENDING_STATUSES = {
    "OPEN", "PENDING", "TRIGGER PENDING", "MODIFY PENDING",
    "AMO REQ RECEIVED", "VALIDATION PENDING",
}


class ExecutionMode(str, Enum):
    PAPER = "PAPER"
    REAL = "REAL"
    LIVE = "REAL"  # Backward-compatible alias.


class LiveOrderError(RuntimeError):
    """A live order was rejected, unresolved, or needs manual reconciliation."""


class OrderManager:
    """Routes approved NIFTY decisions to simulated or broker-confirmed fills."""

    def __init__(
        self,
        tracker: PositionTracker,
        mode: ExecutionMode = ExecutionMode.PAPER,
        broker_client=None,
    ):
        self.tracker = tracker
        self.mode = ExecutionMode(mode)
        self.broker_client = broker_client
        self._entry_submission_blocked = False

        if self.mode == ExecutionMode.REAL and self.broker_client is None:
            raise ValueError("broker_client is required for real execution mode")

    @property
    def entry_submission_blocked(self) -> bool:
        return self._entry_submission_blocked

    def execute(self, decision: Decision, ltp: float, entry_metadata: dict | None = None):
        if decision.action not in ("BUY", "SELL"):
            log.info("No order placed: decision action=%s", decision.action)
            return None

        if self.mode == ExecutionMode.PAPER:
            return self._execute_paper(decision, ltp, entry_metadata)

        return self._execute_live(decision, ltp, entry_metadata)

    def _execute_paper(self, decision: Decision, ltp: float, entry_metadata: dict | None = None):
        metadata = dict(entry_metadata or {})
        metadata["execution_mode"] = ExecutionMode.PAPER.value
        position = self.tracker.open_position(
            symbol=decision.underlying,
            strike=decision.strike,
            option_type=decision.option_type,
            side="BUY",
            quantity=decision.quantity,
            entry_price=ltp,
            index_entry=decision.index_entry,
            index_sl=decision.index_sl,
            index_target=decision.index_target,
            entry_metadata=metadata,
            trailing_stop=decision.index_sl,
            entry_iv=metadata.get("iv"),
            entry_vix=metadata.get("vix"),
        )
        log.info("[PAPER] Simulated fill: %s", position)
        return position

    def has_existing_nifty_activity(self) -> bool:
        """Fail closed if Angel One reports a NIFTY position or pending NFO order."""
        api = self._api()
        positions = self._account_rows(api, "position")
        orders = self._account_rows(api, "orderBook")

        for position in positions:
            if not self._is_nifty_instrument(position):
                continue
            try:
                net_quantity = int(float(position.get("netqty", position.get("netQty", 0)) or 0))
            except (TypeError, ValueError) as exc:
                raise LiveOrderError("Could not parse an Angel One NIFTY position quantity") from exc
            if net_quantity:
                log.error("Live trading blocked: an existing NIFTY broker position was found.")
                return True

        for order in orders:
            if (
                self._is_nifty_instrument(order)
                and str(order.get("orderstatus", order.get("status", ""))).strip().upper()
                in PENDING_STATUSES
            ):
                log.error("Live trading blocked: an existing pending NIFTY broker order was found.")
                return True
        return False

    def check_exits(self, symbol: str, current_index_price: float, ltp_lookup) -> list:
        """Check index-based exits and close positions only after confirmed fills."""
        closed = []
        for position in list(self.tracker.open_positions()):
            pos_symbol = getattr(position, "symbol", getattr(position, "underlying", None))
            if pos_symbol and pos_symbol != symbol:
                continue

            reason = self.tracker.check_exit(position, current_index_price)
            if not reason:
                continue

            exit_ltp = ltp_lookup(position)
            if exit_ltp is None and not self._is_live_position(position):
                log.warning(
                    "Cannot close position #%s (%s hit): no exit LTP available",
                    getattr(position, "trade_number", "N/A"),
                    reason,
                )
                continue

            closed_position = self.close_position(position, exit_ltp, reason)
            if closed_position is not None:
                closed.append(closed_position)
        return closed

    def close_position(
        self, position: Position, ltp: float | None, reason: str = "MANUAL"
    ) -> Position | None:
        """Close paper positions at LTP; close live positions at confirmed broker fill."""
        if not self._is_live_position(position):
            if ltp is None:
                log.warning("Cannot close paper position #%s without an exit LTP", position.trade_number)
                return None
            return self.tracker.close_position(position, ltp, reason=reason)

        metadata = position.entry_metadata
        if metadata.get("live_order_manual_review"):
            log.error(
                "Live exit blocked for #%s: broker state requires manual reconciliation",
                position.trade_number,
            )
            return None

        try:
            order_id = metadata.get("pending_live_exit_order_id")
            if order_id:
                fill = self._wait_for_fill(
                    str(order_id), position.quantity, position.entry_metadata["option_instrument"]
                )
            else:
                contract = metadata.get("option_instrument")
                order_id = self._place_market_order(
                    contract=contract,
                    transaction_type="SELL",
                    quantity=position.quantity,
                )
                metadata["pending_live_exit_order_id"] = order_id
                self.tracker._notify_change()
                fill = self._wait_for_fill(order_id, position.quantity, contract)
        except LiveOrderError as exc:
            metadata["live_order_manual_review"] = True
            self.tracker._notify_change()
            log.error("Live exit for #%s not confirmed: %s", position.trade_number, exc)
            return None

        metadata.pop("pending_live_exit_order_id", None)
        self.tracker._notify_change()
        return self.tracker.close_position(position, fill, reason=reason)

    def _execute_live(self, decision: Decision, ltp: float, entry_metadata: dict | None = None):
        if decision.underlying != "NIFTY":
            raise LiveOrderError("Real execution is enabled only for NIFTY options")
        if self._entry_submission_blocked:
            raise LiveOrderError(
                "NIFTY live entries are blocked after an unconfirmed broker order; "
                "reconcile the Angel One order book before restarting"
            )

        metadata = dict(entry_metadata or {})
        metadata["execution_mode"] = ExecutionMode.REAL.value
        contract = metadata.get("option_instrument")
        if not isinstance(contract, dict):
            raise LiveOrderError("NIFTY option contract metadata is missing")
        quantity = self._positive_integer(decision.quantity, "order quantity")

        try:
            order_id = self._place_market_order(contract, "BUY", quantity)
            fill = self._wait_for_fill(order_id, quantity, contract)
        except LiveOrderError as exc:
            if getattr(exc, "order_id", None):
                self._entry_submission_blocked = True
                log.critical(
                    "NIFTY entry order %s needs broker reconciliation; no further live entries will be sent",
                    exc.order_id,
                )
            raise

        metadata["live_entry_order_id"] = order_id
        return self.tracker.open_position(
            symbol=decision.underlying,
            strike=decision.strike,
            option_type=decision.option_type,
            side="BUY",
            quantity=quantity,
            entry_price=fill,
            index_entry=decision.index_entry,
            index_sl=decision.index_sl,
            index_target=decision.index_target,
            entry_metadata=metadata,
            trailing_stop=decision.index_sl,
            entry_iv=metadata.get("iv"),
            entry_vix=metadata.get("vix"),
        )

    def _api(self):
        if self.broker_client is None:
            raise LiveOrderError("Angel One broker session is unavailable")
        api = self.broker_client.get_api()
        if api is None:
            raise LiveOrderError("Angel One broker API is unavailable")
        return api

    def _account_rows(self, api, method_name: str) -> list[dict]:
        method = getattr(api, method_name, None)
        if not callable(method):
            raise LiveOrderError(f"Angel One API does not provide {method_name}()")
        try:
            response = method()
        except Exception as exc:
            raise LiveOrderError(f"Angel One {method_name}() request failed") from exc
        if not isinstance(response, dict) or response.get("status") is False:
            raise LiveOrderError(f"Angel One {method_name}() returned an invalid response")
        rows = response.get("data")
        if rows is None:
            return []
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise LiveOrderError(f"Angel One {method_name}() returned an invalid data shape")
        return rows

    @staticmethod
    def _is_nifty_instrument(record: dict) -> bool:
        if str(record.get("exchange", "")).strip().upper() != "NFO":
            return False
        symbol_name = str(record.get("symbolname", record.get("name", ""))).strip().upper()
        trading_symbol = str(record.get("tradingsymbol", "")).strip().upper()
        if symbol_name:
            return symbol_name == "NIFTY"
        return trading_symbol.startswith("NIFTY") and not trading_symbol.startswith("NIFTYNXT50")

    def _place_market_order(self, contract: dict | None, transaction_type: str, quantity: int) -> str:
        if not isinstance(contract, dict):
            raise LiveOrderError("NIFTY option contract metadata is missing")
        symbol = str(contract.get("symbol") or "").strip()
        token = str(contract.get("token") or "").strip()
        exchange = str(contract.get("exch_seg") or "").strip().upper()
        if not symbol or not token or exchange != "NFO":
            raise LiveOrderError("NIFTY option contract must include a valid NFO symbol and token")

        params = {
            "variety": "NORMAL",
            "tradingsymbol": symbol,
            "symboltoken": token,
            "transactiontype": transaction_type,
            "exchange": "NFO",
            "ordertype": "MARKET",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": "0",
            "squareoff": "0",
            "stoploss": "0",
            "quantity": str(quantity),
        }
        try:
            order_id = self._api().placeOrder(params)
        except Exception as exc:
            self._entry_submission_blocked = transaction_type == "BUY"
            raise LiveOrderError(
                f"Angel One {transaction_type} order submission failed; check broker order book"
            ) from exc

        if isinstance(order_id, dict):
            order_id = (order_id.get("data") or {}).get("orderid")
        if not order_id:
            self._entry_submission_blocked = transaction_type == "BUY"
            error = LiveOrderError(
                f"Angel One returned no {transaction_type} order id; check broker order book"
            )
            error.order_id = "UNKNOWN"
            raise error
        return str(order_id)

    def _wait_for_fill(self, order_id: str, quantity: int, contract: dict) -> float:
        try:
            api = self._api()
            for attempt in range(ORDER_CONFIRMATION_ATTEMPTS):
                rows = self._account_rows(api, "orderBook")
                order = next(
                    (
                        row for row in rows
                        if str(row.get("orderid", row.get("orderId", ""))) == str(order_id)
                    ),
                    None,
                )
                if order is not None:
                    status = str(order.get("orderstatus", order.get("status", ""))).strip().upper()
                    filled = self._filled_quantity(order)
                    if status in FILLED_STATUSES:
                        if filled and filled != quantity:
                            raise LiveOrderError(
                                f"Order {order_id} was only partially filled; manual reconciliation required"
                            )
                        average_price = self._positive_price(
                            order.get("averageprice", order.get("averagePrice"))
                        )
                        if average_price is None:
                            raise LiveOrderError(
                                f"Order {order_id} is filled but has no valid average fill price"
                            )
                        return average_price
                    if status in REJECTED_STATUSES:
                        if filled:
                            raise LiveOrderError(
                                f"Order {order_id} was partially filled then rejected; "
                                "manual reconciliation required"
                            )
                        raise LiveOrderError(f"Angel One rejected order {order_id}")
                    if status and status not in PENDING_STATUSES:
                        raise LiveOrderError(
                            f"Order {order_id} has unrecognized status {status!r}; manual reconciliation required"
                        )
                if attempt + 1 < ORDER_CONFIRMATION_ATTEMPTS:
                    time.sleep(ORDER_CONFIRMATION_DELAY_SECONDS)
        except LiveOrderError as exc:
            if getattr(exc, "order_id", None) is None:
                exc.order_id = order_id
            raise
        except Exception as exc:
            error = LiveOrderError(
                f"Could not confirm broker order {order_id}; check Angel One order book"
            )
            error.order_id = order_id
            raise error from exc

        error = LiveOrderError(
            f"Order {order_id} was not confirmed filled; check Angel One order book"
        )
        error.order_id = order_id
        raise error

    @staticmethod
    def _filled_quantity(order: dict) -> int:
        try:
            return int(float(order.get("filledshares", order.get("filledShares", 0)) or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _positive_integer(value, field: str) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise LiveOrderError(f"{field} must be a positive integer") from exc
        if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
            raise LiveOrderError(f"{field} must be a positive integer")
        return int(numeric)

    @staticmethod
    def _positive_price(value) -> float | None:
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        return price if math.isfinite(price) and price > 0 else None

    @staticmethod
    def _is_live_position(position: Position) -> bool:
        return str((position.entry_metadata or {}).get("execution_mode", "")).upper() == "REAL"
