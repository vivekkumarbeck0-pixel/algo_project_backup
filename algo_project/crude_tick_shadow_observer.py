"""Read-only, background PPO observer for live Crude option ticks."""

from __future__ import annotations

import math
import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

IST = ZoneInfo("Asia/Kolkata")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "crude_tick_ppo_80_20.zip"
WORKSHEET_TITLE = "PPO_Shadow_Logs"
LOG_HEADERS = (
    "Timestamp",
    "Symbol",
    "Option Type",
    "Live Option Price (LTP)",
    "Current Baseline SL",
    "Current Baseline TP",
    "PPO Suggested SL",
    "PPO Suggested TP",
    "PPO Adjustment Action",
    "Hypothetical PnL Points",
    "Exit Reason",
)
logger = logging.getLogger("crude_option_buyer")


@dataclass(frozen=True)
class TickSnapshot:
    timestamp: datetime
    tick_key: float
    trade_id: str
    symbol: str
    option_type: str
    live_price: float
    entry_price: float
    atr: float
    baseline_sl: float
    baseline_tp: float


def _as_ist(value: datetime) -> datetime:
    return value.replace(tzinfo=IST) if value.tzinfo is None else value.astimezone(IST)


class CrudeTickShadowObserver:
    """Infer and append shadow-only decisions without touching broker state."""

    def __init__(self, market_start: str = "09:00", market_close: str = "23:30"):
        self._market_start = self._parse_time(market_start)
        self._market_close = self._parse_time(market_close)
        self._snapshots: queue.Queue[TickSnapshot] = queue.Queue(maxsize=512)
        self._model: Any = None
        self._thread: threading.Thread | None = None
        self._states: dict[str, dict[str, Any]] = {}
        self._worksheet: Any = None

    @staticmethod
    def _parse_time(value: str) -> dt_time:
        hour, minute = (int(part) for part in value.split(":"))
        return dt_time(hour, minute)

    def _market_is_open(self, timestamp: datetime) -> bool:
        local_time = _as_ist(timestamp)
        if local_time.weekday() >= 5:
            return False
        current = local_time.time().replace(tzinfo=None)
        if self._market_start <= self._market_close:
            return self._market_start <= current < self._market_close
        return current >= self._market_start or current < self._market_close

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="crude-tick-ppo-shadow", daemon=True
        )
        self._thread.start()

    def submit_tick(
        self,
        *,
        timestamp: datetime,
        tick_key: float,
        trade_id: str,
        symbol: str,
        option_type: str,
        live_price: float,
        entry_price: float,
        atr: float,
        baseline_sl: float,
        baseline_tp: float,
    ) -> bool:
        """Queue a tick without waiting; return False if closed or overloaded."""
        try:
            if not self._market_is_open(timestamp):
                return False
            snapshot = TickSnapshot(
                timestamp=_as_ist(timestamp),
                tick_key=float(tick_key),
                trade_id=str(trade_id),
                symbol=str(symbol),
                option_type=str(option_type),
                live_price=float(live_price),
                entry_price=float(entry_price),
                atr=float(atr),
                baseline_sl=float(baseline_sl),
                baseline_tp=float(baseline_tp),
            )
            self._snapshots.put_nowait(snapshot)
            return True
        except Exception:
            return False

    def _load_model(self) -> Any:
        if not MODEL_PATH.is_file():
            logger.warning(
                "Tick PPO shadow observer disabled: model file is missing at %s.",
                MODEL_PATH,
            )
            return None
        try:
            from stable_baselines3 import PPO

            model = PPO.load(str(MODEL_PATH))
        except Exception:
            logger.exception(
                "Tick PPO shadow observer disabled: could not load model %s.",
                MODEL_PATH,
            )
            return None
        logger.info("Tick PPO shadow observer loaded model from %s.", MODEL_PATH)
        return model

    def _run(self) -> None:
        try:
            self._model = self._load_model()
            if self._model is None:
                return
        except Exception:
            logger.exception("Tick PPO shadow observer stopped during model startup.")
            return

        while True:
            try:
                snapshot = self._snapshots.get()
            except Exception:
                logger.exception("Tick PPO shadow observer could not read its tick queue.")
                continue
            try:
                now = datetime.now(IST)
                if not self._market_is_open(now):
                    continue
                row = self._predict_snapshot(snapshot, now=now)
                if row is not None:
                    self._append_shadow_row(row)
            except Exception:
                logger.exception(
                    "Tick PPO shadow observer failed for trade %s.",
                    snapshot.trade_id,
                )
                continue

    def _predict_snapshot(
        self, snapshot: TickSnapshot, *, now: datetime | None = None
    ) -> dict[str, Any] | None:
        now = _as_ist(now or datetime.now(IST))
        if (
            self._model is None
            or not self._market_is_open(snapshot.timestamp)
            or not self._market_is_open(now)
        ):
            return None
        values = (
            snapshot.live_price,
            snapshot.entry_price,
            snapshot.atr,
            snapshot.baseline_sl,
            snapshot.baseline_tp,
        )
        if not all(math.isfinite(value) for value in values) or (
            snapshot.live_price <= 0 or snapshot.entry_price <= 0 or snapshot.atr <= 0
        ):
            return None

        state = self._states.get(snapshot.trade_id)
        if state is None:
            state = {
                "last_tick_key": None,
                "sl_atr_multiplier": 1.0,
                "tp_atr_multiplier": 2.0,
                "sl_buffer_atr": 0.1,
                "sl_price": snapshot.entry_price - 1.1 * snapshot.atr,
                "tp_price": snapshot.entry_price + 2.0 * snapshot.atr,
                "max_price": snapshot.entry_price,
                "samples_elapsed": 0,
                "finished": False,
            }
            self._states[snapshot.trade_id] = state
        if state["finished"] or state["last_tick_key"] == snapshot.tick_key:
            return None

        state["max_price"] = max(state["max_price"], snapshot.live_price)
        observation = np.asarray(
            [
                snapshot.live_price / snapshot.entry_price - 1.0,
                (snapshot.live_price - snapshot.entry_price) / snapshot.atr,
                snapshot.atr / snapshot.entry_price,
                (snapshot.live_price - state["sl_price"]) / snapshot.atr,
                (state["tp_price"] - snapshot.live_price) / snapshot.atr,
                state["sl_atr_multiplier"],
                state["tp_atr_multiplier"],
                state["sl_buffer_atr"],
                (state["max_price"] - snapshot.entry_price) / snapshot.atr,
                state["samples_elapsed"] / 5000.0,
            ],
            dtype=np.float32,
        )
        predicted, _ = self._model.predict(observation, deterministic=True)
        action = np.asarray(predicted, dtype=np.float32).reshape(-1)
        if (
            action.size != 3
            or not np.isfinite(action).all()
            or (action < 0.0).any()
            or (action > 1.0).any()
        ):
            return None

        sl_multiplier = 0.25 + float(action[0]) * (4.0 - 0.25)
        tp_multiplier = 0.25 + float(action[1]) * (8.0 - 0.25)
        sl_buffer = float(action[2]) * 2.0
        state["sl_price"] = max(
            state["sl_price"],
            snapshot.entry_price - (sl_multiplier + sl_buffer) * snapshot.atr,
        )
        state["tp_price"] = snapshot.entry_price + tp_multiplier * snapshot.atr
        state["sl_atr_multiplier"] = sl_multiplier
        state["tp_atr_multiplier"] = tp_multiplier
        state["sl_buffer_atr"] = sl_buffer
        state["last_tick_key"] = snapshot.tick_key
        state["samples_elapsed"] += 1

        exit_reason = ""
        if snapshot.live_price <= state["sl_price"]:
            exit_reason = "PPO_STOP_LOSS"
        elif snapshot.live_price >= state["tp_price"]:
            exit_reason = "PPO_TAKE_PROFIT"
        if exit_reason:
            state["finished"] = True

        return {
            "Timestamp": snapshot.timestamp.isoformat(),
            "Symbol": snapshot.symbol,
            "Option Type": snapshot.option_type,
            "Live Option Price (LTP)": snapshot.live_price,
            "Current Baseline SL": snapshot.baseline_sl,
            "Current Baseline TP": snapshot.baseline_tp,
            "PPO Suggested SL": state["sl_price"],
            "PPO Suggested TP": state["tp_price"],
            "PPO Adjustment Action": (
                f"SL={sl_multiplier:.4f} ATR; TP={tp_multiplier:.4f} ATR; "
                f"buffer={sl_buffer:.4f} ATR"
            ),
            "Hypothetical PnL Points": snapshot.live_price - snapshot.entry_price,
            "Exit Reason": exit_reason,
        }

    def _append_shadow_row(self, row: dict[str, Any]) -> None:
        import gspread

        from sheets_logger import get_gspread_client

        if self._worksheet is None:
            spreadsheet = get_gspread_client().open("Crude_Algo_Trade_Logs")
            try:
                self._worksheet = spreadsheet.worksheet(WORKSHEET_TITLE)
            except gspread.exceptions.WorksheetNotFound:
                self._worksheet = spreadsheet.add_worksheet(
                    title=WORKSHEET_TITLE, rows=1000, cols=len(LOG_HEADERS)
                )
            if not self._worksheet.get_all_values():
                self._worksheet.append_row(list(LOG_HEADERS))

        self._worksheet.append_row([row.get(header, "") for header in LOG_HEADERS])
