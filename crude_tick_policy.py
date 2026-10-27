"""Shared tick-level trading logic for the independent crude PPO agent.

The same ``TickFeatureBuilder`` and ``TradeEngine`` drive both training
(``CrudeTradingEnv``) and paper trading (``crude_ppo_paper_runner``), so the
agent sees identical observations and fills in both places.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

MIN_REWARD_RISK_RATIO = 1.0
TRAILING_GAP_R_BOUNDS = (0.5, 1.0)

FLAT, LONG, SHORT = 0, 1, -1
SIDE_NAMES = {FLAT: "FLAT", LONG: "LONG", SHORT: "SHORT"}
SL_ATR_BOUNDS = (1.0, 3.0)  # >= 1 ATR so slippage/noise cannot instantly stop a trade out
MIN_ENTRY_TARGET_RR = 1.5  # every new trade must aim for at least this RR
TP_RR_BOUNDS = (MIN_ENTRY_TARGET_RR, 4.0)
# Minimum signed momentum (ATR units) for 5- and 20-tick entry confirmation.
MIN_ENTRY_MOMENTUM_ATR = (0.1, 0.2)
# Mirrors trading_crude.py: the trail arms after a fixed premium gain and the
# stop jumps to entry + buffer the moment it does.
TRAIL_ACTIVATION_POINTS = 12.0
TRAIL_BREAKEVEN_BUFFER_POINTS = 1.0
MOMENTUM_WINDOWS = (5, 20, 100)
VOLATILITY_WINDOWS = (20, 100)
_EMA_FAST, _EMA_SLOW = 20, 100

FEATURE_NAMES = (
    *(f"momentum_{w}_atr" for w in MOMENTUM_WINDOWS),
    *(f"volatility_{w}_atr" for w in VOLATILITY_WINDOWS),
    "distance_to_window_high_atr",
    "distance_to_window_low_atr",
    "distance_to_ema_fast_atr",
    "distance_to_ema_slow_atr",
    "atr_fraction",
    "position_flat",
    "position_long",
    "position_short",
    "unrealized_r",
    "stop_gap_atr",
    "target_gap_atr",
    "max_favorable_r",
    "hold_ticks_scaled",
    "target_rr",
)
OBSERVATION_SIZE = len(FEATURE_NAMES)


def _lerp(value: float, bounds: tuple[float, float]) -> float:
    return bounds[0] + float(np.clip(value, 0.0, 1.0)) * (bounds[1] - bounds[0])


def map_action(
    action: Sequence[float],
    *,
    sl_atr_bounds: tuple[float, float] = SL_ATR_BOUNDS,
    tp_rr_bounds: tuple[float, float] = TP_RR_BOUNDS,
    trailing_gap_r_bounds: tuple[float, float] = TRAILING_GAP_R_BOUNDS,
) -> tuple[int, float, float, float]:
    """Map a 4-value action in [0, 1] to (intent, SL ATR mult, target RR, trail gap R).

    Intent is SHORT below 1/3, LONG above 2/3, otherwise hold/stay flat.
    """
    a = np.asarray(action, dtype=float).reshape(-1)
    if a.size != 4 or not np.all(np.isfinite(a)):
        raise ValueError("action must contain four finite values")
    intent = SHORT if a[0] < 1 / 3 else LONG if a[0] > 2 / 3 else FLAT
    return (
        intent,
        _lerp(a[1], sl_atr_bounds),
        _lerp(a[2], tp_rr_bounds),
        _lerp(a[3], trailing_gap_r_bounds),
    )


@dataclass(frozen=True)
class EngineConfig:
    slippage_points: float = 0.0  # adverse slippage applied to every fill
    cost_points: float = 0.0  # round-trip fees, deducted at exit
    min_risk_points: float = 1e-6
    # Tunable action ranges and entry gate (defaults are the module constants).
    sl_atr_bounds: tuple[float, float] = SL_ATR_BOUNDS
    tp_rr_bounds: tuple[float, float] = TP_RR_BOUNDS
    trailing_gap_r_bounds: tuple[float, float] = TRAILING_GAP_R_BOUNDS
    min_entry_momentum_atr: tuple[float, ...] = MIN_ENTRY_MOMENTUM_ATR
    trail_activation_points: float = TRAIL_ACTIVATION_POINTS
    trail_breakeven_buffer_points: float = TRAIL_BREAKEVEN_BUFFER_POINTS

    def __post_init__(self) -> None:
        for name in (
            "slippage_points",
            "cost_points",
            "min_risk_points",
            "trail_activation_points",
            "trail_breakeven_buffer_points",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("sl_atr_bounds", "tp_rr_bounds", "trailing_gap_r_bounds"):
            bounds = getattr(self, name)
            if (
                len(bounds) != 2
                or not all(math.isfinite(b) for b in bounds)
                or bounds[0] <= 0
                or bounds[1] < bounds[0]
            ):
                raise ValueError(f"{name} must be two positive, ordered, finite values")
        if len(self.min_entry_momentum_atr) > len(MOMENTUM_WINDOWS) or not all(
            math.isfinite(m) for m in self.min_entry_momentum_atr
        ):
            raise ValueError("min_entry_momentum_atr needs finite values, one per leading window")


class TickFeatureBuilder:
    """Rolling multi-timeframe tick features, expressed in ATR units."""

    def __init__(self) -> None:
        self._prices: deque[float] = deque(maxlen=max(MOMENTUM_WINDOWS + VOLATILITY_WINDOWS) + 1)
        self._ema_fast: float | None = None
        self._ema_slow: float | None = None

    def reset(self) -> None:
        self._prices.clear()
        self._ema_fast = self._ema_slow = None

    def update(self, price: float) -> None:
        price = float(price)
        self._prices.append(price)
        if self._ema_fast is None:
            self._ema_fast = self._ema_slow = price
            return
        self._ema_fast += (price - self._ema_fast) * 2.0 / (_EMA_FAST + 1)
        self._ema_slow += (price - self._ema_slow) * 2.0 / (_EMA_SLOW + 1)

    def momentum_atr(self, atr: float) -> tuple[float, ...]:
        """Signed momentum per MOMENTUM_WINDOWS in ATR units."""
        atr = max(float(atr), 1e-9)
        prices = np.fromiter(self._prices, dtype=float)
        return tuple(
            float((prices[-1] - prices[max(0, len(prices) - 1 - w)]) / atr)
            for w in MOMENTUM_WINDOWS
        )

    def observation(self, atr: float, engine: "TradeEngine") -> np.ndarray:
        if not self._prices:
            raise RuntimeError("update() must be called before observation()")
        atr = max(float(atr), 1e-9)
        prices = np.fromiter(self._prices, dtype=float)
        price = prices[-1]
        features: list[float] = []
        for window in MOMENTUM_WINDOWS:
            features.append((price - prices[max(0, len(prices) - 1 - window)]) / atr)
        for window in VOLATILITY_WINDOWS:
            diffs = np.diff(prices[-(window + 1):])
            features.append(float(np.std(diffs)) / atr if diffs.size >= 2 else 0.0)
        features += [
            (price - prices.max()) / atr,
            (price - prices.min()) / atr,
            (price - self._ema_fast) / atr,
            (price - self._ema_slow) / atr,
            atr / max(price, 1e-9),
        ]
        features += engine.state_features(price, atr)
        return np.clip(
            np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0), -1e4, 1e4
        )


class TradeEngine:
    """Tick-by-tick paper position with adaptive SL, TP and trailing stop.

    The agent may enter either side, exit on demand, tighten the stop, extend
    the target and tune the trailing gap. Stops only ever tighten and targets
    only ever extend, so the agent cannot dodge a loss by loosening risk or
    cut winners by pulling the target in (it can still exit explicitly).
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.reset()

    def reset(self) -> None:
        self.side = FLAT
        self.entry = self.sl = self.tp = self.risk = self.best = 0.0
        self.tp_rr = self.trail_gap_r = 0.0
        self.hold_ticks = 0
        self.trail_active = False

    def state_features(self, price: float, atr: float) -> list[float]:
        if self.side == FLAT:
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        favorable = self.side * (self.best - self.entry)
        return [
            0.0,
            float(self.side == LONG),
            float(self.side == SHORT),
            self.side * (price - self.entry) / self.risk,
            self.side * (price - self.sl) / atr,
            self.side * (self.tp - price) / atr,
            favorable / self.risk,
            self.hold_ticks / 500.0,
            self.tp_rr,
        ]

    def step(
        self,
        price: float,
        atr: float,
        action: Sequence[float],
        *,
        allow_entry: bool = True,
        momentum: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Process one tick. Returns entry/exit events (at most one per tick).

        ``momentum`` (from ``TickFeatureBuilder.momentum_atr``) gates entries:
        a trade is taken only if every window supports its direction. Pass None
        to skip the gate.
        """
        price = float(price)
        cfg = self.config
        intent, sl_mult, tp_rr, trail_gap = map_action(
            action,
            sl_atr_bounds=cfg.sl_atr_bounds,
            tp_rr_bounds=cfg.tp_rr_bounds,
            trailing_gap_r_bounds=cfg.trailing_gap_r_bounds,
        )
        if self.side != FLAT:
            self.hold_ticks += 1
            self.best = max(self.best, price) if self.side == LONG else min(self.best, price)
            self._tighten_trailing()
            if self.side * (price - self.sl) <= 0:
                return [self._close(price, "trailing_stop" if self.trail_active else "stop_loss")]
            if self.side * (price - self.tp) >= 0:
                return [self._close(price, "take_profit")]
            if intent == -self.side:
                return [self._close(price, "policy_exit")]
            candidate_sl = self.entry - self.side * sl_mult * atr
            self.sl = max(self.sl, candidate_sl) if self.side == LONG else min(self.sl, candidate_sl)
            if tp_rr > self.tp_rr:
                self.tp_rr = tp_rr
                self.tp = self.entry + self.side * tp_rr * self.risk
            self.trail_gap_r = trail_gap
            self._tighten_trailing()
            return []
        if allow_entry and intent != FLAT and self._momentum_supports(intent, momentum):
            return [self._open(intent, price, atr, sl_mult, tp_rr, trail_gap)]
        return []

    def _momentum_supports(self, side: int, momentum: Sequence[float] | None) -> bool:
        if momentum is None:
            return True
        thresholds = self.config.min_entry_momentum_atr
        return len(momentum) >= len(thresholds) and all(
            side * float(m) >= threshold
            for m, threshold in zip(momentum[:len(thresholds)], thresholds)
        )

    def force_close(self, price: float, reason: str) -> list[dict[str, Any]]:
        return [self._close(float(price), reason)] if self.side != FLAT else []

    def _tighten_trailing(self) -> None:
        cfg = self.config
        if self.side == FLAT or self.side * (self.best - self.entry) < cfg.trail_activation_points:
            return
        self.trail_active = True
        # Breakeven floor (never beyond the best price) plus the gap-based trail;
        # the stop only ever ratchets in the favorable direction.
        breakeven = self.entry + self.side * cfg.trail_breakeven_buffer_points
        breakeven = min(breakeven, self.best) if self.side == LONG else max(breakeven, self.best)
        trail = self.best - self.side * self.trail_gap_r * self.risk
        level = max(breakeven, trail) if self.side == LONG else min(breakeven, trail)
        self.sl = max(self.sl, level) if self.side == LONG else min(self.sl, level)

    def _open(self, side: int, price: float, atr: float, sl_mult: float, tp_rr: float, trail_gap: float) -> dict[str, Any]:
        self.side = side
        self.entry = price + side * self.config.slippage_points
        self.risk = max(sl_mult * atr, self.config.min_risk_points)
        self.sl = self.entry - side * self.risk
        self.tp_rr = tp_rr
        self.tp = self.entry + side * tp_rr * self.risk
        self.best = price
        self.trail_gap_r = trail_gap
        self.hold_ticks = 0
        self.trail_active = False
        return {
            "type": "entry", "side": SIDE_NAMES[side], "signal_price": price,
            "entry_price": self.entry, "stop_loss": self.sl, "take_profit": self.tp,
            "risk_points": self.risk, "target_rr": tp_rr, "atr": atr,
        }

    def _close(self, price: float, reason: str) -> dict[str, Any]:
        fill = price - self.side * self.config.slippage_points
        pnl_points = self.side * (fill - self.entry) - self.config.cost_points
        event = {
            "type": "exit", "side": SIDE_NAMES[self.side], "reason": reason,
            "signal_price": price, "exit_price": fill, "entry_price": self.entry,
            "stop_loss": self.sl, "take_profit": self.tp, "risk_points": self.risk,
            "pnl_points": pnl_points, "pnl_r": pnl_points / self.risk,
            "max_favorable_r": self.side * (self.best - self.entry) / self.risk,
            "hold_ticks": self.hold_ticks,
            "trail_active": self.trail_active,
        }
        self.reset()
        return event