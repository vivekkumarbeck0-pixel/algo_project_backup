"""Offline, trade-level Gymnasium environment for crude options PPO experiments."""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


ACTION_HOLD = 0
ACTION_ENTER_CE = 1
ACTION_ENTER_PE = 2
ACTION_EXIT = 3

ACTION_NAMES = {
    ACTION_HOLD: "hold",
    ACTION_ENTER_CE: "enter_ce",
    ACTION_ENTER_PE: "enter_pe",
    ACTION_EXIT: "exit",
}

IDLE_REWARD = -1.5
ENTRY_BONUS = 1.5
UNSUPPORTED_ACTION_REWARD = -1.5
TRADE_REWARD_SCALE = 0.5
INVALID_REENTRY_REWARD = -2.5

OBSERVATION_FEATURES = (
    "entry_open",
    "entry_high",
    "entry_low",
    "entry_close",
    "entry_atr",
    "volume",
    "oi_change",
    "nymex_trend",
    "support_distance_atr",
    "resistance_distance_atr",
    "candle_body_atr",
    "candle_range_atr",
    "position_open",
    "position_side",
)

_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("entry open", "entry candle open", "open", "open price", "entry_open"),
    "high": ("entry high", "entry candle high", "high", "high price", "entry_high"),
    "low": ("entry low", "entry candle low", "low", "low price", "entry_low"),
    "close": (
        "entry close",
        "entry candle close",
        "close",
        "close price",
        "entry futures price",
        "underlying entry price",
        "futures price",
        "spot",
        "entry_close",
    ),
    "atr": ("entry atr", "atr", "entry_atr"),
    "volume": ("entry volume", "volume", "entry_volume"),
    "oi_change": (
        "entry oi_change",
        "entry oi change",
        "oi change",
        "oi_change",
        "entry_oi_change",
    ),
    "trend": (
        "entry nymex trend",
        "nymex trend",
        "nymex_trend",
        "entry market regime",
        "market regime",
        "trend",
    ),
    "support_distance": (
        "support distance",
        "distance to support",
        "support_distance",
        "support distance atr",
        "distance to support atr",
        "support_distance_atr",
    ),
    "resistance_distance": (
        "resistance distance",
        "distance to resistance",
        "resistance_distance",
        "resistance distance atr",
        "distance to resistance atr",
        "resistance_distance_atr",
    ),
    "support": ("support", "pivot support", "support level"),
    "resistance": ("resistance", "pivot resistance", "resistance level"),
    "side": ("option side", "option type", "side", "trade side", "action"),
    "pnl": ("pnl", "net pnl", "profit loss", "profit_loss", "realized pnl"),
    "timestamp": ("entry timestamp", "timestamp", "datetime", "date"),
    "outcome": ("outcome", "exit reason", "exit scenario", "result", "trade outcome"),
}


def _normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _column_map(frame: pd.DataFrame) -> dict[str, str]:
    return {_normalize_name(column): str(column) for column in frame.columns}


def _column_name(frame: pd.DataFrame, field: str) -> str | None:
    columns = _column_map(frame)
    for alias in _ALIASES[field]:
        name = columns.get(_normalize_name(alias))
        if name is not None:
            return name
    return None


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _normalize_side(value: object) -> str:
    normalized = _normalize_name(value)
    if normalized in {"ce", "call", "call option", "enter ce", "buy ce"}:
        return "CE"
    if normalized in {"pe", "put", "put option", "enter pe", "buy pe"}:
        return "PE"
    return ""


def _reward_for_trade(outcome: object, pnl: float) -> float:
    """Scale trade outcomes to reduce the impact of individual losses."""
    label = _normalize_name(outcome)
    if label:
        if "invalid" in label and ("sl" in label or "stop" in label):
            base_reward = -10.0
        elif "panic" in label:
            base_reward = -2.0
        elif "stop" in label or re.search(r"\bsl\b", label):
            base_reward = -8.0
        elif "optimal" in label:
            base_reward = 10.0
        elif "target" in label:
            base_reward = 9.0
        elif pnl > 0:
            base_reward = 8.0
        elif pnl < 0:
            base_reward = -8.0
        else:
            base_reward = 0.0
    elif pnl > 0:
        base_reward = 8.0
    elif pnl < 0:
        base_reward = -8.0
    else:
        base_reward = 0.0
    return base_reward * TRADE_REWARD_SCALE


def load_trade_data(path: str | Path) -> pd.DataFrame:
    """Load CSV or JSONL trade rows without changing any live-bot data."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"trade dataset does not exist: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source, encoding="utf-8-sig")
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(source, lines=True)
    else:
        raise ValueError("dataset must be a .csv, .jsonl, or .ndjson file")
    return prepare_trade_data(frame)


def prepare_trade_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate completed trades and add canonical fields used by the environment."""
    if frame.empty:
        raise ValueError("trade dataset is empty")

    side_column = _column_name(frame, "side")
    pnl_column = _column_name(frame, "pnl")
    if side_column is None:
        raise ValueError("trade data must contain Option Side/Option Type (CE or PE)")
    if pnl_column is None:
        raise ValueError("trade data must contain realized PnL for holdout evaluation")

    prepared = frame.copy()
    prepared["_ppo_side"] = prepared[side_column].map(_normalize_side)
    prepared["_ppo_pnl"] = pd.to_numeric(prepared[pnl_column], errors="coerce")
    finite_pnl = np.isfinite(prepared["_ppo_pnl"].to_numpy(dtype=float, na_value=np.nan))
    complete = prepared["_ppo_side"].ne("") & finite_pnl

    dropped = int((~complete).sum())
    if dropped:
        warnings.warn(
            f"Skipped {dropped} row(s) without a CE/PE side and finite realized PnL.",
            stacklevel=2,
        )
    prepared = prepared.loc[complete].copy()
    if prepared.empty:
        raise ValueError("no completed CE/PE trades with finite realized PnL were found")

    outcome_column = _column_name(prepared, "outcome")
    if outcome_column is None:
        prepared["_ppo_outcome"] = ""
    else:
        prepared["_ppo_outcome"] = prepared[outcome_column].fillna("").astype(str)
    prepared["_ppo_reward"] = [
        _reward_for_trade(outcome, float(pnl))
        for outcome, pnl in zip(prepared["_ppo_outcome"], prepared["_ppo_pnl"])
    ]

    timestamp_column = _column_name(prepared, "timestamp")
    if timestamp_column is not None:
        timestamps = pd.to_datetime(
            prepared[timestamp_column], errors="coerce", dayfirst=True
        )
        if timestamps.isna().any():
            raise ValueError(
                f"column {timestamp_column!r} contains invalid timestamps; "
                "fix them before making a chronological split"
            )
        prepared["_ppo_timestamp"] = timestamps
        prepared = prepared.sort_values("_ppo_timestamp", kind="stable")
    prepared = prepared.reset_index(drop=True)
    prepared.attrs["skipped_incomplete_rows"] = dropped
    prepared.attrs["has_timestamp"] = timestamp_column is not None
    return prepared


def _lookup(row: pd.Series, frame: pd.DataFrame, field: str) -> object | None:
    name = _column_name(frame, field)
    if name is None:
        return None
    return row[name]


def _trend_value(value: object) -> float:
    if isinstance(value, (int, float, np.number)):
        return float(np.clip(_finite_float(value), -1.0, 1.0))
    trend = _normalize_name(value)
    if any(word in trend for word in ("bull", "uptrend", "up trend", "positive")):
        return 1.0
    if any(word in trend for word in ("bear", "downtrend", "down trend", "negative")):
        return -1.0
    return 0.0


def _row_features(row: pd.Series, frame: pd.DataFrame) -> np.ndarray:
    atr = max(_finite_float(_lookup(row, frame, "atr")), 0.0)
    close_value = _lookup(row, frame, "close")
    close = _finite_float(close_value)
    open_price = _finite_float(_lookup(row, frame, "open"), close)
    high = _finite_float(_lookup(row, frame, "high"), close)
    low = _finite_float(_lookup(row, frame, "low"), close)

    support_distance_value = _lookup(row, frame, "support_distance")
    resistance_distance_value = _lookup(row, frame, "resistance_distance")
    support_distance_column = _column_name(frame, "support_distance")
    resistance_distance_column = _column_name(frame, "resistance_distance")
    close_column = _column_name(frame, "close")
    has_close = close_column is not None and pd.notna(row[close_column])
    if support_distance_value is None or pd.isna(support_distance_value):
        support = _lookup(row, frame, "support")
        support_distance = (
            close - _finite_float(support)
            if support is not None and has_close
            else 0.0
        )
    else:
        support_distance = _finite_float(support_distance_value)
    if resistance_distance_value is None or pd.isna(resistance_distance_value):
        resistance = _lookup(row, frame, "resistance")
        resistance_distance = (
            _finite_float(resistance) - close
            if resistance is not None and has_close
            else 0.0
        )
    else:
        resistance_distance = _finite_float(resistance_distance_value)

    # Distances are expressed in ATR units so they remain comparable over time.
    scale = atr if atr > 0.0 else 1.0
    if (
        support_distance_column is not None
        and "atr" in _normalize_name(support_distance_column)
    ):
        support_distance *= scale
    if (
        resistance_distance_column is not None
        and "atr" in _normalize_name(resistance_distance_column)
    ):
        resistance_distance *= scale
    volume = max(_finite_float(_lookup(row, frame, "volume")), 0.0)
    oi_change = _finite_float(_lookup(row, frame, "oi_change"))
    trend = _trend_value(_lookup(row, frame, "trend"))
    features = np.asarray(
        [
            open_price,
            high,
            low,
            close,
            atr,
            math.log1p(volume),
            math.copysign(math.log1p(abs(oi_change)), oi_change) if oi_change else 0.0,
            trend,
            support_distance / scale,
            resistance_distance / scale,
            (close - open_price) / scale,
            max(high - low, 0.0) / scale,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=1_000_000.0, neginf=-1_000_000.0)


def missing_feature_groups(frame: pd.DataFrame) -> list[str]:
    """Return state groups absent from the dataset; absent values are zero-filled."""
    missing = [
        f"entry {field}"
        for field in ("open", "high", "low", "close")
        if _column_name(frame, field) is None
    ]
    for label, field in (
        ("ATR", "atr"),
        ("volume", "volume"),
        ("OI change", "oi_change"),
        ("NYMEX trend", "trend"),
    ):
        if _column_name(frame, field) is None:
            missing.append(label)
    if (
        _column_name(frame, "support_distance") is None
        and (
            _column_name(frame, "support") is None
            or _column_name(frame, "close") is None
        )
    ):
        missing.append("support distance or support level")
    if (
        _column_name(frame, "resistance_distance") is None
        and (
            _column_name(frame, "resistance") is None
            or _column_name(frame, "close") is None
        )
    ):
        missing.append("resistance distance or resistance level")
    return missing


class CrudePPOEnv(gym.Env[np.ndarray, int]):
    """Replay one completed historical trade per episode.

    Only the recorded option side has a known outcome. Opposite-side entries are
    marked unsupported and receive no fabricated counterfactual PnL or reward.
    Since rows are completed trades rather than intratrade candles, either Hold
    or Exit closes an opened replay at its already-recorded outcome.
    """

    metadata = {"render_modes": []}

    def __init__(self, data: pd.DataFrame | str | Path):
        super().__init__()
        frame = load_trade_data(data) if isinstance(data, (str, Path)) else prepare_trade_data(data)
        self.data = frame.reset_index(drop=True)
        self._features = np.vstack([_row_features(row, self.data) for _, row in self.data.iterrows()])
        self._cursor = 0
        self._row_index = 0
        self._position_side = 0
        self._position_open = False
        self._episode_finished = False

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(OBSERVATION_FEATURES),),
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        position = np.asarray(
            [float(self._position_open), float(self._position_side)],
            dtype=np.float32,
        )
        return np.concatenate((self._features[self._row_index], position)).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        requested_index = (options or {}).get("row_index")
        if requested_index is None:
            self._row_index = self._cursor
            self._cursor = (self._cursor + 1) % len(self.data)
        else:
            self._row_index = int(requested_index)
            if not 0 <= self._row_index < len(self.data):
                raise ValueError("reset option row_index is outside the dataset")
        self._position_side = 0
        self._position_open = False
        self._episode_finished = False
        return self._observation(), {"row_index": self._row_index}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_finished:
            raise RuntimeError("episode is finished; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be in [0, 3], received {action!r}")

        row = self.data.iloc[self._row_index]
        info: dict[str, Any] = {
            "row_index": self._row_index,
            "action": ACTION_NAMES[int(action)],
            "recorded_side": str(row["_ppo_side"]),
            "entered_trade": False,
            "unsupported_counterfactual": False,
            "pnl": 0.0,
            "reward": 0.0,
        }

        if self._position_open:
            if action in (ACTION_ENTER_CE, ACTION_ENTER_PE):
                self._episode_finished = True
                info["invalid_reentry"] = True
                info["reward"] = INVALID_REENTRY_REWARD
                return self._observation(), INVALID_REENTRY_REWARD, True, False, info

            pnl = float(row["_ppo_pnl"])
            reward = float(row["_ppo_reward"])
            info.update(
                {
                    "entered_trade": True,
                    "pnl": pnl,
                    "reward": reward,
                    "exit_outcome": str(row["_ppo_outcome"]),
                }
            )
            self._episode_finished = True
            return self._observation(), reward, True, False, info

        if action in (ACTION_HOLD, ACTION_EXIT):
            self._episode_finished = True
            info["skipped_trade"] = True
            info["reward"] = IDLE_REWARD
            return self._observation(), IDLE_REWARD, True, False, info

        requested_side = "CE" if action == ACTION_ENTER_CE else "PE"
        if requested_side != row["_ppo_side"]:
            self._episode_finished = True
            info["unsupported_counterfactual"] = True
            info["reward"] = UNSUPPORTED_ACTION_REWARD
            return self._observation(), UNSUPPORTED_ACTION_REWARD, True, False, info

        self._position_side = 1 if requested_side == "CE" else -1
        self._position_open = True
        info["position_opened"] = True
        info["reward"] = ENTRY_BONUS
        return self._observation(), ENTRY_BONUS, False, False, info
