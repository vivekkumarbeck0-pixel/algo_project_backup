"""Validated handoff between offline optimization and the live Crude engine."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
PARAMETER_RANGES: dict[str, tuple[float, float]] = {
    "crude_atr_multiplier": (0.5, 4.0),
    "buffer_points": (0.0, 20.0),
    "crude_risk_reward_ratio": (1.0, 4.0),
    "crude_trailing_activation_points": (0.0, 100.0),
    "crude_min_entry_volume": (0.0, 1_000_000_000.0),
    "crude_min_abs_oi_change": (0.0, 1_000_000_000.0),
    "crude_min_entry_atr": (0.0, 1_000_000.0),
    "crude_max_entry_atr": (0.0, 1_000_000.0),
}
ALLOWED_REGIMES = {"TRENDING", "SIDEWAYS", "UNKNOWN"}


class AdaptiveConfigError(ValueError):
    pass


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AdaptiveConfigError("unsupported adaptive config schema_version")
    if payload.get("status") != "approved":
        raise AdaptiveConfigError("adaptive config is not approved for deployment")

    raw_parameters = payload.get("parameters")
    if not isinstance(raw_parameters, Mapping):
        raise AdaptiveConfigError("parameters must be an object")

    unknown = set(raw_parameters) - (set(PARAMETER_RANGES) | {"crude_allowed_regimes"})
    if unknown:
        raise AdaptiveConfigError(f"unsupported parameters: {sorted(unknown)}")

    parameters: dict[str, Any] = {}
    for name, value in raw_parameters.items():
        if name == "crude_allowed_regimes":
            if not isinstance(value, list) or not value:
                raise AdaptiveConfigError("crude_allowed_regimes must be a non-empty list")
            regimes = tuple(str(item).upper() for item in value)
            if not set(regimes) <= ALLOWED_REGIMES:
                raise AdaptiveConfigError("crude_allowed_regimes contains an invalid regime")
            parameters[name] = regimes
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AdaptiveConfigError(f"{name} must be numeric")
        minimum, maximum = PARAMETER_RANGES[name]
        numeric_value = float(value)
        if not minimum <= numeric_value <= maximum:
            raise AdaptiveConfigError(f"{name} must be between {minimum} and {maximum}")
        parameters[name] = numeric_value

    minimum_atr = parameters.get("crude_min_entry_atr", 0.0)
    maximum_atr = parameters.get("crude_max_entry_atr", float("inf"))
    if minimum_atr > maximum_atr:
        raise AdaptiveConfigError("crude_min_entry_atr cannot exceed crude_max_entry_atr")
    return parameters


def load_payload(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise AdaptiveConfigError("adaptive config root must be an object")
    return validate_payload(payload), payload


def apply_parameters(settings: Any, parameters: Mapping[str, Any]) -> None:
    missing = [name for name in parameters if not hasattr(settings, name)]
    if missing:
        raise AdaptiveConfigError(f"live settings do not define {missing}")
    for name, value in parameters.items():
        setattr(settings, name, value)


def publish_payload(path: str | Path, parameters: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    validated = validate_payload({
        "schema_version": SCHEMA_VERSION,
        "status": "approved",
        "parameters": dict(parameters),
    })
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "approved",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in validated.items()
        },
        "validation": dict(validation),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


@dataclass
class AdaptiveConfigReloader:
    path: Path
    last_mtime_ns: int | None = None

    def refresh(self, settings: Any, position_is_open: bool) -> bool:
        if position_is_open or not self.path.exists():
            return False
        mtime_ns = self.path.stat().st_mtime_ns
        if self.last_mtime_ns == mtime_ns:
            return False
        parameters, _ = load_payload(self.path)
        apply_parameters(settings, parameters)
        self.last_mtime_ns = mtime_ns
        return True