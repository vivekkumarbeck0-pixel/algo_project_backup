"""Date-scoped JSON persistence for the intraday trading book."""

import json
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)


class DailyStateStore:
    """Persists one trading day's positions and risk summary atomically."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, trading_date: str) -> dict | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Unable to load daily state from %s: %s", self.path, exc)
            return None
        if payload.get("date") == trading_date:
            return payload

        # A restart after midnight must not lose an open paper position. Start a
        # fresh daily book while carrying only positions that are still active.
        open_positions = [
            position for position in payload.get("positions", [])
            if isinstance(position, dict) and position.get("status", "OPEN") == "OPEN"
        ]
        rollover = {
            "date": trading_date,
            "positions": open_positions,
            "trades_today": 0,
            "consecutive_stop_losses": 0,
            "daily_realized_pnl": 0.0,
        }
        self.save(rollover)
        log.info(
            "Rolled daily state from %s to %s while preserving %d open position(s)",
            payload.get("date"),
            trading_date,
            len(open_positions),
        )
        return rollover

    def save(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary_path.replace(self.path)