"""Append-only signal dataset for offline probability-model training."""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4


class TrainingDataStore:
    """Stores every evaluated signal and later attaches its trade outcome."""

    def __init__(self, path: str = "data/training_signals.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def record_signal(self, symbol, snapshot, market_structure, decision) -> str:
        record_id = uuid4().hex
        row = {
            "record_id": record_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "symbol": symbol,
            "spot": self._number(snapshot.get("spot")),
            "market_strike": self._number(snapshot.get("market_strike")),
            "support": self._number(decision.support),
            "resistance": self._number(decision.resistance),
            "trend": market_structure.get("trend"),
            "micro_momentum": market_structure.get("micro_momentum"),
            "entry_confirmed": bool(market_structure.get("entry_confirmed")),
            "option_chain_rows": len((market_structure.get("option_chain") or {}).get("rows", [])),
            "smc_patterns": (market_structure.get("candlestick_patterns") or {}).get("patterns", []),
            "action": decision.action,
            "option_type": decision.option_type,
            "strike": self._number(decision.strike),
            "confidence": decision.confidence,
            "risk_approved": bool(decision.risk_approved),
            "outcome": None,
            "pnl": None,
            "trade_number": None,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        return record_id

    def attach_trade(self, record_id: str, trade_number: int) -> None:
        self._update(record_id, {"trade_number": trade_number})

    def attach_outcome(self, trade_number: int, outcome: str, pnl: float | None) -> None:
        self._update_by_trade_number(trade_number, {"outcome": outcome, "pnl": pnl})

    def attach_outcome_for_record(self, record_id: str, outcome: str, pnl: float | None) -> None:
        self._update(record_id, {"outcome": outcome, "pnl": pnl})

    def _update(self, record_id: str, changes: dict) -> None:
        self._rewrite(lambda row: row.get("record_id") == record_id, changes)

    def _update_by_trade_number(self, trade_number: int, changes: dict) -> None:
        self._rewrite(lambda row: row.get("trade_number") == trade_number, changes)

    def _rewrite(self, predicate, changes: dict) -> None:
        if not self.path.exists():
            return
        rows = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if predicate(row):
                    row.update(changes)
                rows.append(row)
        with self.path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
