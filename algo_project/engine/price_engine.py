"""Derives option-level price context (ATM strike, CE/PE snapshot)
from an AOCParser snapshot, for use by DecisionEngine / OrderManager.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)


@dataclass
class OptionQuote:
    strike: float
    option_type: str  # "CE" or "PE"
    ltp_chg: str = ""
    oi: str = ""
    oi_chg: str = ""
    volume: str = ""
    iv_delta: str = ""


class PriceEngine:
    """Reads an AOCParser snapshot and exposes tradable price context."""

    PIVOT_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "daily_pivot_cache.json"

    @staticmethod
    def classic_pivots(high: float, low: float, close: float) -> dict:
        """Calculate standard daily pivot levels from a completed session."""
        high, low, close = float(high), float(low), float(close)
        pp = (high + low + close) / 3.0
        return {
            "pp": pp,
            "s1": (2.0 * pp) - high,
            "r1": (2.0 * pp) - low,
            "s2": pp - (high - low),
            "r2": pp + (high - low),
        }

    @classmethod
    def load_pivot_cache(cls, symbol: str) -> dict:
        try:
            payload = json.loads(cls.PIVOT_CACHE_PATH.read_text(encoding="utf-8"))
            entry = payload.get(str(symbol).upper(), {}) if isinstance(payload, dict) else {}
            pivots = entry.get("pivots", {}) if isinstance(entry, dict) else {}
            if all(isinstance(pivots.get(key), (int, float)) for key in ("s1", "r1")):
                return entry
        except (OSError, ValueError, TypeError):
            pass
        return {}

    @classmethod
    def save_pivot_cache(cls, symbol: str, high: float, low: float, close: float) -> dict:
        pivots = cls.classic_pivots(high, low, close)
        payload = {}
        try:
            if cls.PIVOT_CACHE_PATH.exists():
                loaded = json.loads(cls.PIVOT_CACHE_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
        except (OSError, ValueError, TypeError):
            payload = {}
        payload[str(symbol).upper()] = {
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "pivots": pivots,
        }
        cls.PIVOT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.PIVOT_CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload[str(symbol).upper()]

    def analyze(self, snapshot: dict) -> dict:
        market_strike = snapshot.get("market_strike")
        rows = snapshot.get("rows", [])

        row = self._find_row(rows, market_strike) if market_strike is not None else None

        result = {
            "market_strike": market_strike,
            "atm_ce": self._quote(row, "ce", market_strike) if row else None,
            "atm_pe": self._quote(row, "pe", market_strike) if row else None,
        }

        log.debug("PriceEngine result: %s", result)
        return result

    @staticmethod
    def _number(value) -> float | None:
        try:
            if value is None or not str(value).strip():
                return None
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    @classmethod
    def build_unified_chain(
        cls,
        option_chain: dict,
        pivots: dict,
        spot: float | None,
        strike_step: float = 50.0,
        radius: int = 2,
    ) -> dict:
        """Build the near-ATM strike/OI/pivot ladder used by decisions and UI."""
        grouped = option_chain.get("by_strike", {}) if isinstance(option_chain, dict) else {}
        if spot is None:
            return {"spot": spot, "strikes": [], "levels": [], "by_strike": {}, "target_up": None, "target_down": None}

        step = abs(float(strike_step)) or 50.0
        atm = round(float(spot) / step) * step
        # Use only strikes actually returned by the broker; never invent an OC level.
        strikes = sorted({cls._number(value) for value in grouped} - {None})
        nearby = [strike for strike in strikes if abs(strike - atm) <= radius * step]
        numeric_quotes = {
            cls._number(raw_strike): (sides or {})
            for raw_strike, sides in grouped.items()
            if cls._number(raw_strike) is not None
        }
        pivot_values = []
        for name, value in (pivots or {}).items():
            numeric = cls._number(value)
            if numeric is not None:
                aligned = round(numeric / step) * step
                pivot_values.append((aligned, name.upper(), numeric))

        levels = []
        for index, strike in enumerate(nearby):
            sides = numeric_quotes.get(strike, {})
            ce = sides.get("CE", {}) or {}
            pe = sides.get("PE", {}) or {}
            ce_score = cls._wall_score(ce)
            pe_score = cls._wall_score(pe)
            pivot_labels = [name for aligned, name, _ in pivot_values if aligned == strike]
            # Each strike is a ladder node: its support is the node itself and
            # its resistance is the next broker strike. Therefore R of one node
            # is exactly S of the next node.
            next_strike = nearby[index + 1] if index + 1 < len(nearby) else None
            is_support = strike <= float(spot)
            is_resistance = next_strike is not None and next_strike > float(spot)
            levels.append({
                "strike": strike,
                "support": strike,
                "resistance": next_strike,
                "support_score": pe_score,
                "resistance_score": ce_score,
                "support_target_score": pe_score if strike < float(spot) else 0.0,
                "resistance_target_score": ce_score if next_strike is not None and next_strike > float(spot) else 0.0,
                "support_source": "PE_OI_VOLUME" if pe_score > 0 else "PRICE_ACTION_FLIP" if is_support else "CONNECTED_LADDER",
                "resistance_source": "CE_OI_VOLUME" if ce_score > 0 else "PRICE_ACTION_FLIP" if is_resistance else "CONNECTED_LADDER",
                "pivot_labels": pivot_labels,
                "next_strike": next_strike,
                "ce_oi": cls._number(ce.get("open_interest")),
                "pe_oi": cls._number(pe.get("open_interest")),
            })

        pivot_targets = sorted({aligned for aligned, _, _ in pivot_values})
        above = [level for level in pivot_targets if level > float(spot)]
        below = [level for level in pivot_targets if level < float(spot)]
        chain_above = [level for level in levels if level["resistance"] is not None and level["resistance"] > float(spot)]
        chain_below = [level for level in levels if level["support"] is not None and level["strike"] < float(spot)]
        target_up = min(above + [level["resistance"] for level in chain_above]) if above or chain_above else None
        target_down = max(below + [level["support"] for level in chain_below]) if below or chain_below else None
        return {
            "spot": float(spot),
            "atm": atm,
            "strike_step": step,
            "strikes": nearby,
            "levels": levels,
            "by_strike": {str(level["strike"]): level for level in levels},
            "target_up": target_up,
            "target_down": target_down,
        }

    @classmethod
    def _wall_score(cls, quote: dict) -> float:
        oi = cls._number(quote.get("open_interest")) or 0.0
        oi_change = max(cls._number(quote.get("oi_change")) or 0.0, 0.0)
        volume = cls._number(quote.get("trade_volume")) or 0.0
        return oi + oi_change + volume

    @staticmethod
    def _find_row(rows: list, strike: float | None) -> dict | None:
        if strike is None:
            return None
            
        target_strike = round(float(strike), 2)
        for row in rows:
            row_strike = row.get("strike")
            if row_strike is not None:
                try:
                    if round(float(row_strike), 2) == target_strike:
                        return row
                except (ValueError, TypeError):
                    continue
        return None

    @staticmethod
    def combine_support_resistance(
        current_price: float | None,
        aoc_support: float | None = None,
        aoc_resistance: float | None = None,
        market_support: float | None = None,
        market_resistance: float | None = None,
    ) -> dict:
        result = {
            "support": market_support,
            "resistance": market_resistance,
            "sources": {
                "aoc_support": aoc_support,
                "aoc_resistance": aoc_resistance,
                "market_support": market_support,
                "market_resistance": market_resistance,
            },
        }
        log.debug("Using broker classic pivot support/resistance: %s", result)
        return result

    @staticmethod
    def _quote(row: dict, side: str, strike: float) -> OptionQuote:
        data = row.get(side, {}) or {}
        return OptionQuote(
            strike=strike,
            option_type=side.upper(),
            ltp_chg=str(data.get("ltp_chg", "")),
            oi=str(data.get("oi", "")),
            oi_chg=str(data.get("oi_chg", "")),
            volume=str(data.get("volume", "")),
            iv_delta=str(data.get("iv_delta", "")),
        )