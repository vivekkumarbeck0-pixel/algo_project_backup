"""Combines AOCEngine's market-structure signal with risk checks to
produce a final, actionable trading `Decision`.

Flow:
    OCRReader -> AOCParser -> market snapshot -> DecisionEngine.decide()
                                                   -> RiskManager.evaluate()
                                                   -> OrderManager.execute() (paper or live)
"""

from dataclasses import dataclass, field
from datetime import datetime
import pytz

from config import settings, SYMBOL_REGISTRY, DEFAULT_SYMBOL
from engine.price_engine import PriceEngine
from engine.risk_manager import RiskManager
from logger import get_logger

log = get_logger(__name__)


@dataclass
class Decision:
    action: str  # "BUY", "SELL", "HOLD", "WAIT", "NO_TRADE"
    option_type: str | None  # "CE" or "PE"
    strike: float | None
    quantity: int
    confidence: str  # "LOW", "MEDIUM", "HIGH"
    reasons: list[str] = field(default_factory=list)
    risk_approved: bool = False
    # Dynamic symbol switching (e.g. "NIFTY" vs "CRUDEOIL")
    underlying: str = DEFAULT_SYMBOL
    exchange: str = "NFO"
    # Dynamic support/resistance
    support: float | None = None
    resistance: float | None = None
    support_range: tuple | None = None  # (min, max) range around support
    resistance_range: tuple | None = None  # (min, max) range around resistance
    trade_type: str | None = None  # "BUY_CE" or "BUY_PE"
    square_off: bool = False  # intraday cutoff reached: exit all, no new entries
    # Index-point based trade management
    index_entry: float | None = None
    index_sl: float | None = None
    index_target: float | None = None
    sl_points: float = 0.0
    target_points: float = 0.0
    # Market status
    market_status: str = "OPEN"  # "OPEN" or "CLOSED"


class DecisionEngine:
    """Turns raw AOC/market signals into one final trade decision."""

    def __init__(
        self,
        risk_manager: RiskManager,
        price_engine: PriceEngine | None = None,
    ):
        self.price_engine = price_engine or PriceEngine()
        self.risk_manager = risk_manager

    def _get_symbol_config(self, underlying: str) -> dict:
        """Helper to get symbol config with robust fallbacks."""
        return SYMBOL_REGISTRY.get(underlying, SYMBOL_REGISTRY.get(DEFAULT_SYMBOL, {
            "exchange": "NFO",
            "lot_size": 25,
            "sl_points": 20.0,
            "target_points": 40.0,
            "strike_step": 50
        }))

    @staticmethod
    def _in_range(price, level_range) -> bool:
        if price is None or not level_range:
            return False
        try:
            low, high = level_range
            return float(low) <= float(price) <= float(high)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _aoc_level_matches(level, candidates, strike_step) -> bool:
        if level is None:
            return False
        candidates = candidates or []
        if not candidates:
            return True
        try:
            tolerance = max(float(strike_step), float(strike_step) * 1.5)
            level_value = float(level)
            for candidate in candidates:
                try:
                    if abs(level_value - float(candidate)) <= tolerance:
                        return True
                except (TypeError, ValueError):
                    continue
            return False
        except (TypeError, ValueError):
            return False

    def _is_auto_square_off_time(self, underlying: str = DEFAULT_SYMBOL) -> bool:
        """Apply the 15:15 cutoff only to index sessions, not MCX Crude."""
        if str(underlying).upper() == "CRUDEOIL":
            return False
        try:
            tz = pytz.timezone("Asia/Kolkata")
            now_ist = datetime.now(tz)
            cutoff_time = now_ist.replace(hour=15, minute=15, second=0, microsecond=0)
            return now_ist >= cutoff_time
        except Exception as e:
            log.error("Error checking market time: %s", e)
            return False

    @staticmethod
    def _to_number(value) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _rank_wall(cls, rows: list[dict], current_price: float | None, side: str) -> float | None:
        """Pick the best OC wall using OI + OI change + volume with proximity tie-break."""
        if not rows:
            return None

        max_oi = max((row["oi"] for row in rows), default=0.0) or 1.0
        max_oi_change = max((max(row["oi_change"], 0.0) for row in rows), default=0.0) or 1.0
        max_volume = max((row["volume"] for row in rows), default=0.0) or 1.0

        best = None
        best_score = None
        for row in rows:
            oi_score = row["oi"] / max_oi if max_oi else 0.0
            oi_change_score = max(row["oi_change"], 0.0) / max_oi_change if max_oi_change else 0.0
            volume_score = row["volume"] / max_volume if max_volume else 0.0

            composite = (0.50 * oi_score) + (0.30 * oi_change_score) + (0.20 * volume_score)

            distance = 0.0
            if current_price is not None:
                distance = abs(row["strike"] - float(current_price))

            candidate = (composite, -distance, row["strike"])
            if best_score is None or candidate > best_score:
                best_score = candidate
                best = row["strike"]

        # Directional guard: CE resistance must be above/at current, PE support below/at current.
        if current_price is not None and best is not None:
            if side == "CE" and best < float(current_price):
                return None
            if side == "PE" and best > float(current_price):
                return None
        return best

    @classmethod
    def _oi_walls(cls, option_chain: dict, current_price: float | None) -> tuple[float | None, float | None]:
        """Return OC-derived PE support and CE resistance using weighted wall scoring."""
        grouped = option_chain.get("by_strike", {}) if isinstance(option_chain, dict) else {}
        supports = []
        resistances = []
        for raw_strike, sides in grouped.items():
            strike = cls._to_number(raw_strike)
            if strike is None:
                continue
            pe = sides.get("PE", {}) or {}
            ce = sides.get("CE", {}) or {}
            pe_oi = cls._to_number(pe.get("open_interest"))
            pe_oi_change = cls._to_number(pe.get("oi_change")) or 0.0
            pe_volume = cls._to_number(pe.get("trade_volume")) or 0.0
            if (pe_oi is not None or pe_oi_change > 0.0) and (
                current_price is None or strike <= float(current_price)
            ):
                supports.append({"oi": pe_oi or 0.0, "oi_change": pe_oi_change, "volume": pe_volume, "strike": strike})

            ce_oi = cls._to_number(ce.get("open_interest"))
            ce_oi_change = cls._to_number(ce.get("oi_change")) or 0.0
            ce_volume = cls._to_number(ce.get("trade_volume")) or 0.0
            if (ce_oi is not None or ce_oi_change > 0.0) and (
                current_price is None or strike >= float(current_price)
            ):
                resistances.append({"oi": ce_oi or 0.0, "oi_change": ce_oi_change, "volume": ce_volume, "strike": strike})

        return (
            cls._rank_wall(supports, current_price, side="PE"),
            cls._rank_wall(resistances, current_price, side="CE"),
        )

    @classmethod
    def _opposite_side_oi_peak_strike(
        cls,
        option_chain: dict,
        current_price: float,
        trade_option_type: str,
    ) -> tuple[float | None, str]:
        """Choose the outermost opposite-side strike at 100% OI or OI change."""
        source_side = "PE" if trade_option_type == "CE" else "CE"
        grouped = option_chain.get("by_strike", {}) if isinstance(option_chain, dict) else {}
        oi_rows = []
        change_rows = []
        for raw_strike, sides in grouped.items():
            strike = cls._to_number(raw_strike)
            if strike is None:
                continue
            quote = (sides or {}).get(source_side, {}) or {}
            oi = cls._to_number(quote.get("open_interest"))
            oi_change = cls._to_number(quote.get("oi_change"))
            if oi is not None and oi > 0:
                oi_rows.append((oi, strike))
            if oi_change is not None and oi_change > 0:
                change_rows.append((oi_change, strike))

        peak_sources_by_strike = {}
        for rows, label in ((oi_rows, "OI 100%"), (change_rows, "OI-change 100%")):
            if not rows:
                continue
            peak_value = max(value for value, _ in rows)
            winners = [strike for value, strike in rows if value == peak_value]
            for strike in winners:
                peak_sources_by_strike.setdefault(strike, []).append(label)
        if not peak_sources_by_strike:
            return None, ""

        peak_strikes = list(peak_sources_by_strike)
        selected = max(peak_strikes) if trade_option_type == "CE" else min(peak_strikes)
        return selected, "/".join(peak_sources_by_strike[selected])

    @classmethod
    def _within_oi_pivot_cutoff(
        cls,
        level,
        pivot_ladder: dict,
        current_price: float,
        option_type: str,
    ) -> bool:
        numeric_level = cls._to_number(level)
        if numeric_level is None:
            return False
        level_key = "resistance" if option_type == "CE" else "support"
        for raw_strike, sr in (pivot_ladder or {}).items():
            strike = cls._to_number(raw_strike)
            if strike is None:
                continue
            if option_type == "CE" and strike <= current_price:
                continue
            if option_type == "PE" and strike >= current_price:
                continue
            if cls._to_number((sr or {}).get(level_key)) == numeric_level:
                return True
        return False

    @classmethod
    def _pivot_target(
        cls,
        market_structure: dict,
        current_price: float,
        option_type: str,
        pivot_ladder: dict | None = None,
    ) -> float | None:
        if pivot_ladder is not None:
            level_key = "resistance" if option_type == "CE" else "support"
            candidates = []
            for raw_strike, sr in pivot_ladder.items():
                strike = cls._to_number(raw_strike)
                level = cls._to_number((sr or {}).get(level_key))
                if strike is None or level is None:
                    continue
                if option_type == "CE" and strike > current_price and level > current_price:
                    candidates.append(level)
                elif option_type == "PE" and strike < current_price and level < current_price:
                    candidates.append(level)
            if not candidates:
                return None
            return min(candidates) if option_type == "CE" else max(candidates)

        unified = market_structure.get("unified_chain") or {}
        chain_target = unified.get("target_up" if option_type == "CE" else "target_down")
        if cls._to_number(chain_target) is not None:
            return float(chain_target)
        pivots = market_structure.get("pivot_levels") or {}
        keys = ("r1", "r2") if option_type == "CE" else ("s1", "s2")
        valid = []
        for key in keys:
            level = cls._to_number(pivots.get(key))
            if level is None:
                continue
            if option_type == "CE" and level > current_price:
                if pivot_ladder is None or cls._within_oi_pivot_cutoff(
                    level, pivot_ladder, current_price, option_type
                ):
                    valid.append(level)
            elif option_type == "PE" and level < current_price:
                if pivot_ladder is None or cls._within_oi_pivot_cutoff(
                    level, pivot_ladder, current_price, option_type
                ):
                    valid.append(level)
        if not valid:
            return None
        return min(valid) if option_type == "CE" else max(valid)

    @staticmethod
    def _select_banded_target(current_price: float, candidates: list, direction: int, fallback_points: float) -> tuple[float, str]:
        """Keep index targets inside the configured 40-60 or 90-110 point bands."""
        numeric = sorted({
            float(level)
            for level in candidates
            if isinstance(level, (int, float))
        })

        bands = ((40.0, 60.0, "T1"), (90.0, 110.0, "T2"))
        for minimum, maximum, label in bands:
            eligible = [
                level for level in numeric
                if minimum <= direction * (level - current_price) <= maximum
            ]
            if eligible:
                nearest = min(eligible, key=lambda level: abs(level - current_price))
                return nearest, label

        return current_price + (direction * fallback_points), "T1_FALLBACK"

    @classmethod
    def _ladder_pivot_level(cls, pivot_ladder: dict, strike: float, option_type: str) -> float | None:
        """Map a strike onto its exact pivot S/R level from the connected ladder."""
        ladder_level = cls._ladder_node(pivot_ladder, strike)
        if not ladder_level:
            return None

        level_key = "resistance" if option_type == "CE" else "support"
        return cls._to_number(ladder_level.get(level_key))

    @classmethod
    def _ladder_node(cls, pivot_ladder: dict, strike: float) -> dict | None:
        for raw_strike, level in (pivot_ladder or {}).items():
            if cls._to_number(raw_strike) == strike:
                return level
        return None

    @classmethod
    def _nifty_oi_entry(cls, option_chain, pivot_ladder, current_price, trade_option_type):
        strike, source = cls._opposite_side_oi_peak_strike(
            option_chain, current_price, trade_option_type
        )
        node = cls._ladder_node(pivot_ladder, strike) if strike is not None else None
        if node is None:
            return strike, source, None, None
        entry_level = node.get("support") if trade_option_type == "CE" else node.get("resistance")
        target_level = node.get("resistance") if trade_option_type == "CE" else node.get("support")
        return strike, source, cls._to_number(entry_level), cls._to_number(target_level)

    @classmethod
    def _peak_oi_pivot_target(
        cls,
        option_chain: dict,
        pivot_ladder: dict,
        current_price: float,
        option_type: str,
    ) -> tuple[float | None, float | None, str]:
        """Pivot target at the 100% OI or 100% OI-change strike, whichever comes first.

        Both peaks are scored as a percentage of their own maximum; the peak that
        sits nearest to spot wins, because price reaches it first.
        """
        grouped = option_chain.get("by_strike", {}) if isinstance(option_chain, dict) else {}
        oi_by_strike: dict[float, float] = {}
        oi_change_by_strike: dict[float, float] = {}

        for raw_strike, sides in grouped.items():
            strike = cls._to_number(raw_strike)
            if strike is None:
                continue
            if option_type == "CE" and strike <= current_price:
                continue
            if option_type == "PE" and strike >= current_price:
                continue
            quote = (sides or {}).get(option_type, {}) or {}
            oi = cls._to_number(quote.get("open_interest"))
            if oi is not None and oi > 0:
                oi_by_strike[strike] = oi
            oi_change = cls._to_number(quote.get("oi_change"))
            if oi_change is not None and oi_change > 0:
                oi_change_by_strike[strike] = oi_change

        peaks = []
        if oi_by_strike:
            peak_strike = max(oi_by_strike, key=lambda key: oi_by_strike[key])
            peaks.append((peak_strike, "OI 100%"))
        if oi_change_by_strike:
            peak_strike = max(oi_change_by_strike, key=lambda key: oi_change_by_strike[key])
            peaks.append((peak_strike, "OI-change 100%"))

        if not peaks:
            return None, None, ""

        strike, source = min(peaks, key=lambda item: abs(item[0] - current_price))
        return strike, cls._ladder_pivot_level(pivot_ladder, strike, option_type), source

    @classmethod
    def _pivot_oi_change_target(
        cls,
        option_chain: dict,
        pivot_ladder: dict,
        current_price: float,
        option_type: str,
    ) -> tuple[float | None, float | None]:
        """Return the pivot level for the strongest directional OI-change strike."""
        grouped = option_chain.get("by_strike", {}) if isinstance(option_chain, dict) else {}
        candidates = []
        for raw_strike, sides in grouped.items():
            strike = cls._to_number(raw_strike)
            if strike is None:
                continue
            if option_type == "CE" and strike <= current_price:
                continue
            if option_type == "PE" and strike >= current_price:
                continue
            quote = (sides or {}).get(option_type, {}) or {}
            oi_change = cls._to_number(quote.get("oi_change"))
            if oi_change is not None and oi_change > 0:
                candidates.append((oi_change, strike))

        if not candidates:
            return None, None

        _, strike = max(candidates, key=lambda item: (item[0], -abs(item[1] - current_price)))
        return strike, cls._ladder_pivot_level(pivot_ladder, strike, option_type)

    @classmethod
    def _option_wall_pivot_target(
        cls,
        option_chain: dict,
        pivot_ladder: dict,
        current_price: float,
        option_type: str,
    ) -> tuple[float | None, float | None]:
        """Return the pivot target for the first directional OI/OI-change strike.

        CE scans strikes upward from spot and PE scans downward from spot.
        The first strike with either positive OI or positive OI change wins;
        OI magnitude is deliberately not used for target selection.
        """
        grouped = option_chain.get("by_strike", {}) if isinstance(option_chain, dict) else {}
        candidates = []
        for raw_strike, sides in grouped.items():
            strike = cls._to_number(raw_strike)
            if strike is None:
                continue
            if option_type == "CE" and strike <= current_price:
                continue
            if option_type == "PE" and strike >= current_price:
                continue
            quote = (sides or {}).get(option_type, {}) or {}
            oi = cls._to_number(quote.get("open_interest"))
            oi_change = cls._to_number(quote.get("oi_change"))
            if (oi is not None and oi > 0) or oi_change is not None:
                candidates.append(strike)

        if not candidates:
            return None, None

        strike = min(candidates) if option_type == "CE" else max(candidates)

        ladder_level = (pivot_ladder or {}).get(str(strike))
        if ladder_level is None:
            ladder_level = (pivot_ladder or {}).get(str(float(strike)))
        if ladder_level is None:
            for ladder_strike, level in (pivot_ladder or {}).items():
                if cls._to_number(ladder_strike) == strike:
                    ladder_level = level
                    break
        if not ladder_level:
            return strike, None

        level_key = "resistance" if option_type == "CE" else "support"
        return strike, cls._to_number(ladder_level.get(level_key))

    @classmethod
    def _same_side_oi_target(
        cls,
        option_chain: dict,
        pivot_ladder: dict,
        current_price: float,
        trade_option_type: str,
    ) -> tuple[float | None, float | None, str]:
        """Use the traded option side's 100% OI/OI-change strike as target."""
        source_side = trade_option_type
        direction = 1 if trade_option_type == "CE" else -1
        grouped = option_chain.get("by_strike", {}) if isinstance(option_chain, dict) else {}
        oi_rows = []
        change_rows = []
        for raw_strike, sides in grouped.items():
            strike = cls._to_number(raw_strike)
            if strike is None or direction * (strike - current_price) <= 0:
                continue
            quote = (sides or {}).get(source_side, {}) or {}
            oi = cls._to_number(quote.get("open_interest"))
            oi_change = cls._to_number(quote.get("oi_change"))
            if oi is not None and oi > 0:
                oi_rows.append((oi, strike))
            if oi_change is not None and oi_change > 0:
                change_rows.append((oi_change, strike))

        if change_rows:
            _, strike = max(change_rows, key=lambda item: (item[0], -abs(item[1] - current_price)))
            source = f"{source_side} OI-change 100%"
        elif oi_rows:
            _, strike = max(oi_rows, key=lambda item: (item[0], -abs(item[1] - current_price)))
            source = f"{source_side} OI 100%"
        else:
            return None, None, ""

        level = cls._ladder_pivot_level(pivot_ladder, strike, trade_option_type)
        return strike, level, source

    @staticmethod
    def _buffer_touch(current_price, level, side, buffer_min=7.0, buffer_max=8.0) -> bool:
        if current_price is None or level is None:
            return False
        distance = float(current_price) - float(level)
        if side == "CE":
            return buffer_min <= distance <= buffer_max
        return buffer_min <= -distance <= buffer_max

    @staticmethod
    def _smc_zone_touch(current_price, zones, side, buffer=8.0) -> bool:
        for block in zones or []:
            zone = block.get("zone", {}) if isinstance(block, dict) else {}
            low, high = zone.get("low"), zone.get("high")
            if low is None or high is None:
                continue
            if side == "CE" and float(low) <= float(current_price) <= float(high) + buffer:
                return True
            if side == "PE" and float(low) - buffer <= float(current_price) <= float(high):
                return True
        return False

    def _resolve_option_side(self, action, signal, market_structure, current_price, snapshot=None) -> str | None:
        """Symmetric CE and PE resolution with clear priority to Trend & AOC State."""
        trend = str(
            (market_structure or {}).get("trend")
            or (snapshot or {}).get("trend")
            or (market_structure or {}).get("micro_momentum")
            or (snapshot or {}).get("trend_1m")
            or ""
        ).upper()
        scenario = signal.get("scenario")
        aoc_scenario = str(signal.get("aoc_scenario") or "").lower()
        bias = str(signal.get("bias") or "").upper()

        # Direct priority based on clear Trend or AOC state
        if trend == "BEARISH" or bias == "BEARISH" or "bearish" in aoc_scenario:
            return "PE"
        if trend == "BULLISH" or "bullish" in aoc_scenario:
            return "CE"

        # Structural S/R Rejection / Bounce evaluation
        bearish = (
            action == "SELL"
            or scenario in ("near_resistance", "above_resistance", "above_red", "bearish_order_block")
            or self._in_range(current_price, signal.get("resistance_range"))
        )
        bullish = (
            action == "BUY"
            or scenario in ("near_support", "below_support", "below_green", "bullish_order_block")
            or self._in_range(current_price, signal.get("support_range"))
        )

        if bearish and not bullish:
            return "PE"
        if bullish and not bearish:
            return "CE"
        if action == "SELL":
            return "PE"
        if action == "BUY":
            return "CE"
        return None

    @staticmethod
    def _broker_signal(current_price, market_structure):
        support = market_structure.get("support")
        resistance = market_structure.get("resistance")
        trend = str(market_structure.get("trend") or "").upper()
        scenario = "neutral"
        if current_price is not None and support is not None and resistance is not None:
            if current_price <= support:
                scenario = "near_support"
            elif current_price >= resistance:
                scenario = "near_resistance"
            elif current_price < support:
                scenario = "below_support"
            elif current_price > resistance:
                scenario = "above_resistance"
            else:
                scenario = "between_support_resistance"
        smc = market_structure.get("smc") or {}
        patterns = market_structure.get("candlestick_patterns") or {}
        if patterns.get("bullish") or smc.get("order_blocks", {}).get("bullish") or smc.get("fair_value_gaps", {}).get("bullish"):
            scenario = "bullish_order_block"
        if patterns.get("bearish") or smc.get("order_blocks", {}).get("bearish") or smc.get("fair_value_gaps", {}).get("bearish"):
            scenario = "bearish_order_block"
        return {
            "scenario": scenario,
            "aoc_scenario": None,
            "bias": "BULLISH" if trend == "BULLISH" else "BEARISH" if trend == "BEARISH" else "NEUTRAL",
            "current_price": current_price,
            "support_strike": support,
            "resistance_strike": resistance,
            "support_range": None,
            "resistance_range": None,
            "market_status": "OPEN",
            "square_off": False,
            "reason": [],
            "aoc_watchlist": {"top": [], "bottom": []},
            "aoc_color_available": False,
        }

    def decide(self, aoc_snapshot: dict, market_structure: dict | None = None) -> Decision:
        current_price = aoc_snapshot.get("spot") or aoc_snapshot.get("current_price")

        underlying = aoc_snapshot.get("underlying") or DEFAULT_SYMBOL
        symbol_cfg = self._get_symbol_config(underlying)
        exchange = aoc_snapshot.get("exchange") or symbol_cfg.get("exchange", "NFO")

        # Dynamic S/R combined levels
        broker_support = (market_structure or {}).get("support")
        broker_resistance = (market_structure or {}).get("resistance")
        combined = self.price_engine.combine_support_resistance(
            current_price,
            aoc_support=aoc_snapshot.get("support_strike"),
            aoc_resistance=aoc_snapshot.get("resistance_strike"),
            market_support=broker_support,
            market_resistance=broker_resistance,
        )

        combined_market_structure = dict(market_structure or {})
        combined_market_structure["support"] = combined.get("support")
        combined_market_structure["resistance"] = combined.get("resistance")

        signal = self._broker_signal(current_price, combined_market_structure)
        price_context = self.price_engine.analyze(aoc_snapshot)

        action = signal.get("action", "WAIT")
        reasons = list(signal.get("reason", []))
        aoc_watchlist = signal.get("aoc_watchlist") or {"top": [], "bottom": []}
        aoc_color_available = bool(signal.get("aoc_color_available"))
        market_status = signal.get("market_status", "OPEN")
        support_range = signal.get("support_range")
        resistance_range = signal.get("resistance_range")
        support_level = signal.get("support_strike") or combined.get("support")
        resistance_level = signal.get("resistance_strike") or combined.get("resistance")
        option_chain = (market_structure or {}).get("option_chain") or {}
        pivot_ladder = (market_structure or {}).get("pivot_ladder") or {}
        unified_chain = (market_structure or {}).get("unified_chain") or {}
        chain_levels = unified_chain.get("levels") or []
        if underlying == "NIFTY" and current_price is not None and pivot_ladder:
            candle_supports = [
                self._to_number((sr or {}).get("support"))
                for sr in pivot_ladder.values()
            ]
            candle_resistances = [
                self._to_number((sr or {}).get("resistance"))
                for sr in pivot_ladder.values()
            ]
            candle_supports = [level for level in candle_supports if level is not None and level <= float(current_price)]
            candle_resistances = [level for level in candle_resistances if level is not None and level >= float(current_price)]
            if candle_supports:
                support_level = max(candle_supports)
            if candle_resistances:
                resistance_level = min(candle_resistances)
        elif underlying != "NIFTY" and current_price is not None and chain_levels:
            chain_supports = [
                float(level["strike"])
                for level in chain_levels
                if level.get("support") is not None and float(level["strike"]) <= float(current_price)
            ]
            chain_resistances = [
                float(level["strike"])
                for level in chain_levels
                if level.get("resistance") is not None and float(level["strike"]) >= float(current_price)
            ]
            if chain_supports:
                support_level = max(chain_supports)
            if chain_resistances:
                resistance_level = min(chain_resistances)
        pattern_state = (market_structure or {}).get("candlestick_patterns") or {}
        smc_state = (market_structure or {}).get("smc") or {}
        oi_support, oi_resistance = self._oi_walls(option_chain, current_price)
        if underlying == "NIFTY" and pivot_ladder:
            oi_support = self._ladder_pivot_level(pivot_ladder, oi_support, "PE") if oi_support is not None else None
            oi_resistance = self._ladder_pivot_level(pivot_ladder, oi_resistance, "CE") if oi_resistance is not None else None
        nifty_entries = {}
        nifty_targets = {}
        if underlying == "NIFTY" and current_price is not None and pivot_ladder:
            for trade_side in ("CE", "PE"):
                nifty_entries[trade_side] = self._nifty_oi_entry(
                    option_chain, pivot_ladder, float(current_price), trade_side
                )
                nifty_targets[trade_side] = self._peak_oi_pivot_target(
                    option_chain, pivot_ladder, float(current_price), trade_side
                )

        # Strict 50-point strike rounding for NIFTY
        strike_step = float(symbol_cfg.get("strike_step", 50) or 50)
        raw_strike = price_context.get("market_strike") or current_price
        try:
            atm_strike = round(float(raw_strike) / strike_step) * strike_step
        except (TypeError, ValueError):
            atm_strike = None

        option_type = self._resolve_option_side(
            action,
            signal,
            combined_market_structure,
            current_price,
            snapshot=aoc_snapshot,
        )

        bullish_smc_setup = self._smc_zone_touch(
            current_price,
            smc_state.get("order_blocks", {}).get("bullish", [])
            + smc_state.get("fair_value_gaps", {}).get("bullish", []),
            "CE",
        )
        bearish_smc_setup = self._smc_zone_touch(
            current_price,
            smc_state.get("order_blocks", {}).get("bearish", [])
            + smc_state.get("fair_value_gaps", {}).get("bearish", []),
            "PE",
        )
        bullish_pivot_setup = self._buffer_touch(current_price, support_level, "CE")
        bearish_pivot_setup = self._buffer_touch(current_price, resistance_level, "PE")
        bullish_oi_confirmation = self._buffer_touch(current_price, oi_support, "CE")
        bearish_oi_confirmation = self._buffer_touch(current_price, oi_resistance, "PE")
        if underlying == "NIFTY" and nifty_entries:
            ce_entry = nifty_entries.get("CE", (None, "", None, None))
            pe_entry = nifty_entries.get("PE", (None, "", None, None))
            bullish_pivot_setup = self._buffer_touch(current_price, ce_entry[2], "CE")
            bearish_pivot_setup = self._buffer_touch(current_price, pe_entry[2], "PE")
            bullish_oi_confirmation = (
                ce_entry[0] is not None and ce_entry[2] is not None
            )
            bearish_oi_confirmation = (
                pe_entry[0] is not None and pe_entry[2] is not None
            )
        bullish_setup = bullish_smc_setup or (bullish_pivot_setup and bullish_oi_confirmation)
        bearish_setup = bearish_smc_setup or (bearish_pivot_setup and bearish_oi_confirmation)
        bullish_trigger = bool(pattern_state.get("bullish"))
        bearish_trigger = bool(pattern_state.get("bearish"))
        trend = str((market_structure or {}).get("trend") or "").upper()
        micro_momentum = str((market_structure or {}).get("micro_momentum") or "").upper()
        bullish_trend_alignment = trend == "BULLISH" and micro_momentum == "BULLISH"
        bearish_trend_alignment = trend == "BEARISH" and micro_momentum == "BEARISH"
        if option_type is None:
            if bullish_setup and bullish_trigger and not bearish_setup:
                option_type = "CE"
            elif bearish_setup and bearish_trigger and not bullish_setup:
                option_type = "PE"

        entry_strike = atm_strike
        if underlying == "NIFTY" and option_type in nifty_entries:
            selected_strike, _, entry_level, _ = nifty_entries[option_type]
            if selected_strike is not None and entry_level is not None:
                entry_strike = selected_strike

        oi_pivot_setup = (
            option_type == "CE" and bullish_pivot_setup and bullish_oi_confirmation
        ) or (
            option_type == "PE" and bearish_pivot_setup and bearish_oi_confirmation
        )

        sr_tolerance = 8.0
        sr_trigger = False
        if support_level is not None and current_price is not None:
            if abs(float(current_price) - float(support_level)) <= sr_tolerance:
                sr_trigger = True
        if resistance_level is not None and current_price is not None:
            if abs(float(current_price) - float(resistance_level)) <= sr_tolerance:
                sr_trigger = True

        entry_confirmed = (
            option_type == "CE"
            and bullish_setup
            and (bullish_trigger or bullish_trend_alignment)
        ) or (
            option_type == "PE"
            and bearish_setup
            and (bearish_trigger or bearish_trend_alignment)
        )
        # Check 3:15 PM IST Auto Square-off cutoff
        time_cutoff_reached = self._is_auto_square_off_time(underlying)
        square_off = bool(signal.get("square_off")) or time_cutoff_reached

        if time_cutoff_reached:
            reasons.append("Intraday 3:15 PM cutoff reached: auto square-off triggered.")

        # No trade conditions or intraday square-off
        if square_off or option_type is None or action == "no_price" or market_status == "CLOSED":
            decision = Decision(
                action="SQUARE_OFF" if square_off else ("NO_TRADE" if action in ("no_price", "WAIT", "HOLD") else action),
                option_type=None,
                strike=atm_strike,
                quantity=0,
                confidence="LOW",
                reasons=reasons,
                underlying=underlying,
                exchange=exchange,
                support=support_level,
                resistance=resistance_level,
                support_range=support_range,
                resistance_range=resistance_range,
                market_status=market_status,
                square_off=square_off,
            )
            log.info("Decision: %s (no trade, square_off=%s)", decision.action, square_off)
            return decision

        if not entry_confirmed:
            reasons.append(
                "Waiting for S/R/SMC setup with matching candle or aligned trend confirmation."
            )
            return Decision(
                action="NO_TRADE",
                option_type=None,
                strike=atm_strike,
                quantity=0,
                confidence="LOW",
                reasons=reasons,
                underlying=underlying,
                exchange=exchange,
                support=support_level,
                resistance=resistance_level,
                support_range=support_range,
                resistance_range=resistance_range,
                market_status=market_status,
            )

        if not bullish_trigger and not bearish_trigger:
            reasons.append("Candlestick trigger absent; aligned higher and micro trend accepted.")

        if option_type == "PE":
            reasons.append("Bearish bias: buying PE (resistance rejection / bearish trend)")
        else:
            reasons.append("Bullish bias: buying CE (support bounce / bullish trend)")

        quantity = symbol_cfg.get("lot_size", 25)

        # Risk check evaluation
        risk_result = self.risk_manager.evaluate(quantity)
        reasons.extend(getattr(risk_result, "reasons", []))
        risk_approved = getattr(risk_result, "approved", False)

        sl_points = float(symbol_cfg.get("sl_points", getattr(settings, "sl_points", 25.0)))
        target_points = float(symbol_cfg.get("target_points", getattr(settings, "target_points", 30.0)))
        sl_points = max(20.0, min(30.0, sl_points))

        index_sl = index_target = None
        if current_price is not None:
            if option_type == "CE":
                index_sl = current_price - sl_points
                if underlying == "NIFTY" and "CE" in nifty_targets:
                    oi_target_strike, oi_target, oi_target_source = nifty_targets["CE"]
                    if oi_target is not None and oi_target > float(current_price):
                        index_target = oi_target
                        reasons.append(
                            f"NIFTY target: CE same-side {oi_target_source} strike {oi_target_strike}; "
                            f"target R={index_target}"
                        )
                else:
                    oi_target_strike, oi_target, oi_target_source = self._same_side_oi_target(
                        option_chain, pivot_ladder, float(current_price), option_type
                    )
                    if oi_pivot_setup and oi_target is not None:
                        index_target = oi_target
                        reasons.append(
                            f"NIFTY target: same-side CE {oi_target_source} strike {oi_target_strike} pivot {index_target}"
                        )
                peak_strike, peak_target, peak_source = nifty_targets.get("CE") or self._peak_oi_pivot_target(
                    option_chain, pivot_ladder, float(current_price), option_type
                )
                wall_strike, wall_pivot_target = self._option_wall_pivot_target(
                    option_chain, pivot_ladder, float(current_price), option_type
                )
                oi_change_strike, pivot_target = self._pivot_oi_change_target(
                    option_chain, pivot_ladder, float(current_price), option_type
                )
                candidates = [peak_target] if underlying == "NIFTY" else [
                    peak_target,
                    pivot_target,
                    wall_pivot_target,
                    self._pivot_target(
                        market_structure or {}, float(current_price), option_type,
                        pivot_ladder if underlying == "NIFTY" else None,
                    ),
                    oi_resistance,
                    resistance_level,
                ]
                if underlying == "NIFTY":
                    candidates = [
                        candidate for candidate in candidates
                        if self._within_oi_pivot_cutoff(
                            candidate, pivot_ladder, float(current_price), option_type
                        )
                    ]
                if index_target is None:
                    index_target = float(current_price) + (1.0 * target_points)
                    reasons.append(
                        f"NIFTY fallback target: CE configured target_points={target_points} -> {index_target}"
                    )
                if peak_target is not None:
                    reasons.append(f"CE {peak_source} strike {peak_strike}: pivot R target {index_target}")
                if wall_pivot_target is not None:
                    reasons.append(f"First CE OI/OI-change strike {wall_strike}: pivot R target {index_target}")
                if pivot_target is not None:
                    reasons.append(f"Highest CE OI-change strike {oi_change_strike}: pivot R target {index_target}")
            elif option_type == "PE":
                index_sl = current_price + sl_points
                if underlying == "NIFTY" and "PE" in nifty_targets:
                    oi_target_strike, oi_target, oi_target_source = nifty_targets["PE"]
                    if oi_target is not None and oi_target < float(current_price):
                        index_target = oi_target
                        reasons.append(
                            f"NIFTY target: PE same-side {oi_target_source} strike {oi_target_strike}; "
                            f"target S={index_target}"
                        )
                else:
                    oi_target_strike, oi_target, oi_target_source = self._same_side_oi_target(
                        option_chain, pivot_ladder, float(current_price), option_type
                    )
                    if oi_pivot_setup and oi_target is not None:
                        index_target = oi_target
                        reasons.append(
                            f"NIFTY target: same-side PE {oi_target_source} strike {oi_target_strike} pivot {index_target}"
                        )
                peak_strike, peak_target, peak_source = nifty_targets.get("PE") or self._peak_oi_pivot_target(
                    option_chain, pivot_ladder, float(current_price), option_type
                )
                wall_strike, wall_pivot_target = self._option_wall_pivot_target(
                    option_chain, pivot_ladder, float(current_price), option_type
                )
                oi_change_strike, pivot_target = self._pivot_oi_change_target(
                    option_chain, pivot_ladder, float(current_price), option_type
                )
                candidates = [peak_target] if underlying == "NIFTY" else [
                    peak_target,
                    pivot_target,
                    wall_pivot_target,
                    self._pivot_target(
                        market_structure or {}, float(current_price), option_type,
                        pivot_ladder if underlying == "NIFTY" else None,
                    ),
                    oi_support,
                    support_level,
                ]
                if underlying == "NIFTY":
                    candidates = [
                        candidate for candidate in candidates
                        if self._within_oi_pivot_cutoff(
                            candidate, pivot_ladder, float(current_price), option_type
                        )
                    ]
                if index_target is None:
                    index_target = float(current_price) - (1.0 * target_points)
                    reasons.append(
                        f"NIFTY fallback target: PE configured target_points={target_points} -> {index_target}"
                    )
                if peak_target is not None:
                    reasons.append(f"PE {peak_source} strike {peak_strike}: pivot S target {index_target}")
                if wall_pivot_target is not None:
                    reasons.append(f"First PE OI/OI-change strike {wall_strike}: pivot S target {index_target}")
                if pivot_target is not None:
                    reasons.append(f"Highest PE OI-change strike {oi_change_strike}: pivot S target {index_target}")

        if index_target is not None:
            reasons.append(f"Targeting next SR level: {index_target}")

        # One-sided fixed stop loss risk only. Do not use a paired risk model here.
        if index_sl is not None and current_price is not None:
            reasons.append(f"Fixed SL risk: {sl_points} points")

        decision = Decision(
            action="BUY" if risk_approved else "NO_TRADE",
            option_type=option_type if risk_approved else None,
            strike=entry_strike,
            quantity=quantity if risk_approved else 0,
            confidence="HIGH" if risk_approved else "LOW",
            reasons=reasons,
            risk_approved=risk_approved,
            underlying=underlying,
            exchange=exchange,
            support=support_level,
            resistance=resistance_level,
            support_range=support_range,
            resistance_range=resistance_range,
            trade_type=f"BUY_{option_type}" if risk_approved else None,
            index_entry=current_price if risk_approved else None,
            index_sl=index_sl if risk_approved else None,
            index_target=index_target if risk_approved else None,
            sl_points=sl_points,
            target_points=target_points,
            market_status=market_status,
        )

        log.info(
            "Decision: action=%s trade=%s option=%s strike=%s qty=%s risk_approved=%s",
            decision.action,
            decision.trade_type,
            decision.option_type,
            decision.strike,
            decision.quantity,
            decision.risk_approved,
        )
        return decision