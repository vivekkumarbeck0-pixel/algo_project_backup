"""Live paper-trading session: broker -> analyze -> decide -> monitor -> dashboard.

Runs the full broker market-data pipeline on a fixed interval (default:
every 60 seconds), enforces the daily trade limit (max 3 trades/day),
tracks the open position against its index-point SL/Target, and prints
a live P&L dashboard, a clear message whenever a trade closes, and a
running end-of-day summary table.
"""

import math
import json
import sys
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from angel_one.instrument_reader import InstrumentReader
from angel_one.login import AngelOneLogin
from angel_one.market_data import MarketDataFetcher
from config import settings, SYMBOL_REGISTRY, DEFAULT_SYMBOL
from engine.decision_engine import DecisionEngine
from engine.daily_state import DailyStateStore
from engine.order_manager import ExecutionMode, OrderManager
from engine.position_tracker import Position, PositionTracker
from engine.price_engine import PriceEngine
from engine.risk_manager import RiskManager
from engine.training_data import TrainingDataStore
from engine.pattern_engine import detect_patterns
from engine.smc_engine import detect_smc
from engine.pivot_ladder import build_connected_ladder
from logger import get_logger

log = get_logger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class LivePaperTradingSession:
    """Owns one broker session and one paper-trading book for a single run.

    The traded symbol is fixed by the runner, keeping NIFTY and CRUDEOIL
    market data, pivots, option chains, and state isolated.
    """

    def __init__(self, default_symbol: str = DEFAULT_SYMBOL):
        self.default_symbol = default_symbol
        self._active_symbol = default_symbol
        self._fixed_symbol = False

        self.instrument_reader = InstrumentReader()
        self.instrument_reader.load()

        try:
            self.client = AngelOneLogin.connect_from_env()
        except Exception as exc:
            log.warning("Broker connection unavailable: %s", exc)
            self.client = None
        self.market_data = MarketDataFetcher(client=self.client)

        self.state_store = DailyStateStore(self._state_file())
        self.tracker = PositionTracker(on_change=self._persist_state)
        self.risk_manager = RiskManager(self.tracker)
        self.decision_engine = DecisionEngine(
            self.risk_manager, price_engine=PriceEngine()
        )
        execution_mode = ExecutionMode(settings.execution_mode)
        self.order_manager = OrderManager(
            self.tracker,
            mode=execution_mode,
            broker_client=self.client if execution_mode == ExecutionMode.REAL else None,
        )

        self._underlying_cache: dict[str, dict] = {}  # symbol -> resolved instrument, cached
        self._market_structure_cache: dict[str, dict] = {}
        self._last_metrics: dict[str, float | None] = {
            "india_vix": None,
            "option_iv": None,
            "crude_volatility": None,
        }
        self._last_metrics_refresh = 0.0
        self._metrics_updated_at: datetime | None = None
        self._last_structural_refresh = 0.0
        self._last_option_chain_refresh = 0.0
        self._last_atm_strike = None
        self._startup_trade_locked_until = 0.0
        self._last_live_tick_price = None
        self._last_trade_ready_at = 0.0
        self.training_data = TrainingDataStore()
        self._pending_training_records: dict[int, str] = {}
        self._pivot_cache: dict[str, dict] = {}
        self._clear_startup_state()
        self._bootstrap_pivots(default_symbol)
        self._load_state()

    def _clear_startup_state(self) -> None:
        """Purge stale signals and cached state after a restart or reconnect."""
        self._underlying_cache = {}
        self._market_structure_cache = {}
        self._pivot_cache = {}
        self._last_metrics = {"india_vix": None, "option_iv": None, "crude_volatility": None}
        self._last_metrics_refresh = 0.0
        self._metrics_updated_at = None
        self._last_structural_refresh = 0.0
        self._last_option_chain_refresh = 0.0
        self._last_atm_strike = None
        self._startup_trade_locked_until = time.monotonic() + 90.0
        self._last_live_tick_price = None
        self._last_trade_ready_at = 0.0

    def _has_fresh_live_confirmation(self) -> bool:
        """Only allow a trade after a fresh, valid live quote is available."""
        if time.monotonic() < self._startup_trade_locked_until:
            log.info("Trade blocked: waiting for fresh live tick confirmation after startup.")
            return False

        token = None
        exchange = None
        try:
            symbol_cfg = SYMBOL_REGISTRY.get(self._active_symbol, SYMBOL_REGISTRY[DEFAULT_SYMBOL])
            underlying = self._resolve_underlying(self._active_symbol)
            token = underlying.get("token")
            exchange = symbol_cfg.get("underlying_exchange")
        except Exception:
            return False

        if token is None:
            return False

        live = self.market_data._live_quote(token, exchange)
        if live is None:
            return False

        age = None
        for key in ("time", "timestamp", "last_trade_time", "tick_time"):
            value = live.get(key)
            if value is not None:
                try:
                    age = time.time() - float(value)
                except (TypeError, ValueError):
                    age = None
                break
        if age is not None and age > 15.0:
            return False

        try:
            current = float(live.get("ltp"))
        except (TypeError, ValueError):
            return False

        if not math.isfinite(current) or current <= 0:
            return False

        self._last_live_tick_price = current
        self._last_trade_ready_at = time.monotonic()
        return True

    def _session_symbol(self) -> str:
        return self._active_symbol

    def _candle_intervals(self) -> tuple[str, ...]:
        return ("FIVE_MINUTE", "ONE_MINUTE")

    def _sr_settings(self) -> dict:
        return {"sr_range_offset": 50.0, "sr_range_tolerance": 5.0}

    def _market_hours(self) -> tuple[dt_time, dt_time]:
        return dt_time(9, 15), dt_time(15, 30)

    def _square_off_time(self) -> dt_time | None:
        """Return the last-entry/forced-exit cutoff for this session."""
        if str(self._session_symbol()).upper() == "CRUDEOIL":
            # MCX closes at 23:30. Exit before the final five minutes so an
            # option LTP is still normally available for paper square-off.
            return dt_time(23, 25)
        if self._market_hours()[1] > dt_time(15, 30):
            return None
        try:
            hour, minute = str(settings.square_off_time).split(":")
            return dt_time(int(hour), int(minute))
        except (ValueError, TypeError, AttributeError):
            return dt_time(15, 15)

    def _square_off_due(self) -> bool:
        cutoff = self._square_off_time()
        return cutoff is not None and datetime.now(IST).time() >= cutoff

    def _state_file(self) -> str:
        return settings.daily_state_file

    def _dashboard_metric_labels(self) -> tuple[str, str]:
        return "India VIX", "Option IV"

    def _strike_step(self) -> float:
        cfg = SYMBOL_REGISTRY.get(self._session_symbol(), SYMBOL_REGISTRY[DEFAULT_SYMBOL])
        try:
            return float(cfg.get("strike_step", 50)) or 50.0
        except (TypeError, ValueError):
            return 50.0

    def _align_strike(self, level):
        """Round a level onto the symbol's strike grid (NIFTY: 50 points)."""
        if not isinstance(level, (int, float)):
            return None
        step = self._strike_step()
        return round(float(level) / step) * step

    @staticmethod
    def _format_exact_sr(level) -> str:
        try:
            numeric = float(level)
        except (TypeError, ValueError):
            return "N/A"
        return f"{numeric:.4f}"

    def _format_sr_range(self, level) -> str:
        return "--"

    def _bootstrap_pivots(self, symbol: str) -> None:
        """Load cached pivots immediately, then refresh them from broker OHLC."""
        cached = PriceEngine.load_pivot_cache(symbol)
        if cached:
            self._pivot_cache[symbol] = cached
            pivots = cached.get("pivots", {})
            log.info("[SYSTEM] Pivot S/R Loaded Successfully: S1=%s, R1=%s", pivots.get("s1"), pivots.get("r1"))

        if not self.client:
            return
        try:
            symbol_cfg = SYMBOL_REGISTRY[symbol]
            underlying = self._resolve_underlying(symbol)
            structure = self.market_data.fetch_market_structure_levels(
                underlying.get("token"), interval="ONE_DAY", days=5,
                exchange=symbol_cfg["underlying_exchange"], lookback=60,
                underlying_name=underlying.get("name") or symbol,
            )
            pivots = structure.get("pivot_levels") or {}
            if all(pivots.get(key) is not None for key in ("s1", "r1", "source_high", "source_low", "source_close")):
                cached = PriceEngine.save_pivot_cache(
                    symbol, pivots["source_high"], pivots["source_low"], pivots["source_close"]
                )
                self._pivot_cache[symbol] = cached
                log.info("[SYSTEM] Pivot S/R Loaded Successfully: S1=%s, R1=%s", pivots.get("s1"), pivots.get("r1"))
        except Exception as exc:
            log.warning("Pivot bootstrap API refresh failed; using local cache: %s", exc)

    @staticmethod
    def _to_float(value):
        try:
            return float(value) if value is not None and str(value).strip() else None
        except (TypeError, ValueError):
            return None

    def _normalize_pivot_structure(self, symbol: str, structure: dict, spot) -> dict:
        """Keep pivot levels numeric, present, and on the correct side of spot."""
        pivots = dict(structure.get("pivot_levels") or {})
        support = self._to_float(structure.get("support"))
        resistance = self._to_float(structure.get("resistance"))
        if support is None:
            support = self._to_float(pivots.get("s1"))
        if resistance is None:
            resistance = self._to_float(pivots.get("r1"))
        structure["pivot_levels"] = pivots
        structure["support"] = support
        structure["resistance"] = resistance
        structure["pivot_support"] = support
        structure["pivot_resistance"] = resistance
        structure["sr_source"] = structure.get("sr_source") or "broker_classic_pivot"
        return structure

    def _remember_metrics(self, metrics: dict) -> dict:
        """Keep the last usable VIX/IV so a throttled tick never shows 0.00."""
        for key in ("india_vix", "option_iv", "crude_volatility"):
            value = metrics.get(key)
            if isinstance(value, (int, float)) and value > 0:
                self._last_metrics[key] = float(value)
            else:
                metrics[key] = self._last_metrics.get(key)
        return metrics

    def _live_metrics(self, symbol: str, market_structure: dict) -> dict:
        """One broker fetch of India VIX + Option IV (called only when due)."""
        iv = self._option_iv(symbol)
        if iv is None:
            one_minute = (market_structure.get("candle_structures") or {}).get("ONE_MINUTE", {})
            iv = self.market_data.calculate_iv_estimate(one_minute.get("candles", []))
        return {"india_vix": self._india_vix(), "option_iv": iv}

    def _refresh_metrics(self, symbol: str, market_structure: dict | None = None, spot=None) -> dict:
        """India VIX / Option IV: exactly one broker refresh per 5 minutes.

        Between refreshes (and on any failure) the last valid values are
        returned unchanged, so the dashboard never blanks them out.
        """
        now = time.monotonic()
        has_values = any(
            self._last_metrics.get(key) is not None
            for key in ("india_vix", "option_iv", "crude_volatility")
        )
        if has_values and (now - self._last_metrics_refresh) < settings.metrics_refresh_interval_seconds:
            return dict(self._last_metrics)

        if self._last_atm_strike is None and spot is not None:
            self._last_atm_strike = self._nearest_strike(symbol, spot)

        try:
            metrics = self._live_metrics(symbol, market_structure or {})
        except Exception as exc:
            log.warning("VIX/IV refresh failed; keeping last known values: %s", exc)
        else:
            self._remember_metrics(metrics)
            # This timestamp means the last successful metrics refresh, not
            # the last time a value changed. VIX/IV may legitimately remain
            # unchanged across refreshes.
            if any(
                self._last_metrics.get(key) is not None
                for key in ("india_vix", "option_iv", "crude_volatility")
            ):
                self._metrics_updated_at = datetime.now()

        if any(
            self._last_metrics.get(key) is not None
            for key in ("india_vix", "option_iv", "crude_volatility")
        ):
            self._last_metrics_refresh = now
        else:
            # Nothing valid yet - retry at a controlled interval, not every
            # polling tick, otherwise a broker outage becomes an API burst.
            self._last_metrics_refresh = (
                now - settings.metrics_refresh_interval_seconds
                + settings.metrics_failure_retry_seconds
            )

        return dict(self._last_metrics)

    def _fallback_metrics(self, symbol: str) -> dict:
        """Last-known broker metrics, used when a live fetch is unavailable."""
        return {
            "india_vix": self._last_metrics.get("india_vix"),
            "option_iv": self._last_metrics.get("option_iv"),
            "crude_volatility": self._last_metrics.get("crude_volatility"),
            "trend": "NEUTRAL",
            "micro_momentum": "NEUTRAL",
        }

    def _fallback_market_structure(self, snapshot: dict) -> dict:
        """Fallback structure that keeps cached or spot-relative S/R available."""
        spot = snapshot.get("spot") or snapshot.get("market_strike")
        metrics = self._fallback_metrics(self._session_symbol())
        cached_pivots = (self._pivot_cache.get(self._session_symbol()) or {}).get("pivots", {})
        if cached_pivots:
            log.info("[SYSTEM] Pivot S/R Loaded Successfully: S1=%s, R1=%s", cached_pivots.get("s1"), cached_pivots.get("r1"))
        else:
            log.warning("Pivot S/R unavailable because both historical OHLC and live spot are unavailable")
        fallback = {
            "current_price": spot,
            "support": cached_pivots.get("s1"),
            "resistance": cached_pivots.get("r1"),
            "pivot_levels": cached_pivots,
            "pivot_support": cached_pivots.get("s1"),
            "pivot_resistance": cached_pivots.get("r1"),
            "candle_structures": {},
            "trend": metrics["trend"],
            "micro_momentum": metrics["micro_momentum"],
            "india_vix": metrics["india_vix"],
            "option_iv": metrics["option_iv"],
            "broker_connected": False,
            "market_open": self._is_market_open(),
            "entry_confirmed": False,
        }
        fallback.update(self._sr_settings())
        return self._normalize_pivot_structure(self._session_symbol(), fallback, spot)

    def _load_state(self) -> None:
        saved_state = self.state_store.load(datetime.now().date().isoformat())
        if not saved_state:
            return
        self.tracker.restore_state(saved_state.get("positions", []))
        today = datetime.now(IST).date()
        stale_positions = [
            position for position in self.tracker.open_positions()
            if position.opened_at.date() != today or self._square_off_due()
        ]
        for position in stale_positions:
            exit_price = self._option_ltp_lookup(position)
            if exit_price is None:
                # Never carry a stale position into a new session. Entry price
                # is a conservative paper fallback when the broker is offline.
                exit_price = position.entry_price
                log.warning(
                    "Closing stale %s position #%s at entry price: no exit LTP",
                    position.symbol, position.trade_number,
                )
            self.tracker.close_position(position, exit_price, reason="SESSION_ROLLOVER")
        if stale_positions:
            self._persist_state()
            log.warning("Closed %d stale position(s) from a previous trading day", len(stale_positions))
        log.info("Restored %d positions from today's persisted trading state", self.tracker.trades_today_count())

    def _persist_state(self) -> None:
        self.state_store.save(
            {
                "date": datetime.now().date().isoformat(),
                "positions": self.tracker.export_state(),
                "trades_today": self.tracker.trades_today_count(),
                "consecutive_stop_losses": self.tracker.consecutive_stop_losses(),
                "daily_realized_pnl": self.tracker.daily_realized_pnl(),
            }
        )

    def set_execution_mode(self, mode: str) -> None:
        """Change the session execution mode without restarting the process."""
        selected_mode = ExecutionMode(mode.upper())
        self.order_manager = OrderManager(
            self.tracker,
            mode=selected_mode,
            broker_client=self.client if selected_mode == ExecutionMode.REAL else None,
        )

    # ------------------------------------------------------------
    # UNDERLYING TOKEN RESOLUTION
    # ------------------------------------------------------------

    def _resolve_underlying(self, symbol: str):
        cached = self._underlying_cache.get(symbol)
        if cached:
            return cached

        resolved = self.instrument_reader.resolve_underlying(symbol)
        if not resolved:
            raise RuntimeError(f"Unable to resolve underlying token for {symbol}")

        self._underlying_cache[symbol] = resolved
        return resolved

    def _option_ltp_lookup(self, position: Position):
        cfg = SYMBOL_REGISTRY.get(position.symbol, SYMBOL_REGISTRY[DEFAULT_SYMBOL])
        info = self.instrument_reader.find_option_token(
            underlying=position.symbol,
            strike=position.strike,
            right=position.option_type,
            instrument_type=cfg["option_instrumenttype"],
            exchange=cfg["exchange"],
        )
        if not info:
            return None
        return self.market_data.fetch_latest_price(
            info.get("token"), exchange=info.get("exch_seg"), tradingsymbol=info.get("symbol")
        )

    # ------------------------------------------------------------
    # ONE TICK
    # ------------------------------------------------------------

    def run_once(self):
        symbol = self.default_symbol
        self._active_symbol = symbol
        snapshot = {
            "underlying": symbol,
            "exchange": SYMBOL_REGISTRY[symbol]["exchange"],
        }
        if not self._is_market_open():
            snapshot.update({
                **self._fallback_metrics(symbol),
                "market_status": "CLOSED",
            })
            self._print_dashboard(snapshot, symbol, None, None, None)
            self._persist_state()
            return

        symbol_cfg = SYMBOL_REGISTRY.get(symbol, SYMBOL_REGISTRY[DEFAULT_SYMBOL])
        underlying = self._resolve_underlying(symbol)
        underlying_exchange = symbol_cfg["underlying_exchange"]

        index_price = self.market_data.fetch_latest_price(
            underlying.get("token"), exchange=underlying_exchange, tradingsymbol=underlying.get("symbol")
        )

        market_structure = self._market_structure(symbol, underlying, underlying_exchange)
        if index_price is None and market_structure:
            index_price = market_structure.get("current_price")

        # Populate the broker-only snapshot.
        fallback_metrics = self._fallback_metrics(symbol)
        metrics = self._refresh_metrics(symbol, market_structure, spot=index_price)
        snapshot["vix"] = metrics["india_vix"]
        snapshot["iv"] = metrics["option_iv"]
        snapshot["trend"] = market_structure.get("trend") or fallback_metrics["trend"]
        snapshot["trend_1m"] = market_structure.get("micro_momentum") or fallback_metrics["micro_momentum"]
        snapshot["market_status"] = "OPEN" if (self._is_market_open() or market_structure.get("market_open")) else "CLOSED"

        snapshot["market_strike"] = self._nearest_strike(symbol, index_price)
        snapshot["spot"] = index_price
        snapshot["current_price"] = index_price
        snapshot["market_status"] = "OPEN"

        # Monitor the existing open position first (SL/Target on index points).
        closed = self._check_all_exits()
        if self._square_off_due():
            closed.extend(self._close_all_positions("TIME_EXIT"))
        else:
            reversal_trend = (
                market_structure.get("trend")
                or snapshot.get("trend")
                or market_structure.get("micro_momentum")
                or snapshot.get("trend_1m")
            )
            closed.extend(self._check_reversal_exits(reversal_trend))
        for position in closed:
            self._record_training_outcome(position)
            self._print_trade_closed(position)
            self.print_daily_summary()

        # Only look for a new trade when flat and under the daily trade limit.
        decision = None
        max_trades = self.risk_manager.limits.max_trades_per_day
        under_limit = max_trades is None or self.tracker.trades_today_count() < max_trades
        scanning_open = datetime.now().strftime("%H:%M") >= settings.trade_start_time
        if not self.tracker.open_positions() and under_limit and scanning_open and not self._square_off_due():
            if not self._has_fresh_live_confirmation():
                log.info("Trade blocked: waiting for a fresh live tick breakout before entry.")
                self._print_dashboard(snapshot, symbol, index_price, decision, market_structure)
                self._persist_state()
                return
            decision = self.decision_engine.decide(snapshot, market_structure=market_structure)
            decision.training_record_id = self.training_data.record_signal(
                symbol=symbol,
                snapshot=snapshot,
                market_structure=market_structure,
                decision=decision,
            )
            clear_trend = str(
                market_structure.get("trend")
                or snapshot.get("trend")
                or market_structure.get("micro_momentum")
                or snapshot.get("trend_1m")
                or ""
            ).upper() in ("BULLISH", "BEARISH")
            if (
                decision.action in ("BUY", "SELL")
                and decision.risk_approved
                and (market_structure.get("entry_confirmed") or clear_trend)
            ):
                self._open_trade(decision, index_price, snapshot, market_structure)

        self._print_dashboard(snapshot, symbol, index_price, decision, market_structure)
        self._persist_state()

    def _market_structure(self, symbol, underlying, exchange):
        now = time.monotonic()
        if now - self._last_structural_refresh < settings.structural_refresh_interval_seconds:
            cached = self._market_structure_cache.get(symbol, {})
            if cached:
                cached["market_open"] = self._is_market_open()
                self._refresh_option_chain(symbol, cached)
                spot = cached.get("current_price")
                return self._normalize_pivot_structure(symbol, cached, spot)
            return cached

        previous = self._market_structure_cache.get(symbol, {}) or {}
        try:
            structures = {
                interval: self.market_data.fetch_market_structure_levels(
                    underlying.get("token"), interval=interval, days=5,
                    exchange=exchange, lookback=60 if interval != "ONE_MINUTE" else 12,
                    underlying_name=underlying.get("name") or symbol,
                )
                for interval in self._candle_intervals()
            }
            master = structures[self._candle_intervals()[0]]
            micro = structures["ONE_MINUTE"]
            
            master_trend = master.get("trend")
            micro_trend = micro.get("trend")
            
            master["micro_momentum"] = micro_trend
            master["entry_confirmed"] = bool(master_trend and micro_trend) and master_trend == micro_trend
            
            current_price = master.get("current_price") or micro.get("current_price")
            # Use exact broker-candle pivot levels for the active market price.
            master["support"] = master.get("pivot_support")
            master["resistance"] = master.get("pivot_resistance")
            if master.get("support") is None and previous.get("support") is not None:
                master["support"] = previous.get("support")
            if master.get("resistance") is None and previous.get("resistance") is not None:
                master["resistance"] = previous.get("resistance")
            cached_pivots = (self._pivot_cache.get(symbol) or {}).get("pivots", {})
            if master.get("support") is None:
                master["support"] = cached_pivots.get("s1")
                master["pivot_support"] = cached_pivots.get("s1")
            if master.get("resistance") is None:
                master["resistance"] = cached_pivots.get("r1")
                master["pivot_resistance"] = cached_pivots.get("r1")
            if not master.get("pivot_levels") and cached_pivots:
                master["pivot_levels"] = cached_pivots
            master["sr_source"] = "broker_classic_pivot"
            master["candle_structures"] = structures
            analysis_candles = micro.get("candles") or master.get("candles") or []
            master["candlestick_patterns"] = detect_patterns(analysis_candles)
            master["smc"] = detect_smc(analysis_candles)
            master["option_chain"] = self.market_data.fetch_option_chain(
                symbol, center=current_price, radius=10
            ) if current_price is not None else {"rows": [], "by_strike": {}, "error": "No current price"}
            master["unified_chain"] = PriceEngine.build_unified_chain(
                master["option_chain"],
                master.get("pivot_levels") or {},
                current_price,
                strike_step=self._strike_step(),
                radius=2,
            )
            chain_levels = master["unified_chain"].get("levels") or []
            if isinstance(current_price, (int, float)) and chain_levels:
                active_supports = [
                    float(level["support"])
                    for level in chain_levels
                    if level.get("support") is not None and float(level["support"]) <= float(current_price)
                ]
                active_resistances = [
                    float(level["resistance"])
                    for level in chain_levels
                    if level.get("resistance") is not None and float(level["resistance"]) > float(current_price)
                ]
                if active_supports and active_resistances:
                    master["support"] = max(active_supports)
                    master["resistance"] = min(active_resistances)
                    master["pivot_support"] = master["support"]
                    master["pivot_resistance"] = master["resistance"]
                    master["sr_source"] = "broker_oc_connected_ladder"
            chain_strikes = [
                float(strike)
                for strike in (master["option_chain"].get("by_strike") or {})
            ]
            master["pivot_ladder"] = build_connected_ladder(
                chain_strikes,
                master.get("pivot_levels") or {},
            )
            master.update(self._sr_settings())
            self._last_atm_strike = self._nearest_strike(symbol, master.get("current_price"))
            master.update(self._refresh_metrics(symbol, master))
            master["broker_connected"] = bool(self.market_data.is_connected() and master.get("current_price") is not None)
            master["market_open"] = self._is_market_open()

            if master.get("current_price") is None:
                fallback = self._fallback_market_structure({"spot": None})
                for key, value in fallback.items():
                    if key not in master or master.get(key) is None or master.get(key) == {}:
                        master[key] = value

            master = self._normalize_pivot_structure(symbol, master, current_price)

            self._market_structure_cache[symbol] = master
            self._last_structural_refresh = now
            return master
        except Exception as exc:
            log.warning("Market structure fetch failed this tick: %s", exc)

            # Reuse the last good structure; never blank out S/R or metrics.
            cached = self._market_structure_cache.get(symbol, {}) or self._fallback_market_structure({"spot": None})
            cached["broker_connected"] = False
            cached["market_open"] = self._is_market_open()
            cached.update(self._remember_metrics({
                "india_vix": cached.get("india_vix"),
                "option_iv": cached.get("option_iv"),
            }))
            return cached

    def _option_chain_refresh_interval(self) -> float:
        """Refresh full option quotes on the same five-minute metrics cycle."""
        return settings.metrics_refresh_interval_seconds

    def _refresh_option_chain(self, symbol: str, structure: dict) -> None:
        now = time.monotonic()
        if (now - self._last_option_chain_refresh) < self._option_chain_refresh_interval():
            return

        # Advance the gate before the network call as well. If Angel One is
        # unavailable, do not retry the same full quote request every poll.
        self._last_option_chain_refresh = now

        current_price = structure.get("current_price")
        if not isinstance(current_price, (int, float)):
            return

        option_chain = self.market_data.fetch_option_chain(symbol, center=current_price, radius=10)
        if not (option_chain.get("by_strike") or {}):
            return

        structure["option_chain"] = option_chain
        structure["unified_chain"] = PriceEngine.build_unified_chain(
            option_chain,
            structure.get("pivot_levels") or {},
            current_price,
            strike_step=self._strike_step(),
            radius=2,
        )
        chain_levels = structure["unified_chain"].get("levels") or []
        active_supports = [
            float(level["support"])
            for level in chain_levels
            if level.get("support") is not None and float(level["support"]) <= float(current_price)
        ]
        active_resistances = [
            float(level["resistance"])
            for level in chain_levels
            if level.get("resistance") is not None and float(level["resistance"]) > float(current_price)
        ]
        if active_supports and active_resistances:
            structure["support"] = max(active_supports)
            structure["resistance"] = min(active_resistances)
            structure["pivot_support"] = structure["support"]
            structure["pivot_resistance"] = structure["resistance"]
            structure["sr_source"] = "broker_oc_connected_ladder"
        chain_strikes = [
            float(strike)
            for strike in (option_chain.get("by_strike") or {})
        ]
        structure["pivot_ladder"] = build_connected_ladder(
            chain_strikes,
            structure.get("pivot_levels") or {},
        )
        self._market_structure_cache[symbol] = structure

    @staticmethod
    def _refresh_console_view() -> None:
        """Refresh in-place to reduce Windows terminal flicker."""
        if sys.stdout.isatty():
            print("\033[H\033[J", end="")

    def _india_vix(self):
        """Fetch India VIX from the broker, falling back to the last-known value.

        Never returns 0.0: a rate-limited/failed fetch reuses the cached
        reading so the dashboard metric does not disappear.
        """
        try:
            # India VIX is an NSE AMXIDX entry, not a tradable symbol in
            # SYMBOL_REGISTRY, so it must not go through _resolve_underlying.
            if not self.instrument_reader.instruments:
                self.instrument_reader.load()
            vix = next(
                (
                    item for item in self.instrument_reader.instruments
                    if str(item.get("exch_seg", "")).upper() == "NSE"
                    and str(item.get("name", "")).upper() == "INDIA VIX"
                ),
                None,
            )
            if not vix:
                raise RuntimeError("India VIX token is missing from the instrument master")
            vix_value = self.market_data.fetch_latest_price(
                vix.get("token"), exchange="NSE", tradingsymbol=vix.get("symbol")
            )
            if isinstance(vix_value, (int, float)) and vix_value > 0:
                return vix_value
            log.debug("India VIX fetch returned no value; using last-known")
        except Exception as exc:
            log.debug("India VIX unavailable: %s", exc)
        return self._last_metrics.get("india_vix")

    def _option_iv(self, symbol: str):
        """ATM IV averaged across the matching CE and PE contracts."""
        cfg = SYMBOL_REGISTRY.get(symbol, SYMBOL_REGISTRY[DEFAULT_SYMBOL])
        strike = self._last_atm_strike
        if strike is None:
            return None

        iv_values = []
        for right in ("CE", "PE"):
            info = self.instrument_reader.find_option_token(
                underlying=symbol, strike=strike, right=right,
                instrument_type=cfg["option_instrumenttype"], exchange=cfg["exchange"],
            )
            if not info:
                continue
            iv = self.market_data.fetch_option_iv(
                info.get("name") or symbol,
                info.get("expiry"),
                strike=strike,
                option_type=right,
            )
            if isinstance(iv, (int, float)) and iv > 0:
                iv_values.append(float(iv))

        return sum(iv_values) / len(iv_values) if iv_values else None

    def _nearest_strike(self, symbol: str, price):
        if price is None:
            return None
        step = SYMBOL_REGISTRY.get(symbol, SYMBOL_REGISTRY[DEFAULT_SYMBOL]).get("strike_step", 50)
        return round(float(price) / step) * step

    def _is_market_open(self) -> bool:
        now = datetime.now(IST)
        market_open, market_close = self._market_hours()
        return now.weekday() < 5 and market_open <= now.time() <= market_close

    def _nearest_level(self, levels, current_price, below: bool):
        valid = [float(level) for level in levels if level is not None]
        if current_price is None or not valid:
            return max(valid) if below and valid else min(valid) if valid else None
        eligible = [level for level in valid if level <= current_price] if below else [level for level in valid if level >= current_price]
        return max(eligible) if below and eligible else min(eligible) if eligible else (max(valid) if below else min(valid))

    def _check_all_exits(self):
        """Check every open position's SL/Target against its own symbol's
        current index price.
        """
        closed = []
        for position in list(self.tracker.open_positions()):
            cfg = SYMBOL_REGISTRY.get(position.symbol, SYMBOL_REGISTRY[DEFAULT_SYMBOL])
            try:
                position_underlying = self._resolve_underlying(position.symbol)
                position_index_price = self.market_data.fetch_latest_price(
                    position_underlying.get("token"),
                    exchange=cfg["underlying_exchange"],
                    tradingsymbol=position_underlying.get("symbol"),
                )
            except Exception as exc:
                log.warning("Could not fetch index price for open %s position: %s", position.symbol, exc)
                continue

            res = self.order_manager.check_exits(
                position.symbol, position_index_price, self._option_ltp_lookup
            )

            if res:
                if isinstance(res, list):
                    closed.extend(res)
                else:
                    closed.append(res)

        return closed

    def _close_all_positions(self, reason: str) -> list:
        """Square off every open CE/PE at its current premium."""
        closed = []
        for position in list(self.tracker.open_positions()):
            exit_price = self._option_ltp_lookup(position)
            if exit_price is None:
                log.warning(
                    "Cannot square off #%s (%s): no exit LTP available",
                    position.trade_number, reason,
                )
                continue
            self.tracker.close_position(position, exit_price, reason=reason)
            closed.append(position)
        return closed

    def _check_reversal_exits(self, trend) -> list:
        """CE is closed on a bearish flip, PE on a bullish flip."""
        trend = str(trend or "").upper()
        if trend not in ("BULLISH", "BEARISH"):
            return []

        closed = []
        for position in list(self.tracker.open_positions()):
            reversed_against = (
                (position.option_type == "CE" and trend == "BEARISH")
                or (position.option_type == "PE" and trend == "BULLISH")
            )
            if not reversed_against:
                continue
            exit_price = self._option_ltp_lookup(position)
            if exit_price is None:
                log.warning("Cannot close #%s on reversal: no exit LTP available", position.trade_number)
                continue
            self.tracker.close_position(position, exit_price, reason="REVERSAL")
            closed.append(position)
        return closed

    def _open_trade(self, decision, index_price=None, snapshot=None, market_structure=None):
        cfg = SYMBOL_REGISTRY.get(decision.underlying, SYMBOL_REGISTRY[DEFAULT_SYMBOL])
        info = self.instrument_reader.find_option_token(
            underlying=decision.underlying,
            strike=decision.strike,
            right=decision.option_type,
            instrument_type=cfg["option_instrumenttype"],
            exchange=cfg["exchange"],
        )
        if not info:
            log.warning(
                "No option token found for %s strike=%s type=%s; skipping trade",
                decision.underlying, decision.strike, decision.option_type,
            )
            return

        ltp = self.market_data.fetch_latest_price(
            info.get("token"), exchange=info.get("exch_seg"), tradingsymbol=info.get("symbol")
        )
        if ltp is None:
            log.warning("No LTP available for %s; skipping trade", info.get("symbol"))
            return

        try:
            decision.quantity = int(float(info.get("lotsize", decision.quantity)))
        except (TypeError, ValueError):
            pass

        # Fallback for index values if decision object didn't calculate them
        snapshot_spot = snapshot.get("spot") if snapshot else None
        current_index = index_price or snapshot_spot or 0.0

        if getattr(decision, "index_entry", None) is None:
            decision.index_entry = current_index
        if getattr(decision, "index_sl", None) is None and current_index:
            # Default 20 points SL if not set by decision engine
            decision.index_sl = current_index - 20.0 if decision.action == "BUY" else current_index + 20.0
        if getattr(decision, "index_target", None) is None and current_index:
            # Default 40 points Target if not set by decision engine
            decision.index_target = current_index + 40.0 if decision.action == "BUY" else current_index - 40.0

        entry_metadata = self._entry_metadata(decision, market_structure or {})
        position = self.order_manager.execute(decision, ltp, entry_metadata=entry_metadata)
        if position:
            record_id = getattr(decision, "training_record_id", None)
            if record_id:
                self.training_data.attach_trade(record_id, position.trade_number)
            # Explicitly attach index attributes to position if order_manager missed them
            if getattr(position, "index_entry", None) is None:
                position.index_entry = decision.index_entry
            if getattr(position, "index_sl", None) is None:
                position.index_sl = decision.index_sl
            if getattr(position, "index_target", None) is None:
                position.index_target = decision.index_target
            position.entry_metadata = entry_metadata
            position.trailing_stop = decision.index_sl
            self._persist_state()
            if position.symbol == "NIFTY":
                from nifty_sheet_logger import log_nifty_entry

                log_nifty_entry(self._nifty_sheet_row(position))

            log.info(
                "[PAPER TRADE OPENED] #%d %s %s %s @ %s (index entry=%s SL=%s Target=%s)",
                position.trade_number, position.side, position.symbol, position.option_type,
                position.entry_price, position.index_entry, position.index_sl, position.index_target,
            )

    @staticmethod
    def _entry_metadata(decision, market_structure: dict) -> dict:
        structures = market_structure.get("candle_structures") or {}
        candles = []
        for interval in ("ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE"):
            candles = (structures.get(interval) or {}).get("candles") or []
            if candles:
                break
        candle = dict(candles[-1]) if candles else {}
        grouped = (market_structure.get("option_chain") or {}).get("by_strike") or {}
        quotes = grouped.get(str(decision.strike)) or grouped.get(str(float(decision.strike))) or {}
        ce = dict(quotes.get("CE") or {})
        pe = dict(quotes.get("PE") or {})
        return {
            "candle": {key: candle.get(key) for key in ("open", "high", "low", "close")},
            "pivot_sr_level": decision.resistance if decision.option_type == "CE" else decision.support,
            "strike": decision.strike,
            "option_type": decision.option_type,
            "ce": {"strike": decision.strike, "oi": ce.get("open_interest"), "oi_change": ce.get("oi_change")},
            "pe": {"strike": decision.strike, "oi": pe.get("open_interest"), "oi_change": pe.get("oi_change")},
        }

    @staticmethod
    def _nifty_sheet_row(position: Position) -> dict:
        metadata = position.entry_metadata or {}
        candle = metadata.get("candle") or {}
        ce = metadata.get("ce") or {}
        pe = metadata.get("pe") or {}
        return {
            "Timestamp": position.opened_at.isoformat(), "Symbol": "NIFTY", "Action": position.side,
            "Execution Price (LTP)": position.entry_price,
            "Candle Open": candle.get("open"), "Candle High": candle.get("high"),
            "Candle Low": candle.get("low"), "Candle Close": candle.get("close"),
            "Pivot / SR Level": metadata.get("pivot_sr_level"), "Strike": position.strike,
            "Option Side": position.option_type, "CE OI": ce.get("oi"), "CE OI Change": ce.get("oi_change"),
            "PE OI": pe.get("oi"), "PE OI Change": pe.get("oi_change"),
            "Strike Context JSON": json.dumps(metadata, separators=(",", ":"), default=str),
            "Target": position.index_target, "Stop Loss": position.index_sl,
        }

    def _record_training_outcome(self, position: Position) -> None:
        if position.close_reason == "TARGET_HIT":
            outcome = "WIN"
        elif position.close_reason == "SL_HIT":
            outcome = "LOSS"
        else:
            outcome = "OTHER"
        self.training_data.attach_outcome(position.trade_number, outcome, position.pnl)

    # ------------------------------------------------------------
    # DASHBOARD / MESSAGES
    # ------------------------------------------------------------

    def _print_dashboard(self, snapshot, symbol, index_price, decision, market_structure=None):
        self._refresh_console_view()
        print("=" * 78)
        now = datetime.now()
        market_status = decision.market_status if decision else snapshot.get("market_status", "CLOSED")
        status = "MARKET CLOSED" if market_status == "CLOSED" else "ACTIVE"
        print(f"TRADING DASHBOARD | {now.strftime('%Y-%m-%d %H:%M:%S')} | MODE: {self.order_manager.mode.value} | {status}")
        print("=" * 78)
        
        # Get market data from snapshot with fallbacks
        trend = snapshot.get("trend")
        vix = snapshot.get("vix")
        iv = snapshot.get("iv")
        trend_1m = snapshot.get("trend_1m")
        
        # Format display values
        trend_str = f"{trend}" if trend else "NEUTRAL"
        vix = vix if isinstance(vix, (int, float)) and vix > 0 else self._last_metrics.get("india_vix")
        iv = iv if isinstance(iv, (int, float)) and iv > 0 else self._last_metrics.get("option_iv")
        vix_str = f"{vix:.2f}" if isinstance(vix, (int, float)) else "--"
        iv_str = f"{iv:.2f}" if isinstance(iv, (int, float)) else "--"
        trend_1m_str = f"{trend_1m}" if trend_1m else "NEUTRAL"
        
        print("MARKET INFO")
        volatility_label, iv_label = self._dashboard_metric_labels()
        volatility_value = self._last_metrics.get("crude_volatility") if symbol == "CRUDEOIL" else vix
        volatility_display = volatility_value if isinstance(volatility_value, (int, float)) and volatility_value > 0 else None
        volatility_str = f"{volatility_display:.2f}" if volatility_display is not None else "--"
        print(f"Symbol: {symbol} | Spot LTP: {index_price} | Trend: {trend_str} | {volatility_label}: {volatility_str} | {iv_label}: {iv_str}")
        updated_str = self._metrics_updated_at.strftime("%H:%M:%S") if self._metrics_updated_at else "--:--:--"
        print(f"IV           : {iv_str}")
        if symbol == "CRUDEOIL":
            print(f"CRUDE VOL    : {volatility_str}")
        else:
            print(f"INDIA VIX    : {vix_str}")
        print(f"LAST UPDATED : {updated_str}")
        print("STRUCTURAL LEVELS")
        display_structure = market_structure or {}
        display_pivots = dict(display_structure.get("pivot_levels") or {})
        if not display_pivots:
            display_pivots = dict((self._pivot_cache.get(symbol) or {}).get("pivots") or {})
        display_support = decision.support if decision and decision.support is not None else display_structure.get("support")
        display_resistance = decision.resistance if decision and decision.resistance is not None else display_structure.get("resistance")
        if display_support is None:
            display_support = display_pivots.get("s1")
        if display_resistance is None:
            display_resistance = display_pivots.get("r1")
        support_display = self._format_exact_sr(display_support) if display_support is not None else "N/A"
        resistance_display = self._format_exact_sr(display_resistance) if display_resistance is not None else "N/A"
        print(
            f"PIVOT SR SUPPORT: {support_display} | "
            f"PIVOT SR RESISTANCE: {resistance_display} | "
            f"AOC: {snapshot.get('aoc_scenario') or 'NEUTRAL'} | 1m: {trend_1m_str}"
        )
        print(
            f"SR SOURCE        : {display_structure.get('sr_source') or 'broker_classic_pivot'}"
        )
        pivot_levels = display_pivots
        if pivot_levels:
            def pivot_text(key):
                value = pivot_levels.get(key)
                numeric = self._to_float(value)
                return f"{numeric:.4f}" if numeric is not None else "--"

            print(
                "DAILY OHLC PIVOTS: "
                f"PP={pivot_text('pp')} | S1={pivot_text('s1')} | R1={pivot_text('r1')} | "
                f"S2={pivot_text('s2')} | R2={pivot_text('r2')}"
            )
        unified_chain = display_structure.get("unified_chain") or {}
        chain_levels = unified_chain.get("levels") or []
        if chain_levels:
            print("LIVE BROKER OC S/R LADDER")
            for level in chain_levels:
                labels = "/".join(level.get("pivot_labels") or []) or "-"
                support = level.get("support") if level.get("support") is not None else "-"
                resistance = level.get("resistance") if level.get("resistance") is not None else "-"
                print(
                    f"  {level['strike']:.0f}: S={support} R={resistance} "
                    f"PIVOT={labels} | PEscore={level.get('support_score', 0):.0f} "
                    f"CEscore={level.get('resistance_score', 0):.0f}"
                )
            print(
                f"CHAIN TARGETS    : UP={unified_chain.get('target_up') or '-'} | "
                f"DOWN={unified_chain.get('target_down') or '-'}"
            )
        if decision:
            support_range_str = f" (range: {decision.support_range[0]:.2f}-{decision.support_range[1]:.2f})" if decision.support_range else ""
            resistance_range_str = f" (range: {decision.resistance_range[0]:.2f}-{decision.resistance_range[1]:.2f})" if decision.resistance_range else ""
            print(f"SUPPORT/RESIST   : {decision.support}{support_range_str} / {decision.resistance}{resistance_range_str}")
        watchlist = snapshot.get("aoc_watchlist") or {"top": [], "bottom": []}
        print(f"AOC 75%+ WATCH   : TOP={watchlist.get('top', [])} | BOTTOM={watchlist.get('bottom', [])}")

        trades_today = self.tracker.trades_today_count()
        limit = self.risk_manager.limits.max_trades_per_day
        print("POSITION & RISK")
        max_stop_losses = self.risk_manager.limits.max_consecutive_stop_losses
        trade_limit_label = limit if limit is not None else "Unlimited"
        stop_limit_label = max_stop_losses if max_stop_losses is not None else "Unlimited"
        print(f"Trades Today: {trades_today}/{trade_limit_label} | Consecutive SL: {self.tracker.consecutive_stop_losses()}/{stop_limit_label} | Realized P&L: {self.tracker.daily_realized_pnl():.2f}")

        open_positions = self.tracker.open_positions()
        if not open_positions:
            print("Running Trade: None")
        else:
            for position in open_positions:
                current_ltp = self._option_ltp_lookup(position)
                unrealized = None
                if current_ltp is not None:
                    direction = 1 if position.side == "BUY" else -1
                    unrealized = direction * (current_ltp - position.entry_price) * position.quantity
                print(
                    f"OPEN #{position.trade_number}: {position.side} {position.symbol} "
                    f"{position.strike}{position.option_type} qty={position.quantity} "
                    f"entry={position.entry_price} ltp={current_ltp} "
                    f"index_entry={position.index_entry} sl={position.index_sl} "
                    f"target={position.index_target} unrealized_pnl={unrealized}"
                )
        print("=" * 78)

    def _print_trade_closed(self, position: Position):
        reasons = {
            "SL_HIT": "SL Hit",
            "TIME_EXIT": "Time Square-off (15:15 IST)",
            "REVERSAL": "Trend Reversal Exit",
        }
        reason = reasons.get(position.close_reason, "TP Hit")
        print()
        print("*" * 78)
        print(f"TRADE CLOSED - #{position.trade_number}")
        print("*" * 78)
        print(f"Symbol      : {position.symbol} {position.strike}{position.option_type}")
        print(f"Entry Time  : {position.opened_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Exit Time   : {position.closed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Buy Price   : {position.entry_price if position.side == 'BUY' else position.exit_price}")
        print(f"Sell Price  : {position.exit_price if position.side == 'BUY' else position.entry_price}")
        print(f"P&L         : {position.pnl:.2f}")
        print(f"Reason      : {reason}")
        print("*" * 78)

    def print_daily_summary(self):
        print()
        print("#" * 78)
        print(f"DAILY SUMMARY - {datetime.now().strftime('%Y-%m-%d')}")
        print("#" * 78)

        closed_today = sorted(self.tracker.closed_today(), key=lambda p: p.opened_at)
        if not closed_today:
            print("No trades closed yet today.")
        else:
            header = (
                f"{'#':<4}{'Entry Time':<12}{'Exit Time':<12}{'Symbol':<24}"
                f"{'Buy':<10}{'Sell':<10}{'P&L':<10}{'Reason'}"
            )
            print(header)
            print("-" * len(header))
            for position in closed_today:
                buy_price = position.entry_price if position.side == "BUY" else position.exit_price
                sell_price = position.exit_price if position.side == "BUY" else position.entry_price
                reason = (
                    "SL Hit" if position.close_reason == "SL_HIT"
                    else "TP Hit" if position.close_reason == "TARGET_HIT"
                    else position.close_reason
                )
                symbol_label = f"{position.symbol} {position.strike}{position.option_type}"
                print(
                    f"{position.trade_number:<4}"
                    f"{position.opened_at.strftime('%H:%M:%S'):<12}"
                    f"{position.closed_at.strftime('%H:%M:%S'):<12}"
                    f"{symbol_label:<24}"
                    f"{buy_price:<10.2f}"
                    f"{sell_price:<10.2f}"
                    f"{position.pnl:<10.2f}"
                    f"{reason}"
                )

        print("-" * 78)
        limit = self.risk_manager.limits.max_trades_per_day
        print(f"TOTAL TRADES : {len(closed_today)} / {limit if limit is not None else 'Unlimited'}")
        print(f"TOTAL P&L    : {self.tracker.daily_realized_pnl():.2f}")
        print("#" * 78)

    # ------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------

    def run_forever(self, interval_seconds: float | None = None):
        interval = interval_seconds if interval_seconds is not None else settings.live_poll_interval_seconds
        interval = max(2.0, min(10.0, interval))

        log.info("Starting live paper trading session (interval=%ss)", interval)

        try:
            while True:
                try:
                    self.run_once()
                except Exception:
                    log.error("Error during trading session tick", exc_info=True)

                max_trades = self.risk_manager.limits.max_trades_per_day
                limit_reached = max_trades is not None and self.tracker.trades_today_count() >= max_trades
                if limit_reached and not self.tracker.open_positions():
                    print("\nDaily trade limit reached and no open positions - stopping session.")
                    break

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nSession interrupted by user.")
        finally:
            self.print_daily_summary()