"""Standalone MCX Crude Oil option-buying configuration.

This file intentionally contains no Nifty-specific symbols, imports, or
shared project modules. It is designed to run as a fully isolated crude-oil
trading engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _getenv(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def _getenv_any(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _load_dotenv() -> None:
    dotenv_path = Path(__file__).resolve().with_name(".env")
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


_load_dotenv()

# MCX Crude-only strategy defaults. Nifty does not consume this constant.
ATR_MULTIPLIER = 2.0


@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str
    exchange: str
    future_instrument_type: str
    option_instrument_type: str
    strike_step: int
    lot_size: int


@dataclass
class Settings:
    """Configuration values for the MCX crude-only futures + option strategy."""

    # SAFE DEFAULT: keep the engine in paper mode until the user explicitly enables real trading.
    execution_mode: Literal["PAPER", "REAL"] = "PAPER"
    allow_real_trading: bool = False
    market_name: str = "MCX_CRUDE_OIL"

    # Broker authentication
    angel_api_key: str = _getenv_any("ANGEL_API_KEY", "ANGEL_ONE_API_KEY")
    angel_client_code: str = _getenv_any("ANGEL_CLIENT_CODE", "ANGEL_ONE_CLIENT_ID")
    angel_pin: str = _getenv_any("ANGEL_PIN", "ANGEL_ONE_PASSWORD")
    angel_feed_token: str = _getenv_any("ANGEL_FEED_TOKEN", "ANGEL_ONE_FEED_TOKEN")
    angel_jwt_token: str = _getenv_any("ANGEL_JWT_TOKEN", "ANGEL_ONE_JWT_TOKEN")
    angel_totp_secret: str = _getenv_any("ANGEL_TOTP_SECRET", "ANGEL_ONE_TOTP")

    # Market identity
    futures_symbol: str = "CRUDEOIL"
    futures_exchange: str = "MCX"
    futures_instrument_type: str = "FUTCOM"
    option_exchange: str = "MCX"
    option_instrument_type: str = "OPTFUT"
    strike_step: int = 50
    future_lot_size: int = 100
    option_lot_size: int = 100

    # Session timing
    market_start_time_ist: str = "09:00"
    market_close_time_ist: str = "23:30"
    trade_start_time_ist: str = "09:15"
    trade_end_time_ist: str = "23:15"
    # Every open position is squared off this many minutes before market close.
    square_off_buffer_minutes: int = 5

    # Strategy constants
    pivot_lookback_bars: int = 200
    # MCX Crude-only pivot/entry controls. These are not read by Nifty.
    crude_atr_multiplier: float = ATR_MULTIPLIER
    crude_entry_candle_zone: float = 0.35
    crude_ai_proximity_tolerance: float = 0.002
    crude_ai_min_momentum: float = 0.05
    crude_ai_max_momentum: float = 2.0
    atr_period: int = 14
    ma_volume_period: int = 20
    volume_spike_factor: float = 1.2
    # Confirmation cushion so wicks that only graze a pivot cannot trigger an entry.
    buffer_points: float = 4.0
    # Skip entries whose stop would eat more than this share of the premium.
    max_stop_premium_fraction: float = 0.5
    atr_stop_mult: float = 1.5
    atr_target_mult: float = 3.0
    trailing_stop_points: float = 100.0
    max_trailing_stop_points: float = 150.0
    default_option_margin: float = 1.0

    # Real-time data timing
    futures_tick_timeout_seconds: float = 15.0
    websocket_reconnect_seconds: float = 3.0
    yfinance_refresh_seconds: float = 60.0
    max_futures_history_bars: int = 2000
    max_nymex_history_bars: int = 2000

    # ----------------------------------------------------------
    # NSE / Nifty session settings (engine/ + trading_nifty.py).
    # The crude engine does not read anything below this line.
    # ----------------------------------------------------------
    trade_start_time: str = "09:20"
    square_off_time: str = "15:28"
    daily_state_file: str = "data/daily_state.json"
    live_poll_interval_seconds: float = 5.0
    metrics_refresh_interval_seconds: float = 15.0
    metrics_failure_retry_seconds: float = 5.0
    structural_refresh_interval_seconds: float = 300.0

    # Angel One REST throttling shared by engine/ and angel_one/market_data.py
    api_min_request_interval_seconds: float = 0.35
    # The historical-candle endpoint throttles far harder than quotes.
    candle_min_request_interval_seconds: float = 1.5
    candle_disk_cache_file: str = "data/candle_cache.json"
    candle_disk_cache_max_age_seconds: float = 86400.0
    # Only used when nothing is cached yet; a blind start would otherwise kill trading.
    candle_fetch_retries: int = 3
    candle_retry_backoff_seconds: float = 5.0
    candle_cooldown_max_wait_seconds: float = 35.0
    # An empty option chain silently disables OI-based targets, so retry then reuse.
    option_chain_retries: int = 3
    option_chain_retry_backoff_seconds: float = 2.0
    option_chain_max_stale_seconds: float = 120.0
    api_rate_limit_cooldown_seconds: float = 30.0
    api_cache_ttl_seconds: float = 5.0
    ltp_cache_ttl_seconds: float = 1.0

    # SmartStream websocket feed; quotes newer than this age bypass the REST API.
    use_websocket_feed: bool = True
    ws_tick_max_age_seconds: float = 6.0

    # Nifty default risk sizing; SYMBOL_REGISTRY entries override per symbol.
    sl_points: float = 25.0
    target_points: float = 30.0
    max_open_positions: int = 1
    # Daily trade count is intentionally uncapped; other gates still apply.
    max_trades_per_day: int = 1_000_000
    max_daily_loss: float = 5000.0
    # Consecutive stop-loss lock removed: never halt the session on SL streaks.
    max_consecutive_stop_losses: int | None = None

    # Logging
    log_level: str = _getenv("LOG_LEVEL", "INFO")
    log_file: str = "logs/crude_trade.log"
    paper_trade_log: str = "logs/crude_paper_trades.jsonl"
    paper_trade_summary: str = "logs/crude_paper_summary.json"
    crude_state_file: str = "data/crude_daily_state.json"


settings = Settings()

# Per-symbol broker wiring used by engine/ and angel_one/.
# Every consumer reads these as plain dict keys, so keep the key names stable.
SYMBOL_REGISTRY: dict[str, dict] = {
    "NIFTY": {
        "exchange": "NFO",
        "underlying_exchange": "NSE",
        "underlying_instrumenttype": "",
        "option_instrumenttype": "OPTIDX",
        "strike_step": 50,
        "lot_size": 65,
        "sl_points": 25.0,
        "target_points": 30.0,
    },
    "CRUDEOIL": {
        "exchange": "MCX",
        "underlying_exchange": "MCX",
        "underlying_instrumenttype": "FUTCOM",
        "option_instrumenttype": "OPTFUT",
        "strike_step": 50,
        "lot_size": 100,
        "sl_points": 20.0,
        "target_points": 40.0,
    },
}

DEFAULT_SYMBOL: str = "NIFTY"

MCX_CRUDE_FUTURE: InstrumentConfig = InstrumentConfig(
    symbol=settings.futures_symbol,
    exchange=settings.futures_exchange,
    future_instrument_type=settings.futures_instrument_type,
    option_instrument_type=settings.option_instrument_type,
    strike_step=settings.strike_step,
    lot_size=settings.future_lot_size,
)

MCX_CRUDE_OPTION = InstrumentConfig(
    symbol=settings.futures_symbol,
    exchange=settings.option_exchange,
    future_instrument_type=settings.futures_instrument_type,
    option_instrument_type=settings.option_instrument_type,
    strike_step=settings.strike_step,
    lot_size=settings.option_lot_size,
)

# IMPORTANT: This engine intentionally does not import or use any Nifty-specific
# trading modules. All live decisions are based on MCX Crude FUTCOM price, OI,
# fresh NYMEX CL=F trend, and ATM option execution only.
