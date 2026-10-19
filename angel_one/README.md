FOLDER : angel_one

Purpose :

Angel One (Angel Broking) SmartAPI broker integration - login,
instrument master, and market data fetching.

Files :

login.py

Handles authentication (API key, client id, password, TOTP) loaded
from `.env` / environment variables, and wraps the SmartAPI client
(`smart_api`) for session creation, candle data, and market data
lookups.

Requires the `smartapi-python` package (imports `SmartApi` /
`SmartApi.smartConnect`). Install it in the active virtual
environment if the import fails.

instrument_reader.py

Downloads/searches the Angel One instrument master (contract list)
to resolve trading symbols/tokens (e.g. NIFTY option strikes) needed
before placing or fetching data for an order.

market_data.py

Fetches market data / OHLC candles and derives market structure
levels (support, resistance, order blocks) for a given token.

Environment variables (see root README / `.env.example`) :

ANGEL_ONE_API_KEY
ANGEL_ONE_CLIENT_ID
ANGEL_ONE_PASSWORD
ANGEL_ONE_TOTP

Status :

Login : 🔄 Works once smartapi-python is installed

Instrument Reader : ✅ Working

Market Data : ✅ Working (used for support/resistance levels)

Order Placement : ❌ Not implemented (see engine/order_manager.py)
