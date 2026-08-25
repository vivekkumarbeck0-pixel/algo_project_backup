# ============================================================
# ANGEL ONE MARKET DATA TEST (candles / OHLC / LTP)
#
# Verifies NIFTY 1-minute candles, then NIFTY/CE/PE LTP + candles,
# reusing the single authenticated AngelOneLogin session (no second
# login is created). Never prints credentials/tokens/secrets.
# ============================================================

from angel_one.login import AngelOneLogin
from angel_one.instrument_reader import InstrumentReader
from angel_one.market_data import MarketDataFetcher


def find_nifty_index(instruments):
    for item in instruments:
        if (
            str(item.get("exch_seg", "")).upper() == "NSE"
            and str(item.get("name", "")).upper() == "NIFTY"
            and str(item.get("instrumenttype", "")).upper() == ""
        ):
            return item
    return None


def run_candle_test(label, client, symbol, token, exchange):
    print()
def run_candle_test(label, fetcher, symbol, token, exchange, underlying_name=None):
    print()
    print("=" * 70)
    print(f"{label} - 1-MINUTE CANDLES")
    print("=" * 70)

    from_date, to_date = AngelOneLogin.market_session_range(days=1)

    print()
    print("SYMBOL          :", symbol)
    print("TOKEN           :", token)
    print("EXCHANGE        :", exchange)
    print("REQUEST FROM DATE:", from_date)
    print("REQUEST TO DATE  :", to_date)
    print("REQUEST INTERVAL :", "ONE_MINUTE")

    try:
        response = fetcher.fetch_candles(
            token,
            interval="ONE_MINUTE",
            days=1,
            exchange=exchange,
            underlying_name=underlying_name,
        )

        status = response.get("status") if isinstance(response, dict) else None
        print()
        print("API RESPONSE STATUS :", status)

        data = response.get("data") if isinstance(response, dict) else response

        if not data:
            print("RAW RESPONSE :", response)
            print("NUMBER OF CANDLES RECEIVED : 0")
            return None

        print("NUMBER OF CANDLES RECEIVED :", len(data))
        print("FIRST CANDLE :", data[0])
        print("LAST CANDLE  :", data[-1])
        return data

    except Exception as error:
        print()
        print("HTTP/API ERROR :", type(error).__name__)
        print("RAW RESPONSE   :", error)
        return None


def run_ltp_test(label, fetcher, exchange, tradingsymbol, token):
    print()
    print("=" * 70)
    print(f"{label} - LTP / OHLC")
    print("=" * 70)

    print()
    print("SYMBOL   :", tradingsymbol)
    print("TOKEN    :", token)
    print("EXCHANGE :", exchange)

    try:
        response = fetcher.client.get_ltp(exchange, tradingsymbol, token)
        status = response.get("status") if isinstance(response, dict) else None
        print("API RESPONSE STATUS :", status)

        data = response.get("data") if isinstance(response, dict) else None
        if not data:
            print("RAW RESPONSE :", response)
            return None

        print("LTP   :", data.get("ltp"))
        print("OPEN  :", data.get("open"))
        print("HIGH  :", data.get("high"))
        print("LOW   :", data.get("low"))
        print("CLOSE :", data.get("close"))
        return data

    except Exception as error:
        print()
        print("HTTP/API ERROR :", type(error).__name__)
        print("RAW RESPONSE   :", error)
        return None


print()
print("=" * 70)
print("ANGEL ONE MARKET DATA TEST")
print("=" * 70)


# ============================================================
# 1. LOGIN (single session, reused for every call below)
# ============================================================

print()
print("Connecting to Angel One...")

client = AngelOneLogin.connect_from_env()

print()
print("LOGIN : OK")


# ============================================================
# 2. LOAD INSTRUMENT MASTER
# ============================================================

reader = InstrumentReader()
instruments = reader.load()

print()
print("INSTRUMENT MASTER :", len(instruments))


# ============================================================
# 3. FIND NIFTY INDEX
# ============================================================

print()
print("=" * 70)
print("SEARCHING NIFTY INDEX")
print("=" * 70)

nifty = find_nifty_index(instruments)

if nifty is None:
    print()
    print("ERROR: NIFTY INDEX NOT FOUND")
    raise SystemExit(1)

nifty_token = str(nifty.get("token"))
nifty_symbol = str(nifty.get("symbol"))
nifty_exchange = str(nifty.get("exch_seg"))

print()
print("NIFTY SYMBOL :", nifty_symbol)
print("NIFTY TOKEN  :", nifty_token)
print("EXCHANGE     :", nifty_exchange)


# ============================================================
# 4. NIFTY 1-MINUTE CANDLES (fix validated here first)
# ============================================================

fetcher = MarketDataFetcher(client=client)

nifty_candles = run_candle_test(
    "NIFTY", fetcher, nifty_symbol, nifty_token, nifty_exchange, underlying_name="NIFTY"
)

if not nifty_candles:
    print()
    print("Stopping: NIFTY candle fetch must succeed before testing CE/PE.")
    raise SystemExit(1)


# ============================================================
# 5. NIFTY LTP
# ============================================================

nifty_ltp_data = run_ltp_test("NIFTY", fetcher, nifty_exchange, nifty_symbol, nifty_token)

last_close = float(nifty_candles[-1][4])
spot_price = (nifty_ltp_data or {}).get("ltp") or last_close


# ============================================================
# 6. CE / PE OPTION TOKEN LOOKUP (nearest strike, nearest expiry)
# ============================================================

atm_strike = round(spot_price / 50) * 50

print()
print("=" * 70)
print("CE / PE OPTION LOOKUP")
print("=" * 70)
print()
print("SPOT / LAST CLOSE :", spot_price)
print("ATM STRIKE         :", atm_strike)

ce_info = reader.find_option_token("NIFTY", atm_strike, "CE")
pe_info = reader.find_option_token("NIFTY", atm_strike, "PE")

for label, info in (("CE", ce_info), ("PE", pe_info)):
    if not info:
        print(f"{label}: NO MATCHING OPTION TOKEN FOUND FOR STRIKE {atm_strike}")
        continue

    symbol = info.get("symbol")
    token = str(info.get("token"))
    exchange = info.get("exch_seg")

    run_ltp_test(label, fetcher, exchange, symbol, token)
    run_candle_test(label, fetcher, symbol, token, exchange)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
