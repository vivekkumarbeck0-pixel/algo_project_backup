# ============================================================
# ANGEL ONE INSTRUMENT READER
# ============================================================

import json
import requests
from datetime import datetime
from pathlib import Path


class InstrumentReader:

    # ========================================================
    # SETTINGS
    # ========================================================

    MASTER_URL = (
        "https://margincalculator.angelbroking.com/"
        "OpenAPI_File/files/OpenAPIScripMaster.json"
    )

    MASTER_FILE = Path(
        "data/angel_one_instruments.json"
    )

    # ========================================================
    # INIT
    # ========================================================

    def __init__(self):

        self.instruments = []

    # ========================================================
    # DOWNLOAD INSTRUMENT MASTER
    # ========================================================

    def download(self):

        print()
        print("=" * 70)
        print("DOWNLOADING ANGEL ONE INSTRUMENT MASTER")
        print("=" * 70)

        response = requests.get(
            self.MASTER_URL,
            timeout=30
        )

        response.raise_for_status()

        self.instruments = response.json()

        self.MASTER_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Written via a temp file so a concurrent reader never sees a half-written master.
        temporary_file = self.MASTER_FILE.with_suffix(".tmp")

        with open(
            temporary_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.instruments,
                file,
                ensure_ascii=False
            )

        temporary_file.replace(self.MASTER_FILE)

        print()
        print("INSTRUMENTS :", len(self.instruments))
        print("SAVED :", self.MASTER_FILE)

        return self.instruments

    # ========================================================
    # LOAD LOCAL MASTER
    # ========================================================

    def load(self):

        if not self.MASTER_FILE.exists():

            print()
            print("LOCAL MASTER NOT FOUND")
            print("DOWNLOADING NEW MASTER...")

            return self.download()

        with open(
            self.MASTER_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.instruments = json.load(file)

        print()
        print("LOCAL INSTRUMENT MASTER LOADED")
        print("INSTRUMENTS :", len(self.instruments))

        return self.instruments

    # ========================================================
    # FIND SYMBOL
    # ========================================================

    def find_symbol(self, symbol):

        if not self.instruments:
            self.load()

        symbol = symbol.upper()

        result = []

        for item in self.instruments:

            trading_symbol = str(
                item.get("symbol", "")
            ).upper()

            if trading_symbol == symbol:

                result.append(item)

        return result

    # ========================================================
    # FIND BY PARTIAL SYMBOL
    # ========================================================

    def search(self, text):

        if not self.instruments:
            self.load()

        text = text.upper()

        result = []

        for item in self.instruments:

            trading_symbol = str(
                item.get("symbol", "")
            ).upper()

            if text in trading_symbol:

                result.append(item)

        return result

    # ========================================================
    # PRINT RESULT
    # ========================================================

    def print_results(self, results):

        print()
        print("=" * 70)
        print("INSTRUMENT SEARCH RESULT")
        print("=" * 70)

        if not results:

            print()
            print("NO INSTRUMENT FOUND")

            return

        for item in results:

            print()
            print("SYMBOL :", item.get("symbol"))
            print("TOKEN  :", item.get("token"))
            print("EXCH   :", item.get("exch_seg"))
            print("NAME   :", item.get("name"))
            print("TYPE   :", item.get("instrumenttype"))

    # ========================================================
    # OPTION TOKEN LOOKUP
    # ========================================================

    def find_option_tokens(
        self,
        underlying,
        strike,
        right,
        expiry=None,
        limit=10,
        instrument_type="OPTIDX",
        exchange=None,
    ):
        """Find CE/PE option contracts for `underlying` at `strike`.

        `instrument_type`/`exchange` default to NSE index options
        (OPTIDX/NFO, e.g. NIFTY). Pass instrument_type="OPTFUT",
        exchange="MCX" for commodity options on futures (e.g. CRUDEOIL).
        """

        if not self.instruments:
            self.load()

        underlying = str(underlying or "").upper()
        right = str(right or "").upper()
        instrument_type = str(instrument_type or "").upper()
        exchange = str(exchange or "").upper() if exchange else None
        strike_value = float(strike) if strike not in (None, "") else None

        results = []

        for item in self.instruments:

            symbol = str(item.get("symbol", "")).upper()
            name = str(item.get("name", "")).upper()
            item_instrument_type = str(item.get("instrumenttype", "")).upper()
            item_exchange = str(item.get("exch_seg", "")).upper()
            expiry_value = str(item.get("expiry", "")).upper()

            if item_instrument_type != instrument_type:
                continue

            if exchange and item_exchange != exchange:
                continue

            # Exact name match (not a symbol substring check) so e.g.
            # underlying="NIFTY" never matches "NIFTYNXT50" options, and
            # underlying="CRUDEOIL" never matches the "CRUDEOILM" mini
            # contract - both would otherwise contain the substring.
            if underlying and name != underlying:
                continue

            if right and not symbol.endswith(right):
                continue

            # Angel One stores `strike` multiplied by 100 for every segment
            # (e.g. 24400 -> "2440000.000000") - compare numerically instead
            # of via a fragile substring check, which produced false
            # positives on padded/zero-heavy MCX strike values.
            if strike_value is not None:
                try:
                    item_strike = float(item.get("strike", 0)) / 100
                except (TypeError, ValueError):
                    continue
                if abs(item_strike - strike_value) > 0.01:
                    continue

            if expiry and expiry.upper() not in expiry_value:
                continue

            results.append(item)

        def expiry_key(item):
            expiry_date = item.get("expiry")
            try:
                return datetime.strptime(expiry_date, "%d%b%Y")
            except Exception:
                return datetime.max

        # Drop already-expired contracts (Angel One's live scrip cache purges
        # them even though the locally-cached instrument master JSON may
        # still list them, which caused "Symbol token not found in scrip
        # master cache" errors when the nearest-by-date pick was expired).
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        results = [item for item in results if expiry_key(item) >= today_start]

        results.sort(key=expiry_key)

        return results[:limit]

    def find_option_token(
        self, underlying, strike, right, expiry=None, instrument_type="OPTIDX", exchange=None
    ):
        tokens = self.find_option_tokens(
            underlying,
            strike,
            right,
            expiry=expiry,
            limit=1,
            instrument_type=instrument_type,
            exchange=exchange,
        )

        return tokens[0] if tokens else None

    # ========================================================
    # NEAREST FUTURES CONTRACT (index candle-data proxy, or the
    # tradable underlying itself for commodities like CRUDEOIL)
    # ========================================================

    def find_nearest_future(self, underlying, instrument_type="FUTIDX", exchange=None):
        """Find the nearest-expiry futures contract for an underlying.

        Angel One's historical getCandleData endpoint does not return
        intraday candles for plain NSE index tokens (e.g. NSE:NIFTY). The
        near-month futures contract tracks the index closely and does
        have historical candle data, so it is used as a fallback proxy
        for index candles/OHLC. See angel_one/market_data.py.

        For commodities (e.g. CRUDEOIL/MCX) there is no separate index -
        this same lookup resolves the tradable underlying itself.
        """
        if not self.instruments:
            self.load()

        underlying = str(underlying or "").upper()
        instrument_type = str(instrument_type or "").upper()
        exchange = str(exchange or "").upper() if exchange else None

        results = [
            item for item in self.instruments
            if str(item.get("name", "")).upper() == underlying
            and str(item.get("instrumenttype", "")).upper() == instrument_type
            and (not exchange or str(item.get("exch_seg", "")).upper() == exchange)
        ]

        def expiry_key(item):
            try:
                return datetime.strptime(item.get("expiry"), "%d%b%Y")
            except Exception:
                return datetime.max

        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        results = [item for item in results if expiry_key(item) >= today_start]

        results.sort(key=expiry_key)

        return results[0] if results else None

    # ========================================================
    # RESOLVE UNDERLYING (dynamic symbol switching)
    # ========================================================

    def resolve_underlying(self, symbol):
        """Resolve the tradable "underlying" instrument for a symbol
        detected by aoc_parser (e.g. "NIFTY" or "CRUDEOIL"), using
        config.SYMBOL_REGISTRY to know which exchange/instrument-type
        combination applies.

        NSE indices (NIFTY) have a dedicated spot entry (blank
        instrumenttype). Commodities (CRUDEOIL/MCX) have no separate
        index - the near-month futures contract itself is returned.
        """
        from config import SYMBOL_REGISTRY  # local import avoids a config<->angel_one cycle at module load

        symbol = str(symbol or "").upper()
        cfg = SYMBOL_REGISTRY.get(symbol)
        if not cfg:
            raise ValueError(f"Unknown symbol '{symbol}'; add it to config.SYMBOL_REGISTRY")

        underlying_exchange = cfg["underlying_exchange"]
        underlying_instrumenttype = cfg["underlying_instrumenttype"]

        if underlying_instrumenttype == "":
            if not self.instruments:
                self.load()
            for item in self.instruments:
                if (
                    str(item.get("exch_seg", "")).upper() == underlying_exchange
                    and str(item.get("name", "")).upper() == symbol
                    and str(item.get("instrumenttype", "")).upper() == ""
                ):
                    return item
            return None

        return self.find_nearest_future(
            symbol, instrument_type=underlying_instrumenttype, exchange=underlying_exchange
        )
