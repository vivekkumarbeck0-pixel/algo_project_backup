import sys
from pathlib import Path
import logging

# Enable verbose HTTP and library debugging to capture request/response details
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("SmartApi").setLevel(logging.DEBUG)
logging.getLogger("smartConnect").setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)
logging.getLogger("requests").setLevel(logging.DEBUG)
try:
    import http.client as http_client
    http_client.HTTPConnection.debuglevel = 1
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from angel_one.market_data import MarketDataFetcher
from angel_one.instrument_reader import InstrumentReader


def probe():
    fetcher = MarketDataFetcher()
    if not fetcher.client:
        print("No client available; cannot probe exchanges")
        return

    ir = InstrumentReader()
    tokens = ir.search("NIFTY")
    if not tokens:
        print("No tokens found for NIFTY")
        return

    token = tokens[0].get("token")
    print("Probing token", token)

    for exch in (None, "NFO", "NSE", "NSEFO", "NSECM"):
        try:
            print("Trying exchange", exch)
            price = fetcher.fetch_latest_price(token, exchange=exch, verbose=True)
            print(" ->", price)
        except Exception as exc:
            print(" -> error:", exc)


if __name__ == "__main__":
    probe()
