import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from angel_one.login import AngelOneLogin


def inspect():
    try:
        client = AngelOneLogin.connect_from_env()
    except Exception as exc:
        print("Connect failed:", exc)
        try:
            client = AngelOneLogin.from_env()
        except Exception as exc2:
            print("from_env also failed:", exc2)
            return

    smart = getattr(client, "smart_api", None)
    print("smart_api type:", type(smart))
    if smart is None:
        return

    methods = [m for m in dir(smart) if not m.startswith("_")]
    print("Available smart_api attributes/methods (filtered):")
    for m in methods:
        if any(k in m.lower() for k in ("candle", "candledata", "market", "getmarket", "getcandle", "getcandledata", "getmarketdata", "getmarket")):
            print(" -", m)

    # print docstrings for key methods if available
    for name in ("getCandleData", "getMarketData"):
        if hasattr(smart, name):
            print(f"\nDoc for {name}:\n", getattr(smart, name).__doc__)

    # print a few representative attributes
    for attr in ("access_token", "refresh_token", "feed_token", "userId", "userId"):
        if hasattr(client, attr):
            print(attr, "->", getattr(client, attr))


if __name__ == "__main__":
    inspect()
