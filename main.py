"""Single Render entrypoint for the existing Crude and NIFTY sessions."""

import threading

from trading_crude import CrudeOptionBuyer
from trading_nifty import NiftyTradingSession


def main():
    crude = CrudeOptionBuyer()
    nifty = NiftyTradingSession()
    crude_thread = threading.Thread(target=crude.start, name="crude-engine")
    crude_thread.start()
    nifty.run_forever()
    crude_thread.join()


if __name__ == "__main__":
    main()









