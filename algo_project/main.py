"""Broker-only entrypoint for the NIFTY paper-trading session."""

from engine.trading_session import LivePaperTradingSession


def main():
    LivePaperTradingSession(default_symbol="NIFTY").run_forever()


if __name__ == "__main__":
    main()









