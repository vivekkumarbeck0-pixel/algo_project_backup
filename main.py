"""Single Railway entrypoint for the Crude, NIFTY and PPO paper-trading sessions."""

import logging
import os
import threading

from trading_crude import CrudeOptionBuyer
from trading_nifty import NiftyTradingSession

logger = logging.getLogger(__name__)


def start_ppo_paper_thread():
    """Run the PPO paper runner in a daemon thread; set ENABLE_PPO_PAPER=0 to disable."""
    if os.getenv("ENABLE_PPO_PAPER", "1").strip().lower() in {"0", "false", "no"}:
        logger.info("PPO paper runner disabled via ENABLE_PPO_PAPER.")
        return None
    try:
        from crude_ppo_paper_runner import run_live_forever
    except Exception:
        logger.exception("PPO paper runner unavailable; continuing without it.")
        return None
    thread = threading.Thread(target=run_live_forever, name="ppo-paper-runner", daemon=True)
    thread.start()
    return thread


def main():
    crude = CrudeOptionBuyer()
    nifty = NiftyTradingSession()
    start_ppo_paper_thread()
    crude_thread = threading.Thread(target=crude.start, name="crude-engine")
    crude_thread.start()
    nifty.run_forever()
    crude_thread.join()


if __name__ == "__main__":
    main()









