"""Single Railway entrypoint for the Crude, NIFTY and PPO paper-trading sessions."""

import logging
import os
import threading
import time

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


RETRY_SECONDS = 60


def run_with_restart(name, factory, retry_seconds=RETRY_SECONDS):
    """Build and run a component forever; log, sleep and rebuild after any failure."""
    while True:
        try:
            factory()
            logger.warning("%s returned; restarting in %ss.", name, retry_seconds)
        except KeyboardInterrupt:
            raise
        except BaseException:
            logger.exception("%s crashed; retrying in %ss.", name, retry_seconds)
        time.sleep(retry_seconds)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_ppo_paper_thread()
    threading.Thread(
        target=run_with_restart,
        args=("crude-engine", lambda: CrudeOptionBuyer().start()),
        name="crude-engine",
        daemon=True,
    ).start()
    run_with_restart("nifty-session", lambda: NiftyTradingSession().run_forever())


if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            raise
        except BaseException:
            logging.getLogger(__name__).exception("Top-level failure; restarting in %ss.", RETRY_SECONDS)
            time.sleep(RETRY_SECONDS)









