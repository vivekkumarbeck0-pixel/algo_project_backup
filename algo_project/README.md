PROJECT : NIFTY AI ALGO



Author : Vivek



Purpose :



AI Based Nifty Option Trading Algorithm



Modules :



1\. Capture Engine

2\. OCR Engine

3\. AOC Reader

4\. CE Reader

5\. PE Reader

6\. TradingView Reader

7\. Strategy Engine

8\. Paper Trading

9\. Live Trading



Status :



Capture Engine : 🔄

OCR : Pending

Strategy : Pending

Paper Trade : Pending



Version : 1.0





Environment setup

Create a `.env` file in the project root (copy `.env.example`) or set these environment variables in your shell:

ANGEL_ONE_API_KEY=your_api_key
ANGEL_ONE_CLIENT_ID=your_client_id
ANGEL_ONE_PASSWORD=your_password
ANGEL_ONE_TOTP=your_totp_secret

Do not commit `.env` to source control. Add `.env` to `.gitignore` so your credentials stay private.

Use `angel_one/login.py` like this:

from angel_one.login import AngelOneLogin

login = AngelOneLogin.from_env()
session = login.connect_from_env()


Configuration

All tunable settings (window titles, capture intervals, strike range,
tolerances, log level, etc.) live in `config.py` as a pydantic
`Settings` model. Override any of them with an `ALGO_` prefixed
environment variable or in `.env`, e.g.:

ALGO_AOC_CAPTURE_INTERVAL=30
ALGO_LOG_LEVEL=DEBUG

OCR pixel column ranges are no longer hardcoded either. They live in
`data/column_layout.json` (auto-created on first run from sane
defaults) and can be recalibrated for a different broker layout or
screen resolution without touching any code. `readers/ocr_reader.py`
also rescales these ranges automatically to the actual captured image
width.


Logging

`logger.py` provides `get_logger(name)`, which writes to both the
console and a daily rotating log file under `logs/<YYYY-MM-DD>/algo.log`.
Set `ALGO_LOG_LEVEL` (default `INFO`) to control verbosity.

    from logger import get_logger
    log = get_logger(__name__)
    log.info("message")


Execution engine (paper trading skeleton)

`engine/` implements the structure of the trading pipeline:

    AOCReader -> AOCParser -> AOCEngine.analyze()
              -> DecisionEngine.decide()   (engine/decision_engine.py)
              -> RiskManager.evaluate()    (engine/risk_manager.py)
              -> OrderManager.execute()    (engine/order_manager.py)
              -> PositionTracker           (engine/position_tracker.py)

Paper-mode execution (simulated fills, in-memory position tracking) is
fully working. Live broker order placement is intentionally left as a
`NotImplementedError` in `OrderManager._execute_live` until a verified
SmartAPI `placeOrder` payload is wired in - see `engine/README.md`.

