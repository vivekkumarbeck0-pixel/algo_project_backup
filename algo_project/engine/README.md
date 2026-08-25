FOLDER : engine

Purpose :

Decision-making and trade-execution logic. Takes structured AOC data
(from readers/) and turns it into a final, risk-checked trading
decision, then (optionally) executes it.

Pipeline :

AOCReader -> AOCParser -> AOCEngine.analyze()
          -> DecisionEngine.decide()
          -> RiskManager.evaluate()
          -> OrderManager.execute()
          -> PositionTracker

Files :

aoc_engine.py

Reads an AOC snapshot + optional market structure (support/
resistance/order blocks) and produces a scenario + basic BUY/SELL/
HOLD signal. Status: working.

price_engine.py

Derives option-level price context (ATM strike, CE/PE quote at the
market strike) from an AOCParser snapshot. Status: working skeleton.

decision_engine.py

Combines AOCEngine's signal with PriceEngine's context and
RiskManager approval into one final `Decision` object. The
scenario -> action mapping is a placeholder; replace with a
validated strategy before trusting it with real money.
Status: working skeleton, strategy logic marked with TODOs.

risk_manager.py

Pre-trade checks: max open positions, max quantity per trade, daily
loss limit. Extend `RiskManager.evaluate()` with more rules (margin,
volatility, news blackout, etc.) as the strategy matures.
Status: working (generic, config-less defaults in `RiskLimits`).

order_manager.py

Routes an approved Decision to either a paper fill (fully working,
simulated, updates PositionTracker) or a live broker order.

Live order placement raises NotImplementedError on purpose - it
needs a verified SmartAPI `placeOrder` payload (exchange,
symboltoken, producttype, ordertype, price, stoploss, variety, etc.)
tested against Angel One's docs before being enabled.

position_tracker.py

In-memory tracker for open/closed positions and realized P&L. No
database - state is lost on restart.

Status :

AOC Engine : ✅ Working

Price Engine : ✅ Working (basic)

Decision Engine : 🔄 Structure done, strategy rules are placeholders

Risk Manager : 🔄 Structure done, generic limits

Order Manager : 🔄 Paper mode working, live mode NOT implemented

Position Tracker : ✅ Working
