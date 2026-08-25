FOLDER : readers/chart

Purpose :

Planned chart/candle analysis (TradingView or broker chart
screenshots) to derive market structure - independent of the AOC
option-chain parsing in `readers/aoc_parser.py`.

Files (all currently empty - not implemented) :

candle_reader.py - detect candles / OHLC from a chart image

ce_reader.py / pe_reader.py - read CE/PE specific chart panels

smc_reader.py - Smart Money Concepts structure (order blocks, BOS/CHOCH)

structure_reader.py - general support/resistance/structure detection

Status : ❌ Not implemented. `engine/aoc_engine.py` currently expects a
`market_structure` dict (support/resistance/order_blocks) to be
supplied externally (see `angel_one/market_data.py` for the live
data source used today, via OHLC candles rather than chart-image OCR).
