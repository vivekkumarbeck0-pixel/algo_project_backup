FOLDER : data

Purpose :

Live captured images, OCR-calibration config, and reference
instrument data. Note: an older duplicate `MEADME.md` (typo) exists
here for history - this `README.md` is the current one.

Files :

aoc.png / tradingview.png / ce.png / pe.png

Latest captured screenshots of the AOC calculator, TradingView/index
chart, CE chart, and PE chart. Overwritten continuously by
`capture/` - storage never grows.

column_layout.json

Auto-created by `config.py` on first run. Defines the OCR pixel
column ranges (CE/STRIKE/PE) calibrated against a reference image
width. Edit this file to recalibrate for a different broker layout
or screen resolution - no code changes needed. See
`config.ColumnLayout` / `config.load_column_layout`.

angel_one_instruments.json

Cached Angel One instrument master (contract list) downloaded via
`angel_one/instrument_reader.py`, used to resolve trading
symbols/tokens.

aoc_visual_result.json

Sample/last output of `readers/aoc_visual_parser.py` (currently a
stub).

Note :

Images are overwritten continuously. Storage never increases.
