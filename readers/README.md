FOLDER : readers

Purpose :

Everything that turns a captured screenshot into structured data:
window detection, OCR, and AOC (Advanced Option Chain) parsing.

Note : this folder also contains two old duplicate files with typo
names (`oce_README.md`, `READE.me`) - kept for history, this
`README.md` is the current one.

Files :

window_detector.py

Finds trading platform windows by title (AOC/TradingView/mStock).
Status: working.

window_capture.py

Placeholder for window-capture helpers. Status: empty.

ocr_reader.py

Wraps PaddleOCR, applies the table Y-range filter and column
detection (both sourced from `config.py` / `data/column_layout.json`
so they're recalibratable without code changes). Status: working.

aoc_reader.py

Runs OCRReader and normalizes raw OCR output into a consistent
`{text, column, x, y, confidence}` item list. Status: working.

aoc_parser.py

Core ~400-line parser: spot price detection, strike extraction,
support/resistance inference, row assembly (CE/STRIKE/PE per
strike), and color-range detection. Thresholds (min/max strike, row Y
tolerance) come from `config.settings`. Status: working, needs more
input validation.

aoc_visual_parser.py

Intended to extract non-OCR visual signals (e.g. highlighted/colored
cells) directly from the image. Status: stub, returns hardcoded data.

aoc_color_detector.py / aoc_crop.py / aoc_patterns.py / aoc_models.py

Planned color analysis, cropping, pattern matching, and shared data
models. Status: empty, not implemented.

pe_reader.py / price_reader.py / signal_reader.py / tradingview_reader.py

Specialized readers for PE data, generic price extraction, signal
generation, and TradingView. Status: signal_reader.py is a dummy
placeholder; price_reader.py is implemented but untested; others are
mostly empty.

Subfolders :

aoc/ - modular per-feature AOC readers (active/color/strike/etc.)

chart/ - candle/structure/SMC chart analysis

utils/ - shared color/image/OCR helper functions

Status :

Window Detection : ✅ Working

OCR : ✅ Working

AOC Parser : ✅ Working (core logic)

Visual Parser : ❌ Stub only

Color/Pattern/Model helpers : ❌ Not implemented
