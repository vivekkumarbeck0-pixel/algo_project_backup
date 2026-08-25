FOLDER : readers/utils

Purpose :

Planned shared helper functions for the reader modules, to avoid
duplicating image/color/OCR utility code across `aoc_parser.py`,
`ocr_reader.py`, and the `aoc/` and `chart/` submodules.

Files (all currently empty - not implemented) :

color.py - RGB/HSV helpers for green/red cell detection

image.py - cropping/resizing/preprocessing helpers

ocr.py - shared OCR post-processing helpers (text cleanup, number
parsing, etc.)

Status : ❌ Not implemented. Equivalent logic currently lives inline in
`readers/aoc_parser.py` and `readers/ocr_reader.py`.
