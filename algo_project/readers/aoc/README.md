FOLDER : readers/aoc

Purpose :

Planned modular AOC readers, split by concern (one file per feature)
instead of the single large `aoc_parser.py`.

Files (all currently empty - not implemented) :

active_reader.py - detect the currently active/selected strike

color_reader.py - read green/red cell colors directly from pixels

crop.py - crop the AOC image into per-column regions before OCR

percent_reader.py - read green/red percentage indicators

sr_reader.py - support/resistance level extraction

strike_reader.py - strike column extraction

Status : ❌ Not implemented. `readers/aoc_parser.py` currently handles
all of this in one file; these were intended as a future refactor.
