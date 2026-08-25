FOLDER : output

Purpose :

Debug/inspection artifacts produced while developing the OCR and
parsing pipeline (not the structured application logs - those live
under `logs/`, written by `logger.py`).

Files :

log.txt

Legacy free-form debug log file (currently empty/unused). Prefer
`logger.get_logger(__name__)` (see root README's Logging section)
for new code - it writes to `logs/<YYYY-MM-DD>/algo.log` instead.

paddle_ocr/

Saved PaddleOCR debug output for manual inspection:

  aoc_ocr_text.txt - raw recognized text dump from a test run

  aoc_res.json - raw PaddleOCR result JSON (texts/scores/boxes) used
  to debug column detection and calibrate `data/column_layout.json`

Note :

This folder is for manual debugging artifacts, not production state.


