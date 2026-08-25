"""Centralized logging setup for the NIFTY AI ALGO project.

Replaces scattered `print()` calls with proper structured logging that
writes to both the console and a daily rotating file under `logs/`.

Usage:
    from logger import get_logger

    log = get_logger(__name__)
    log.info("AOC image refreshed: %s", image_path)
    log.error("Failed to capture window", exc_info=True)
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED_LOGGERS: set[str] = set()

_LOG_ROOT = Path(__file__).resolve().with_name("logs")


def _log_file_path() -> Path:
    day_folder = _LOG_ROOT / datetime.now().strftime("%Y-%m-%d")
    day_folder.mkdir(parents=True, exist_ok=True)
    return day_folder / "algo.log"


def get_logger(name: str = "algo") -> logging.Logger:
    """Return a configured logger, wiring handlers only once per name."""

    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        _log_file_path(), maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger
