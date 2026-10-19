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
import re
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED_LOGGERS: set[str] = set()

_SENSITIVE_HEADER_NAMES = (
    "authorization|proxy-authorization|cookie|set-cookie|x-privatekey|"
    "x-clientlocalip|x-clientpublicip|x-macaddress|privatekey|apikey|api-key|"
    "access[_-]?token|refresh[_-]?token|feed[_-]?token"
)
_QUOTED_SENSITIVE_VALUE = re.compile(
    rf"(?i)(?P<prefix>['\"]?(?:{_SENSITIVE_HEADER_NAMES})['\"]?\s*[:=]\s*)"
    r"(?P<quote>['\"])(?P<value>.*?)(?P=quote)"
)
_UNQUOTED_SENSITIVE_VALUE = re.compile(
    rf"(?i)(\b(?:{_SENSITIVE_HEADER_NAMES})\b\s*[:=]\s*)"
    r"([^\s,}\]]+(?:\s+[^\s,}\]]+)?)"
)


def _redact_sensitive_text(text: str) -> str:
    text = _QUOTED_SENSITIVE_VALUE.sub(
        lambda match: f"{match.group('prefix')}{match.group('quote')}[REDACTED]{match.group('quote')}",
        text,
    )
    return _UNQUOTED_SENSITIVE_VALUE.sub(r"\1[REDACTED]", text)


_previous_record_factory = logging.getLogRecordFactory()


def _redacting_record_factory(*args, **kwargs):
    record = _previous_record_factory(*args, **kwargs)
    try:
        record.msg = _redact_sensitive_text(record.getMessage())
        record.args = ()
    except Exception:
        pass
    return record


logging.setLogRecordFactory(_redacting_record_factory)


class _SuppressMissingGreekData(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage().lower()
        except Exception:
            return True
        return not ("ab9019" in message and "optiongreek" in message)


logging.getLogger("logzero_default").addFilter(_SuppressMissingGreekData())

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
