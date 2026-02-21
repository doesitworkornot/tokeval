"""Centralized logging configuration for the tokeval project."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

from tokeval.shared.paths import LOG_DIR

# ============================================================
# Default config
# ============================================================

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = LOG_DIR / "tokeval.log"


def build_logging_config(
    level: str = DEFAULT_LOG_LEVEL,
    log_file: Path | None = DEFAULT_LOG_FILE,
) -> dict:
    """Build logging configuration dictionary.

    Args:
        level : Global logging level
        log_file : Path to log file (None = disable file logging)

    Returns:
        dict: logging.config.dictConfig compatible dictionary

    """
    log_file_handler = {}

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        log_file_handler = {
            "file": {
                "class": "logging.FileHandler",
                "filename": str(log_file),
                "formatter": "detailed",
                "level": level,
                "encoding": "utf-8",
            },
        }

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(levelname)s | %(name)s | %(message)s",
            },
            "detailed": {
                "format": ("%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "level": level,
            },
            **log_file_handler,
        },
        "root": {
            "level": level,
            "handlers": ["console"] + (["file"] if log_file else []),
        },
    }

    return config


# ============================================================
# Setup function
# ============================================================

_is_configured = False


def setup_logging(
    level: str = DEFAULT_LOG_LEVEL,
    log_file: Path | None = DEFAULT_LOG_FILE,
) -> None:
    """Configure global logging once.

    Safe to call multiple times.
    """
    root_logger = logging.getLogger()

    # Already configured → skip
    if root_logger.handlers:
        return

    config = build_logging_config(level, log_file)

    logging.config.dictConfig(config)


# ============================================================
# Logger getter
# ============================================================


def get_logger(name: str) -> logging.Logger:
    """Get configured logger.

    Args:
        name (str): Logger name (usually __name__)

    Returns:
        logging.Logger: Configured logger instance

    """
    if not _is_configured:
        setup_logging()

    return logging.getLogger(name)
