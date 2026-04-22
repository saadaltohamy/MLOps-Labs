"""
Centralized colored logger for the Titanic pipeline.

Log levels are color-coded for terminal readability:
  DEBUG    → Cyan
  INFO     → Green
  WARNING  → Yellow
  ERROR    → Red
  CRITICAL → Bold Red
"""

import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds ANSI color codes to log level output."""

    COLORS = {
        logging.DEBUG: "\033[36m",       # Cyan
        logging.INFO: "\033[32m",        # Green
        logging.WARNING: "\033[33m",     # Yellow
        logging.ERROR: "\033[31m",       # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        levelname = record.levelname
        record.levelname = f"{color}{levelname}{self.RESET}"
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a colored logger.

    Args:
        name: Logger name (typically the module name, e.g. "train", "preprocess").
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance with colored console output.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = ColoredFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent log propagation to root logger (avoids duplicate messages)
    logger.propagate = False

    return logger
