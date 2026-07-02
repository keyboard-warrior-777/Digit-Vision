"""
Centralized logging configuration for DigitVision.

Design Decision:
    Every module in this project calls get_logger(__name__) instead of
    using print(). This gives us:
      - Timestamps on every log line
      - Log level filtering (INFO in prod, DEBUG when debugging)
      - Persistent log file for post-training review
      - Consistent format across all modules

Interview Answer:
    "I replaced all print() calls with a logging module. In production,
    you can't have bare prints — you need timestamped, level-filtered,
    persistent logs you can review after a training run."

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
    logger.warning("No GPU detected — training on CPU")
    logger.error("Failed to load model: %s", path)
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Logs are written to both stdout and a persistent file at logs/digitvision.log.
    Calling this function multiple times with the same name returns the same
    logger (idempotent) — duplicate handlers are never added.

    Args:
        name: The module name. Always pass ``__name__`` from the caller.
        level: Logging level. Defaults to ``logging.INFO``.

    Returns:
        A fully configured :class:`logging.Logger` instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Model loaded successfully")
    """
    # Resolve logs directory relative to this file: src/ → project root → logs/
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Guard: if handlers already exist this logger was already configured.
    # Without this guard, every import would add duplicate handlers.
    if logger.handlers:
        return logger

    logger.setLevel(level)

    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(_formatter)
    logger.addHandler(console_handler)

    # ── File handler ─────────────────────────────────────────────────────────
    log_file = logs_dir / "digitvision.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.DEBUG)  # File always captures DEBUG+
    file_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)

    # Prevent log messages from propagating to the root logger
    # (avoids duplicate output if the root logger is also configured).
    logger.propagate = False

    return logger
