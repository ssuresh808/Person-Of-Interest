"""Centralized logging configuration.

Uses loguru because the default stdlib logger config dance is not the kind
of thing I want to do once per script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logging(
    level: str | None = None,
    log_file: Path | None = None,
    quiet: bool = False,
) -> None:
    """Configure loguru sinks. Idempotent — safe to call multiple times.

    Args:
        level: Log level string. Defaults to LOG_LEVEL env var, then INFO.
        log_file: Optional file to also write logs to.
        quiet: If True, only WARNING and above go to stderr.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = level or os.environ.get("LOG_LEVEL", "INFO")
    if quiet:
        level = "WARNING"

    logger.remove()  # Drop default handler

    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            rotation="100 MB",
            retention=5,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
            ),
        )

    _CONFIGURED = True


def get_logger(name: str | None = None):
    """Return the configured loguru logger.

    The `name` argument exists for API-compatibility with stdlib logging,
    so existing callers don't have to change. Loguru is global anyway.
    """
    if not _CONFIGURED:
        setup_logging()
    return logger
