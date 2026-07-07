"""Logging utilities.

All framework modules obtain loggers via :func:`get_logger`, which namespaces
them under the ``quantlab`` root logger. :func:`configure_logging` attaches a
single stream handler with a consistent format and is safe to call multiple
times (idempotent). Libraries embedding quantlab can skip configuration and
route the ``quantlab`` logger through their own handlers instead.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["configure_logging", "get_logger"]

_ROOT_NAME = "quantlab"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int | str | None = None, *, stream=None) -> logging.Logger:
    """Configure the ``quantlab`` root logger with a stream handler.

    Parameters
    ----------
    level:
        Logging level name or number. Defaults to the configured
        ``Settings.log_level``.
    stream:
        Output stream; defaults to ``sys.stderr``.

    Returns
    -------
    logging.Logger
        The configured ``quantlab`` root logger.
    """
    if level is None:
        # Imported lazily to keep quantlab.log importable during config errors.
        from quantlab.config import get_settings

        level = get_settings().log_level

    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)

    if not any(getattr(h, "_quantlab_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        handler._quantlab_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``quantlab``.

    Pass ``__name__``; module paths already rooted at ``quantlab`` are used
    as-is, anything else is nested beneath the root.
    """
    if name == _ROOT_NAME or name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
