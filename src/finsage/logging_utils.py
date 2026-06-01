"""Centralised logging configuration.

The whole project uses the :mod:`logging` module — there are no ``print``
statements. Call :func:`setup_logging` once at process start (CLI entry points,
the FastAPI app, training scripts) and obtain named loggers with
:func:`get_logger` everywhere else.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: str | int = "INFO") -> None:
    """Configure root logging with a Rich handler.

    Idempotent: repeated calls only update the level, so importing modules can
    safely call it without clobbering an existing configuration.

    Args:
        level: Logging level as a name (e.g. ``"INFO"``) or numeric value.
    """
    global _CONFIGURED

    resolved = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(resolved, int):  # unknown level name -> sensible default
        resolved = logging.INFO

    if _CONFIGURED:
        logging.getLogger().setLevel(resolved)
        return

    handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    logging.basicConfig(
        level=resolved,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Logger name, conventionally ``__name__`` of the calling module.

    Returns:
        A standard library :class:`logging.Logger`.
    """
    return logging.getLogger(name)
