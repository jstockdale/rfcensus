"""Structured logging for rfcensus.

We use the standard library `logging` module but configure it with
`rich` for human-friendly console output. Every module in rfcensus
gets its logger via `get_logger(__name__)`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

from rich.logging import RichHandler

_CONFIGURED: bool = False
_DEFAULT_FORMAT: Final[str] = "%(message)s"
_FILE_FORMAT: Final[str] = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(
    level: str = "INFO",
    logfile: Path | None = None,
    quiet: bool = False,
) -> None:
    """Configure root logger. Safe to call multiple times.

    Parameters
    ----------
    level:
        Log level for console output. One of DEBUG / INFO / WARNING / ERROR.
    logfile:
        Optional path to a file that receives all log output at DEBUG level
        regardless of `level`. Useful for `rfcensus doctor` style post-mortems.
    quiet:
        If true, console output is suppressed entirely. File logging still
        works if `logfile` is supplied.
    """
    global _CONFIGURED

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    # Reset existing handlers on reconfigure
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)  # Let handlers filter

    if not quiet:
        console_handler = RichHandler(
            level=numeric_level,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=False,
        )
        console_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        root.addHandler(console_handler)

    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logfile, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)

    # Suppress chatty third-party loggers unless explicitly enabled
    if os.environ.get("RFCENSUS_DEBUG_ASYNCIO") != "1":
        logging.getLogger("asyncio").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with sensible defaults.

    Automatically configures root logger if it hasn't been done yet.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
