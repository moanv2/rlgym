"""Structured logger with rich console formatting."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

_INITIALIZED = False


def _init_root_logger(level: int = logging.INFO, log_file: Path | None = None) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_path=False, markup=True),
    ]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers, format="%(message)s", datefmt="%H:%M:%S")
    # Quiet noisy deps
    for noisy in ("urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _INITIALIZED = True


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    if not _INITIALIZED:
        _init_root_logger(log_file=log_file)
    return logging.getLogger(name)


def stdout_flush() -> None:
    sys.stdout.flush()
