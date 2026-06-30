"""Small shared helpers for auditable terminal and per-run file logging."""

from __future__ import annotations

import logging
from pathlib import Path
import sys


def configure_run_logger(
    output_dir: Path,
    *,
    logger_name: str,
    file_name: str,
) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{logger_name}.{output_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / file_name, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


__all__ = ["configure_run_logger"]
