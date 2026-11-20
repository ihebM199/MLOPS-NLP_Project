"""Logging setup (plain text locally, compact on Colab)."""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    # Third parties are noisy during long Colab runs.
    for noisy in ("matplotlib", "urllib3", "PIL", "mlflow", "dvc", "git", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str = "deepfake") -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
