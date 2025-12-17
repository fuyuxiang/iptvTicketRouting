from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from pythonjsonlogger import jsonlogger


def setup_logging(cfg: dict) -> None:
    level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)
    log_file = cfg.get("file")

    root = logging.getLogger()
    root.setLevel(level)

    # prevent duplicate handlers in uvicorn reload / tests
    if root.handlers:
        return

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(pathname)s %(lineno)d"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
