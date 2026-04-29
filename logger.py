# logger.py
# Centralized rotating logger for EMG Facial Muscle Detection System

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import LOG_CONFIG


def get_logger(name: str = "emg") -> logging.Logger:
    """
    Returns a configured logger with console + rotating file handlers.

    Usage:
        from logger import get_logger
        log = get_logger(__name__)
        log.info("System started")
    """
    log_dir = Path(LOG_CONFIG["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_CONFIG["log_file"]
    level = getattr(logging, LOG_CONFIG["level"].upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger   # already configured — avoid duplicate handlers

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file  ← this is the log file path from config.py
    fh = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=LOG_CONFIG["max_bytes"],
        backupCount=LOG_CONFIG["backup_count"],
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


log = get_logger("emg.root")
