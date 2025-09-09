from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from loguru import logger as _loguru_logger

# Logging configuration using Loguru

_CONFIGURED: bool = False


# This function creates the log directory, if it doesn't exist
def _ensure_log_dir(log_dir: Path) -> None:
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)


# Configure logger with console and file output, rotation and retention
def configure_logger(
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "14 days",
) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logs_path = Path(log_dir or os.getenv("LOG_DIR") or "logs")
    _ensure_log_dir(logs_path)

    _loguru_logger.remove()

    _loguru_logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level,
        backtrace=False,
        diagnose=False,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    _loguru_logger.add(
        logs_path / "app.log",
        level=log_level,
        rotation=rotation,
        retention=retention,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    _CONFIGURED = True


# This function returns the configured logger, if it's not configured yet, it configures it first
def get_logger():
    if not _CONFIGURED:
        configure_logger()
    return _loguru_logger


