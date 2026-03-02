"""Logging setup for the Tk client subprocess.

Self-contained copy of the server-side setup_logging — no imports from src/.
"""

import io
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(logs_dir: Path, verbose: bool = False, is_frozen: bool = False) -> None:
    """Configure logging for both terminal and frozen GUI modes.

    Creates log directory, sets up file rotation, and optionally adds console
    output when running from terminal.

    Args:
        logs_dir: Directory to store log files.
        verbose: If True, set DEBUG level; otherwise WARNING.
        is_frozen: If True, skip console handler (frozen app has no console).
    """
    if not is_frozen and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

    logs_dir.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.WARNING

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    log_file = logs_dir / "stenographer.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if not is_frozen:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    logging.info("Logging initialized: level=%s, frozen=%s", logging.getLevelName(level), is_frozen)
