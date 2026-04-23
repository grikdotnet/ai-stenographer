"""Tests for the shared server logging setup."""

import io
import logging
import sys
from logging.handlers import RotatingFileHandler

import pytest

from src.LoggingSetup import setup_logging


@pytest.fixture(autouse=True)
def restore_logging_state():
    """Restore global logging and stdio after each test."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    original_websockets_level = logging.getLogger("websockets").level
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    yield

    current_handlers = list(root_logger.handlers)
    root_logger.handlers.clear()
    for handler in current_handlers:
        if handler not in original_handlers:
            handler.close()
    for handler in original_handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(original_level)
    logging.getLogger("websockets").setLevel(original_websockets_level)
    sys.stdout = original_stdout
    sys.stderr = original_stderr


def _flush_root_handlers() -> None:
    """Flush all root handlers to make file assertions deterministic."""
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_setup_logging_writes_warning_to_rotating_log_file(tmp_path):
    """Shared logging setup should create the log file and persist warnings."""
    logs_dir = tmp_path / "logs"

    setup_logging(logs_dir=logs_dir, verbose=False, is_frozen=True)
    logging.warning("server warning")
    _flush_root_handlers()

    root_logger = logging.getLogger()
    log_file = logs_dir / "stenographer.log"

    assert root_logger.level == logging.WARNING
    assert log_file.exists()
    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0], RotatingFileHandler)

    content = log_file.read_text(encoding="utf-8")
    assert "server warning" in content
    assert "[WARNING]" in content


def test_setup_logging_adds_console_handler_in_terminal_mode(tmp_path):
    """Terminal mode should keep both file and console logging enabled."""
    logs_dir = tmp_path / "logs"
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    setup_logging(logs_dir=logs_dir, verbose=False, is_frozen=False)

    handlers = logging.getLogger().handlers

    assert any(isinstance(handler, RotatingFileHandler) for handler in handlers)
    assert any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, RotatingFileHandler)
        for handler in handlers
    )


def test_setup_logging_keeps_root_debug_and_suppresses_websockets(tmp_path):
    """Verbose logging should keep app debug while muting websockets frame dumps."""
    logs_dir = tmp_path / "logs"

    setup_logging(logs_dir=logs_dir, verbose=True, is_frozen=True)

    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("websockets").level == logging.WARNING
