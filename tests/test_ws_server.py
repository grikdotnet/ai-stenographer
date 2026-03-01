"""Tests for WsServer start/stop lifecycle.

Strategy: real asyncio loop + real websockets.serve() on port 0;
SessionManager is mocked since no client connects in these tests.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ServerApplicationState import ServerApplicationState
from src.server.WsServer import WsServer


def _make_server(port: int = 0) -> WsServer:
    session_manager = MagicMock()
    session_manager.create_session = AsyncMock()
    session_manager.destroy_session = AsyncMock()
    app_state = ServerApplicationState()
    return WsServer(session_manager=session_manager, app_state=app_state, port=port)


class TestWsServerStartStop:
    """WsServer lifecycle: bind, stop, thread exit."""

    def test_start_binds_port(self):
        server = _make_server()
        try:
            server.start()
            assert server.port > 0
        finally:
            server.stop()
            server.join(timeout=5.0)

    def test_stop_does_not_raise(self):
        server = _make_server()
        server.start()
        server.stop()
        server.join(timeout=5.0)

    def test_stop_thread_exits(self):
        server = _make_server()
        server.start()
        assert server._thread is not None
        assert server._thread.is_alive()
        server.stop()
        server.join(timeout=5.0)
        assert not server._thread.is_alive()

    def test_stop_before_start_is_noop(self):
        server = _make_server()
        server.stop()

    def test_double_stop_is_noop(self):
        server = _make_server()
        server.start()
        server.stop()
        server.stop()
        server.join(timeout=5.0)

    def test_no_runtime_error_on_stop(self, caplog):
        """Stopping must not produce 'Event loop stopped before Future completed'."""
        server = _make_server()
        with caplog.at_level(logging.ERROR):
            server.start()
            server.stop()
            server.join(timeout=5.0)
        assert "Event loop stopped before Future completed" not in caplog.text
        assert "Task was destroyed but it is pending" not in caplog.text
