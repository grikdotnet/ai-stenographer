"""Tests for WsServer start/stop lifecycle.

Strategy: real asyncio loop + real websockets.serve() on port 0;
SessionManager is mocked since no client connects in these tests.
"""

import asyncio
import json
import logging
import socket
from queue import SimpleQueue
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from conftest import encode_audio_frame
from src.ApplicationState import ApplicationState
from src.network.types import WsAudioFrame
from src.server.WsServer import DEFAULT_SERVER_HOST, WsServer


def _make_server(port: int = 0) -> WsServer:
    session_manager = MagicMock()
    session_manager.create_session = AsyncMock()
    session_manager.destroy_session = AsyncMock()
    session_manager.close_session = AsyncMock()
    session_manager.close_all_sessions = AsyncMock()
    return WsServer(
        session_manager=session_manager,
        app_state=ApplicationState(),
        command_controller=MagicMock(),
        port=port,
    )


class TestWsServerStartStop:
    """WsServer lifecycle: bind, stop, thread exit."""

    def test_is_running_reflects_thread_state(self) -> None:
        server = _make_server()
        assert server.is_running() is False

        try:
            server.start()
            assert server.is_running() is True
        finally:
            server.stop()
            server.join(timeout=5.0)

        assert server.is_running() is False

    def test_start_binds_port(self):
        server = _make_server()
        try:
            server.start()
            assert server.port > 0
        finally:
            server.stop()
            server.join(timeout=5.0)

    def test_stop_thread_exits(self):
        server = _make_server()
        server.start()
        assert server._thread is not None
        assert server._thread.is_alive()
        server.stop()
        server.join(timeout=2.0)
        assert not server._thread.is_alive()

    def test_double_stop_is_noop(self):
        server = _make_server()
        server.stop()
        server.start()
        server.stop()
        server.stop()
        server.join(timeout=2.0)

    def test_no_runtime_error_on_stop(self, caplog):
        """Stopping must not produce 'Event loop stopped before Future completed'."""
        server = _make_server()
        with caplog.at_level(logging.ERROR):
            server.start()
            server.stop()
            server.join(timeout=2.0)
        assert "Event loop stopped before Future completed" not in caplog.text
        assert "Task was destroyed but it is pending" not in caplog.text


class TestWsServerShutdown:
    """WsServer.stop should trigger async shutdown without a broadcast queue."""

    def test_stop_triggers_close_all_sessions(self):
        server = _make_server()
        server.start()
        server.stop()
        server.join(timeout=2.0)
        server._session_manager.close_all_sessions.assert_awaited_once()


class TestWsServerStartupError:
    def test_start_raises_when_websockets_serve_raises(self) -> None:
        server = _make_server()

        with patch("websockets.serve", side_effect=OSError("address already in use")):
            with pytest.raises(OSError, match="address already in use"):
                server.start()

        server.join(timeout=2.0)

    def test_start_raises_on_real_port_conflict(self) -> None:
        with socket.socket() as sock:
            sock.bind((DEFAULT_SERVER_HOST, 0))
            conflict_port = sock.getsockname()[1]

        server1 = _make_server(port=conflict_port)
        server2 = _make_server(port=conflict_port)

        try:
            server1.start()

            with pytest.raises(OSError):
                server2.start()
        finally:
            server1.stop()
            server1.join(timeout=2.0)
            server2.stop()
            server2.join(timeout=2.0)

    def test_stop_and_join_safe_after_failed_start(self) -> None:
        server = _make_server()

        with patch("websockets.serve", side_effect=OSError("address already in use")):
            with pytest.raises(OSError):
                server.start()

        server.stop()
        server.join(timeout=2.0)


class _FakeSession:
    """Minimal ClientSession stand-in for connection handler tests."""

    def __init__(self, session_id: str) -> None:
        import queue

        self._session_id = session_id
        self.chunk_queue: queue.Queue = queue.Queue()

    @property
    def session_id(self) -> str:
        return self._session_id


class _WebSocketSequence:
    """Fake websocket that yields a fixed sequence and captures sent text frames."""

    def __init__(self, messages: list[object]) -> None:
        self._messages = iter(messages)
        self.sent: list[str] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._messages)
        except StopIteration:
            raise StopAsyncIteration

    async def send(self, message: str) -> None:
        self.sent.append(message)


class TestWsServerConnectionHandling:
    """WsServer routes audio and commands through the new orchestration path."""

    def test_close_session_command_uses_controller_and_destroys_session(self) -> None:
        session_manager = MagicMock()
        session_manager.destroy_session = AsyncMock()
        session_manager.close_session = AsyncMock()
        session_manager.close_all_sessions = AsyncMock()
        session_manager.create_session = AsyncMock(return_value=_FakeSession("session-1"))

        command_controller = MagicMock()
        command_controller.handle.return_value = (True, None)

        server = WsServer(
            session_manager=session_manager,
            app_state=ApplicationState(),
            command_controller=command_controller,
        )

        ws = _WebSocketSequence(
            [
                json.dumps(
                    {
                        "type": "control_command",
                        "session_id": "session-1",
                        "command": "close_session",
                        "timestamp": 1000.0,
                    }
                )
            ]
        )

        asyncio.run(server._handle_connection(ws))

        command_controller.handle.assert_called_once()
        session_manager.close_session.assert_awaited_once_with(
            "session-1",
            reason="close_session",
        )

    def test_audio_frame_is_enqueued_when_server_running(self) -> None:
        session = _FakeSession("session-1")
        session_manager = MagicMock()
        session_manager.destroy_session = AsyncMock()
        session_manager.close_session = AsyncMock()
        session_manager.close_all_sessions = AsyncMock()
        session_manager.create_session = AsyncMock(return_value=session)

        app_state = ApplicationState()
        app_state.set_state("running")

        server = WsServer(
            session_manager=session_manager,
            app_state=app_state,
            command_controller=MagicMock(),
        )

        frame = WsAudioFrame(
            session_id="session-1",
            chunk_id=1,
            timestamp=1000.0,
            audio=np.zeros(16, dtype=np.float32),
        )
        ws = _WebSocketSequence([encode_audio_frame(frame)])

        asyncio.run(server._handle_connection(ws))

        assert session.chunk_queue.qsize() == 1
        session_manager.destroy_session.assert_awaited_once_with("session-1")

    def test_audio_frame_when_not_running_sends_model_not_ready(self) -> None:
        session_manager = MagicMock()
        session_manager.destroy_session = AsyncMock()
        session_manager.close_session = AsyncMock()
        session_manager.close_all_sessions = AsyncMock()
        session_manager.create_session = AsyncMock(return_value=_FakeSession("session-1"))

        app_state = ApplicationState()
        app_state.set_state("waiting_for_model")

        server = WsServer(
            session_manager=session_manager,
            app_state=app_state,
            command_controller=MagicMock(),
        )

        frame = WsAudioFrame(
            session_id="session-1",
            chunk_id=1,
            timestamp=1000.0,
            audio=np.zeros(16, dtype=np.float32),
        )
        ws = _WebSocketSequence([encode_audio_frame(frame)])

        asyncio.run(server._handle_connection(ws))

        assert len(ws.sent) == 1
        obj = json.loads(ws.sent[0])
        assert obj["error_code"] == "MODEL_NOT_READY"
