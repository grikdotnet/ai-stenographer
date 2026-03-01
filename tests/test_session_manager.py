"""Tests for SessionManager (Phase 7).

Strategy: patch ClientSession to avoid loading real ONNX models.
Tests verify session lifecycle, unique IDs, session_created wire message,
and multi-session independence.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ServerApplicationState import ServerApplicationState
from src.server.RecognizerService import RecognizerService
from src.server.SessionManager import SessionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG = {
    "audio": {
        "sample_rate": 16000,
        "chunk_duration": 0.032,
        "silence_energy_threshold": 1.5,
        "rms_normalization": {
            "target_rms": 0.05,
            "silence_threshold": 0.001,
            "gain_smoothing": 0.9,
        },
    },
    "vad": {"frame_duration_ms": 32, "threshold": 0.5},
    "windowing": {"max_speech_duration_ms": 3000, "max_window_duration": 7.0},
}
_VAD_MODEL_PATH = Path("/fake/silero_vad.onnx")


def _make_app_state() -> ServerApplicationState:
    app_state = ServerApplicationState()
    app_state.set_state("running")
    return app_state


def _make_recognizer_service(app_state: ServerApplicationState) -> RecognizerService:
    recognizer = MagicMock()
    recognizer.recognize_window.return_value = None
    service = RecognizerService(recognizer=recognizer, app_state=app_state)
    service.start()
    return service


def _make_manager(
    app_state: ServerApplicationState | None = None,
    recognizer_service: RecognizerService | None = None,
) -> tuple[SessionManager, ServerApplicationState, RecognizerService]:
    if app_state is None:
        app_state = _make_app_state()
    if recognizer_service is None:
        recognizer_service = _make_recognizer_service(app_state)
    manager = SessionManager(
        recognizer_service=recognizer_service,
        app_state=app_state,
        config=_CONFIG,
        vad_model_path=_VAD_MODEL_PATH,
    )
    return manager, app_state, recognizer_service


def _make_mock_websocket() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


def _make_mock_session(session_id: str) -> MagicMock:
    session = MagicMock()
    session.start = AsyncMock()
    session.close = AsyncMock()
    session._session_id = session_id
    session.chunk_queue = MagicMock()
    return session


# ---------------------------------------------------------------------------
# create_session: unique IDs and session_created wire message
# ---------------------------------------------------------------------------

class TestCreateSession:
    def test_create_session_returns_unique_session_ids(self) -> None:
        manager, app_state, recognizer_service = _make_manager()

        mock_session_a = _make_mock_session("id-a")
        mock_session_b = _make_mock_session("id-b")

        loop = asyncio.new_event_loop()

        with patch("src.server.SessionManager.ClientSession") as MockClientSession:
            MockClientSession.side_effect = [mock_session_a, mock_session_b]

            ws = _make_mock_websocket()
            loop.run_until_complete(manager.create_session(ws, loop))
            loop.run_until_complete(manager.create_session(ws, loop))

        session_ids = list(manager._sessions.keys())
        assert len(session_ids) == 2
        assert session_ids[0] != session_ids[1]

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()

    def test_create_session_sends_session_created_frame(self) -> None:
        manager, app_state, recognizer_service = _make_manager()

        mock_session = _make_mock_session("irrelevant-id")
        loop = asyncio.new_event_loop()
        ws = _make_mock_websocket()

        with patch("src.server.SessionManager.ClientSession", return_value=mock_session):
            loop.run_until_complete(manager.create_session(ws, loop))

        assert ws.send.called
        sent_json = json.loads(ws.send.call_args[0][0])
        assert sent_json["type"] == "session_created"
        assert "session_id" in sent_json
        assert sent_json["protocol_version"] == "v1"

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()

    def test_create_session_sends_server_config_in_frame(self) -> None:
        manager, app_state, recognizer_service = _make_manager()

        mock_session = _make_mock_session("irrelevant-id")
        loop = asyncio.new_event_loop()
        ws = _make_mock_websocket()

        with patch("src.server.SessionManager.ClientSession", return_value=mock_session):
            loop.run_until_complete(manager.create_session(ws, loop))

        sent_json = json.loads(ws.send.call_args[0][0])
        server_config = sent_json["server_config"]
        assert server_config["sample_rate"] == 16000
        assert server_config["audio_dtype"] == "float32"
        assert server_config["channels"] == 1

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()

    def test_create_session_increments_session_index(self) -> None:
        manager, app_state, recognizer_service = _make_manager()

        loop = asyncio.new_event_loop()
        ws = _make_mock_websocket()

        with patch("src.server.SessionManager.ClientSession") as MockClientSession:
            MockClientSession.return_value = _make_mock_session("id-x")

            loop.run_until_complete(manager.create_session(ws, loop))
            first_index = MockClientSession.call_args.kwargs.get(
                "session_index", MockClientSession.call_args[1].get("session_index")
            )

            MockClientSession.return_value = _make_mock_session("id-y")
            loop.run_until_complete(manager.create_session(ws, loop))
            second_index = MockClientSession.call_args.kwargs.get(
                "session_index", MockClientSession.call_args[1].get("session_index")
            )

        assert second_index == first_index + 1

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()

    def test_create_session_calls_session_start(self) -> None:
        manager, app_state, recognizer_service = _make_manager()

        mock_session = _make_mock_session("some-id")
        loop = asyncio.new_event_loop()
        ws = _make_mock_websocket()

        with patch("src.server.SessionManager.ClientSession", return_value=mock_session):
            loop.run_until_complete(manager.create_session(ws, loop))

        mock_session.start.assert_awaited_once()

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()


# ---------------------------------------------------------------------------
# destroy_session: stops all session components
# ---------------------------------------------------------------------------

class TestDestroySession:
    def test_destroy_session_calls_session_close(self) -> None:
        manager, app_state, recognizer_service = _make_manager()

        mock_session = _make_mock_session("close-me")
        loop = asyncio.new_event_loop()
        ws = _make_mock_websocket()

        with patch("src.server.SessionManager.ClientSession", return_value=mock_session):
            loop.run_until_complete(manager.create_session(ws, loop))

        session_id = list(manager._sessions.keys())[0]
        loop.run_until_complete(manager.destroy_session(session_id))

        mock_session.close.assert_awaited_once()

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()

    def test_destroy_session_removes_it_from_active_sessions(self) -> None:
        manager, app_state, recognizer_service = _make_manager()

        mock_session = _make_mock_session("remove-me")
        loop = asyncio.new_event_loop()
        ws = _make_mock_websocket()

        with patch("src.server.SessionManager.ClientSession", return_value=mock_session):
            loop.run_until_complete(manager.create_session(ws, loop))

        session_id = list(manager._sessions.keys())[0]
        loop.run_until_complete(manager.destroy_session(session_id))

        assert session_id not in manager._sessions

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()

    def test_destroy_unknown_session_does_not_raise(self) -> None:
        manager, app_state, recognizer_service = _make_manager()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(manager.destroy_session("nonexistent-id"))

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()


# ---------------------------------------------------------------------------
# Two sessions run independently
# ---------------------------------------------------------------------------

class TestTwoSessionsIndependence:
    def test_two_sessions_have_different_session_ids(self) -> None:
        manager, app_state, recognizer_service = _make_manager()

        loop = asyncio.new_event_loop()
        ws = _make_mock_websocket()

        with patch("src.server.SessionManager.ClientSession") as MockClientSession:
            MockClientSession.side_effect = [
                _make_mock_session("alpha"),
                _make_mock_session("beta"),
            ]
            loop.run_until_complete(manager.create_session(ws, loop))
            loop.run_until_complete(manager.create_session(ws, loop))

        assert len(manager._sessions) == 2
        ids = list(manager._sessions.keys())
        assert ids[0] != ids[1]

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()

    def test_destroy_one_session_leaves_other_active(self) -> None:
        manager, app_state, recognizer_service = _make_manager()

        loop = asyncio.new_event_loop()
        ws = _make_mock_websocket()

        with patch("src.server.SessionManager.ClientSession") as MockClientSession:
            MockClientSession.side_effect = [
                _make_mock_session("alpha"),
                _make_mock_session("beta"),
            ]
            loop.run_until_complete(manager.create_session(ws, loop))
            loop.run_until_complete(manager.create_session(ws, loop))

        all_ids = list(manager._sessions.keys())
        first_id = all_ids[0]
        second_id = all_ids[1]

        loop.run_until_complete(manager.destroy_session(first_id))

        assert first_id not in manager._sessions
        assert second_id in manager._sessions

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()

    def test_close_all_sessions_closes_every_session(self) -> None:
        manager, app_state, recognizer_service = _make_manager()

        loop = asyncio.new_event_loop()
        ws = _make_mock_websocket()

        mock_a = _make_mock_session("alpha")
        mock_b = _make_mock_session("beta")

        with patch("src.server.SessionManager.ClientSession") as MockClientSession:
            MockClientSession.side_effect = [mock_a, mock_b]
            loop.run_until_complete(manager.create_session(ws, loop))
            loop.run_until_complete(manager.create_session(ws, loop))

        loop.run_until_complete(manager.close_all_sessions())

        mock_a.close.assert_awaited_once()
        mock_b.close.assert_awaited_once()
        assert len(manager._sessions) == 0

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()
