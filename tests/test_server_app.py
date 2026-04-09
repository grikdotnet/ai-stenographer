"""Tests for ServerApp (Phase 8).

Strategy: mock WsServer, RecognizerService, and SessionManager to avoid
loading real ONNX models or binding real sockets.

Tests verify:
- ServerApp.start() starts WsServer and RecognizerService
- bound port is available after start()
- ServerApp.stop() transitions ApplicationState to shutdown
- session_created is emitted when a client connects (integration via WsServer mock)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.ApplicationState import ApplicationState
from src.server.ServerApp import ServerApp


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

_MODELS_DIR = Path("/fake/models")


@pytest.fixture
def server_app_mocks():
    """Provides a ServerApp with mock WsServer and RecognizerService.

    Yields the app, mock_ws_server, and mock_rs inside the patch context,
    so tests can call start()/stop() and inspect mock state.
    """
    mock_ws_server = MagicMock()
    mock_ws_server.port = 9876

    mock_rs = MagicMock()

    with (
        patch("src.server.ServerApp.WsServer", return_value=mock_ws_server),
        patch("src.server.ServerApp.RecognizerService", return_value=mock_rs),
        patch("src.server.ServerApp.SessionManager"),
    ):
        app = ServerApp(
            config=_CONFIG,
            vad_model_path=_MODELS_DIR / "silero_vad" / "silero_vad.onnx",
            app_state=ApplicationState(),
        )
        yield app, mock_ws_server, mock_rs


class TestServerAppStart:
    def test_start(self, server_app_mocks) -> None:
        app, mock_ws_server, mock_rs = server_app_mocks
        mock_ws_server.port = 12345
        app.start()
        mock_ws_server.start.assert_called_once_with()
        mock_rs.start.assert_called_once_with()
        assert app.app_state.get_state() == "running", "app state should be 'running' after start()"
        assert app.port == 12345, f"expected port 12345, got {app.port}"


class TestServerAppStop:
    def test_stop_transitions_state_to_shutdown(self, server_app_mocks) -> None:
        app, _, _ = server_app_mocks
        app.start()
        app.stop()
        assert app.app_state.get_state() == "shutdown", "app state should be 'shutdown' after stop()"

    def test_stop_stops_ws_server(self, server_app_mocks) -> None:
        app, mock_ws_server, _ = server_app_mocks
        app.start()
        app.stop()
        mock_ws_server.stop.assert_called_once_with()

    def test_stop_is_idempotent(self, server_app_mocks) -> None:
        app, _, _ = server_app_mocks
        app.start()
        app.stop()
        app.stop()  # Must not raise
        assert app.app_state.get_state() == "shutdown", "app state should remain 'shutdown' after second stop()"

    def test_stop_joins_ws_server(self, server_app_mocks) -> None:
        app, mock_ws_server, _ = server_app_mocks
        app.start()
        app.stop()
        mock_ws_server.join.assert_called_once()

    def test_stop_joins_recognizer_service(self, server_app_mocks) -> None:
        app, _, mock_rs = server_app_mocks
        app.start()
        app.stop()
        mock_rs.join.assert_called_once()


class TestAttachRecognizer:
    def test_attach_recognizer_delegates_to_recognizer_service(self, server_app_mocks) -> None:
        app, _, mock_rs = server_app_mocks
        recognizer = MagicMock()
        app.attach_recognizer(recognizer)
        mock_rs.attach_recognizer.assert_called_once_with(recognizer)
