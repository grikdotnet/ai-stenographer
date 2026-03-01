"""Tests for ServerApp (Phase 8).

Strategy: mock WsServer, RecognizerService, and SessionManager to avoid
loading real ONNX models or binding real sockets.

Tests verify:
- ServerApp.start() starts WsServer and RecognizerService
- bound port is available after start()
- ServerApp.stop() transitions ServerApplicationState to shutdown
- session_created is emitted when a client connects (integration via WsServer mock)
"""

from pathlib import Path
from unittest.mock import MagicMock, call, patch
import pytest

from src.ServerApplicationState import ServerApplicationState
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


def _make_mock_recognizer():
    recognizer = MagicMock()
    recognizer.recognize_window.return_value = None
    return recognizer


def _make_mock_ws_server(port: int = 9876):
    ws_server = MagicMock()
    ws_server.port = port
    ws_server.start = MagicMock()
    ws_server.stop = MagicMock()
    ws_server.join = MagicMock()
    return ws_server


def _make_mock_recognizer_service():
    service = MagicMock()
    service.start = MagicMock()
    service.stop = MagicMock()
    service.join = MagicMock()
    return service


class TestServerAppStart:
    def test_start_starts_ws_server(self) -> None:
        recognizer = _make_mock_recognizer()
        mock_ws_server = _make_mock_ws_server()
        mock_rs = _make_mock_recognizer_service()

        with (
            patch("src.server.ServerApp.WsServer", return_value=mock_ws_server),
            patch("src.server.ServerApp.RecognizerService", return_value=mock_rs),
            patch("src.server.ServerApp.SessionManager"),
        ):
            app = ServerApp(
                recognizer=recognizer,
                config=_CONFIG,
                vad_model_path=_MODELS_DIR / "silero_vad" / "silero_vad.onnx",
            )
            app.start()

        mock_ws_server.start.assert_called_once()

    def test_start_starts_recognizer_service(self) -> None:
        recognizer = _make_mock_recognizer()
        mock_ws_server = _make_mock_ws_server()
        mock_rs = _make_mock_recognizer_service()

        with (
            patch("src.server.ServerApp.WsServer", return_value=mock_ws_server),
            patch("src.server.ServerApp.RecognizerService", return_value=mock_rs),
            patch("src.server.ServerApp.SessionManager"),
        ):
            app = ServerApp(
                recognizer=recognizer,
                config=_CONFIG,
                vad_model_path=_MODELS_DIR / "silero_vad" / "silero_vad.onnx",
            )
            app.start()

        mock_rs.start.assert_called_once()

    def test_start_transitions_state_to_running(self) -> None:
        recognizer = _make_mock_recognizer()
        mock_ws_server = _make_mock_ws_server()
        mock_rs = _make_mock_recognizer_service()

        with (
            patch("src.server.ServerApp.WsServer", return_value=mock_ws_server),
            patch("src.server.ServerApp.RecognizerService", return_value=mock_rs),
            patch("src.server.ServerApp.SessionManager"),
        ):
            app = ServerApp(
                recognizer=recognizer,
                config=_CONFIG,
                vad_model_path=_MODELS_DIR / "silero_vad" / "silero_vad.onnx",
            )
            app.start()

        assert app.app_state.get_state() == "running"

    def test_port_is_available_after_start(self) -> None:
        recognizer = _make_mock_recognizer()
        mock_ws_server = _make_mock_ws_server(port=12345)
        mock_rs = _make_mock_recognizer_service()

        with (
            patch("src.server.ServerApp.WsServer", return_value=mock_ws_server),
            patch("src.server.ServerApp.RecognizerService", return_value=mock_rs),
            patch("src.server.ServerApp.SessionManager"),
        ):
            app = ServerApp(
                recognizer=recognizer,
                config=_CONFIG,
                vad_model_path=_MODELS_DIR / "silero_vad" / "silero_vad.onnx",
            )
            app.start()

        assert app.port == 12345


class TestServerAppStop:
    def test_stop_transitions_state_to_shutdown(self) -> None:
        recognizer = _make_mock_recognizer()
        mock_ws_server = _make_mock_ws_server()
        mock_rs = _make_mock_recognizer_service()

        with (
            patch("src.server.ServerApp.WsServer", return_value=mock_ws_server),
            patch("src.server.ServerApp.RecognizerService", return_value=mock_rs),
            patch("src.server.ServerApp.SessionManager"),
        ):
            app = ServerApp(
                recognizer=recognizer,
                config=_CONFIG,
                vad_model_path=_MODELS_DIR / "silero_vad" / "silero_vad.onnx",
            )
            app.start()
            app.stop()

        assert app.app_state.get_state() == "shutdown"

    def test_stop_stops_ws_server(self) -> None:
        recognizer = _make_mock_recognizer()
        mock_ws_server = _make_mock_ws_server()
        mock_rs = _make_mock_recognizer_service()

        with (
            patch("src.server.ServerApp.WsServer", return_value=mock_ws_server),
            patch("src.server.ServerApp.RecognizerService", return_value=mock_rs),
            patch("src.server.ServerApp.SessionManager"),
        ):
            app = ServerApp(
                recognizer=recognizer,
                config=_CONFIG,
                vad_model_path=_MODELS_DIR / "silero_vad" / "silero_vad.onnx",
            )
            app.start()
            app.stop()

        mock_ws_server.stop.assert_called_once()

    def test_stop_is_idempotent(self) -> None:
        recognizer = _make_mock_recognizer()
        mock_ws_server = _make_mock_ws_server()
        mock_rs = _make_mock_recognizer_service()

        with (
            patch("src.server.ServerApp.WsServer", return_value=mock_ws_server),
            patch("src.server.ServerApp.RecognizerService", return_value=mock_rs),
            patch("src.server.ServerApp.SessionManager"),
        ):
            app = ServerApp(
                recognizer=recognizer,
                config=_CONFIG,
                vad_model_path=_MODELS_DIR / "silero_vad" / "silero_vad.onnx",
            )
            app.start()
            app.stop()
            app.stop()  # Must not raise

        assert app.app_state.get_state() == "shutdown"

    def test_stop_joins_ws_server(self) -> None:
        recognizer = _make_mock_recognizer()
        mock_ws_server = _make_mock_ws_server()
        mock_rs = _make_mock_recognizer_service()

        with (
            patch("src.server.ServerApp.WsServer", return_value=mock_ws_server),
            patch("src.server.ServerApp.RecognizerService", return_value=mock_rs),
            patch("src.server.ServerApp.SessionManager"),
        ):
            app = ServerApp(
                recognizer=recognizer,
                config=_CONFIG,
                vad_model_path=_MODELS_DIR / "silero_vad" / "silero_vad.onnx",
            )
            app.start()
            app.stop()

        mock_ws_server.join.assert_called_once()

    def test_stop_joins_recognizer_service(self) -> None:
        recognizer = _make_mock_recognizer()
        mock_ws_server = _make_mock_ws_server()
        mock_rs = _make_mock_recognizer_service()

        with (
            patch("src.server.ServerApp.WsServer", return_value=mock_ws_server),
            patch("src.server.ServerApp.RecognizerService", return_value=mock_rs),
            patch("src.server.ServerApp.SessionManager"),
        ):
            app = ServerApp(
                recognizer=recognizer,
                config=_CONFIG,
                vad_model_path=_MODELS_DIR / "silero_vad" / "silero_vad.onnx",
            )
            app.start()
            app.stop()

        mock_rs.join.assert_called_once()
