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
from src.PathResolver import ResolvedPaths
from src.asr.ModelRegistry import ModelRegistry
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
_PATHS = ResolvedPaths(
    app_dir=Path("/fake"),
    internal_dir=Path("/fake"),
    root_dir=Path("/fake"),
    models_dir=_MODELS_DIR,
    config_dir=Path("/fake/config"),
    assets_dir=Path("/fake"),
    logs_dir=Path("/fake/logs"),
    environment="development",
)
_REGISTRY = ModelRegistry(_PATHS)


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
        patch("src.server.ServerApp.WsServer", return_value=mock_ws_server) as mock_ws_server_cls,
        patch("src.server.ServerApp.RecognizerService", return_value=mock_rs),
        patch("src.server.ServerApp.SessionManager") as mock_session_manager_cls,
        patch("src.server.ServerApp.DownloadProgressNotifier") as mock_download_notifier_cls,
    ):
        app = ServerApp(
            config=_CONFIG,
            model_registry=_REGISTRY,
            app_state=ApplicationState(),
        )
        yield (
            app,
            mock_ws_server,
            mock_rs,
            mock_ws_server_cls,
            mock_session_manager_cls,
            mock_download_notifier_cls,
        )


class TestServerAppStart:
    def test_start(self, server_app_mocks) -> None:
        app, mock_ws_server, mock_rs, _, _, _ = server_app_mocks
        mock_ws_server.port = 12345
        app.start()
        mock_ws_server.start.assert_called_once_with()
        mock_rs.start.assert_called_once_with()
        assert app.port == 12345, f"expected port 12345, got {app.port}"

    def test_start_does_not_transition_app_state_to_running(self, server_app_mocks) -> None:
        app, _, _, _, _, _ = server_app_mocks

        app.start()

        assert app.app_state.get_state() == "starting"


class TestServerAppStop:
    def test_stop_transitions_state_to_shutdown(self, server_app_mocks) -> None:
        app, _, _, _, _, _ = server_app_mocks
        app.start()
        app.stop()
        assert app.app_state.get_state() == "shutdown", "app state should be 'shutdown' after stop()"

    def test_stop_stops_ws_server(self, server_app_mocks) -> None:
        app, mock_ws_server, _, _, _, _ = server_app_mocks
        app.start()
        app.stop()
        mock_ws_server.stop.assert_called_once_with()

    def test_stop_is_idempotent(self, server_app_mocks) -> None:
        app, _, _, _, _, _ = server_app_mocks
        app.start()
        app.stop()
        app.stop()  # Must not raise
        assert app.app_state.get_state() == "shutdown", "app state should remain 'shutdown' after second stop()"

    def test_stop_joins_ws_server(self, server_app_mocks) -> None:
        app, mock_ws_server, _, _, _, _ = server_app_mocks
        app.start()
        app.stop()
        mock_ws_server.join.assert_called_once()

    def test_stop_joins_recognizer_service(self, server_app_mocks) -> None:
        app, _, mock_rs, _, _, _ = server_app_mocks
        app.start()
        app.stop()
        mock_rs.join.assert_called_once()


class TestAttachRecognizer:
    def test_attach_recognizer_delegates_to_recognizer_service(self, server_app_mocks) -> None:
        app, _, mock_rs, _, _, _ = server_app_mocks
        recognizer = MagicMock()
        app.attach_recognizer(recognizer)
        mock_rs.attach_recognizer.assert_called_once_with(recognizer)

class TestServerAppLifecycleDelegation:
    def test_join_delegates_to_ws_server(self, server_app_mocks) -> None:
        app, mock_ws_server, _, _, _, _ = server_app_mocks

        app.join(timeout=0.5)

        mock_ws_server.join.assert_called_once_with(timeout=0.5)

    def test_is_running_delegates_to_ws_server(self, server_app_mocks) -> None:
        app, mock_ws_server, _, _, _, _ = server_app_mocks
        mock_ws_server.is_running.return_value = True

        assert app.is_running() is True
        mock_ws_server.is_running.assert_called_once_with()


class TestServerAppWiring:
    def test_ws_server_is_constructed_with_app_state(self, server_app_mocks) -> None:
        _, _, _, mock_ws_server_cls, _, _ = server_app_mocks

        kwargs = mock_ws_server_cls.call_args.kwargs

        assert "app_state" in kwargs

    def test_download_notifier_is_constructed_with_session_manager_broadcaster(self, server_app_mocks) -> None:
        _, _, _, _, mock_session_manager_cls, mock_download_notifier_cls = server_app_mocks

        kwargs = mock_download_notifier_cls.call_args.kwargs

        assert kwargs["broadcaster"] is mock_session_manager_cls.return_value

    def test_command_handler_receives_explicit_models_dir(self, server_app_mocks) -> None:
        app, _, _, _, _, _ = server_app_mocks

        command_controller = app._command_controller
        assert command_controller._model_commands._registry._models_dir == _MODELS_DIR

    def test_command_handler_receives_layer_four_readiness_coordinator(self, server_app_mocks) -> None:
        app, _, _, _, _, _ = server_app_mocks

        model_commands = app._command_controller._model_commands
        assert model_commands._model_readiness is app._recognizer_readiness
        assert model_commands._download_events is app._download_notifier

    def test_session_manager_receives_vad_model_instance(self, server_app_mocks) -> None:
        _, _, _, _, mock_session_manager_cls, _ = server_app_mocks

        kwargs = mock_session_manager_cls.call_args.kwargs
        assert kwargs["vad_model"] is _REGISTRY.get_vad_model()
