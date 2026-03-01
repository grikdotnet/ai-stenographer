"""Tests for main.py _main() entry-point function (Phase 9).

Strategy: import and call _main() directly with patched ModelManager,
onnx_asr, Recognizer, and ServerApp so no real processes, sockets, or
models are involved.

Tests cover:
- --server-only with missing models: prints to stderr, exits with code 1, no GUI
- --server-only with models present: ServerApp started, no subprocess spawned
- Default mode with models present: ServerApp started, Popen called with --server-url,
  daemon watcher thread started
- Watcher function: subprocess exit triggers server_app.stop(), then terminate()
"""

import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import json
import pytest

from main import _main


_FAKE_MODELS_DIR = Path("/fake/models")
_FAKE_LOGS_DIR = Path("/fake/logs")
_FAKE_CONFIG_PATH = "/fake/config/stt_config.json"
_FAKE_CONFIG = json.dumps({
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
})


def _make_server_app_mock(port: int = 9000) -> MagicMock:
    mock = MagicMock()
    mock.port = port
    mock._ws_server = MagicMock()
    mock._ws_server.join = MagicMock()
    return mock


def _make_proc_mock(exit_code: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.wait.return_value = exit_code
    return proc


def _model_patches(stack: ExitStack, server_app_mock: MagicMock) -> None:
    """Enter patches for model loading and server construction into stack."""
    stack.enter_context(patch("main.setup_logging"))
    stack.enter_context(patch("builtins.open", mock_open(read_data=_FAKE_CONFIG)))
    stack.enter_context(patch("main.ModelManager.get_missing_models", return_value=[]))
    stack.enter_context(patch("onnxruntime.SessionOptions", return_value=MagicMock()))
    stack.enter_context(patch("onnxruntime.GraphOptimizationLevel"))
    stack.enter_context(patch("onnx_asr.load_model", return_value=MagicMock()))
    stack.enter_context(patch("src.asr.ExecutionProviderManager.ExecutionProviderManager"))
    stack.enter_context(patch("src.asr.SessionOptionsFactory.SessionOptionsFactory"))
    stack.enter_context(patch("src.asr.Recognizer.Recognizer", return_value=MagicMock()))
    stack.enter_context(patch("src.ServerApplicationState.ServerApplicationState", return_value=MagicMock()))
    stack.enter_context(patch("src.server.ServerApp.ServerApp", return_value=server_app_mock))


class TestServerOnlyMissingModels:
    """--server-only exits with code 1 when models are missing."""

    def test_exits_with_code_1_when_models_missing(self) -> None:
        with (
            patch("main.setup_logging"),
            patch("main.ModelManager.get_missing_models", return_value=["parakeet"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main(["main.py", "--server-only"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        assert exc_info.value.code == 1

    def test_prints_missing_model_names_to_stderr(self, capsys) -> None:
        with (
            patch("main.setup_logging"),
            patch("main.ModelManager.get_missing_models", return_value=["parakeet"]),
            pytest.raises(SystemExit),
        ):
            _main(["main.py", "--server-only"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        captured = capsys.readouterr()
        assert "parakeet" in captured.err

    def test_no_server_app_created_when_models_missing(self) -> None:
        with (
            patch("main.setup_logging"),
            patch("main.ModelManager.get_missing_models", return_value=["parakeet"]),
            patch("src.server.ServerApp.ServerApp") as mock_server_cls,
            pytest.raises(SystemExit),
        ):
            _main(["main.py", "--server-only"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        mock_server_cls.assert_not_called()

    def test_no_subprocess_spawned_when_models_missing(self) -> None:
        with (
            patch("main.setup_logging"),
            patch("main.ModelManager.get_missing_models", return_value=["parakeet"]),
            patch("subprocess.Popen") as mock_popen,
            pytest.raises(SystemExit),
        ):
            _main(["main.py", "--server-only"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        mock_popen.assert_not_called()


class TestServerOnlyModelsPresent:
    """--server-only starts ServerApp headless; no subprocess spawned."""

    def test_server_app_start_called(self) -> None:
        server_app_mock = _make_server_app_mock()

        with ExitStack() as stack:
            _model_patches(stack, server_app_mock)
            _main(["main.py", "--server-only"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        server_app_mock.start.assert_called_once()

    def test_no_subprocess_spawned(self) -> None:
        server_app_mock = _make_server_app_mock()
        mock_popen = MagicMock()

        with ExitStack() as stack:
            _model_patches(stack, server_app_mock)
            stack.enter_context(patch("subprocess.Popen", new=mock_popen))
            _main(["main.py", "--server-only"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        mock_popen.assert_not_called()

    def test_ws_server_join_called_in_server_only_mode(self) -> None:
        server_app_mock = _make_server_app_mock()

        with ExitStack() as stack:
            _model_patches(stack, server_app_mock)
            _main(["main.py", "--server-only"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        server_app_mock._ws_server.join.assert_called_once()


class TestDefaultModeModelsPresent:
    """Default mode: ServerApp started, client.py subprocess spawned, watcher thread started."""

    def test_server_app_started(self) -> None:
        server_app_mock = _make_server_app_mock(port=9001)
        proc_mock = _make_proc_mock()

        with ExitStack() as stack:
            _model_patches(stack, server_app_mock)
            stack.enter_context(patch("subprocess.Popen", return_value=proc_mock))
            stack.enter_context(patch("threading.Thread", return_value=MagicMock()))
            _main(["main.py"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        server_app_mock.start.assert_called_once()

    def test_subprocess_spawned_with_server_url_arg(self) -> None:
        server_app_mock = _make_server_app_mock(port=9001)
        proc_mock = _make_proc_mock()
        mock_popen = MagicMock(return_value=proc_mock)

        with ExitStack() as stack:
            _model_patches(stack, server_app_mock)
            stack.enter_context(patch("subprocess.Popen", new=mock_popen))
            stack.enter_context(patch("threading.Thread", return_value=MagicMock()))
            _main(["main.py"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        mock_popen.assert_called_once()
        popen_args = mock_popen.call_args[0][0]
        assert any("--server-url=" in arg for arg in popen_args)
        assert any("ws://127.0.0.1:9001" in arg for arg in popen_args)

    def test_subprocess_spawned_with_client_py(self) -> None:
        server_app_mock = _make_server_app_mock(port=9002)
        proc_mock = _make_proc_mock()
        mock_popen = MagicMock(return_value=proc_mock)

        with ExitStack() as stack:
            _model_patches(stack, server_app_mock)
            stack.enter_context(patch("subprocess.Popen", new=mock_popen))
            stack.enter_context(patch("threading.Thread", return_value=MagicMock()))
            _main(["main.py"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        popen_args = mock_popen.call_args[0][0]
        assert any("client.py" in str(arg) for arg in popen_args)

    def test_watcher_thread_is_daemon(self) -> None:
        server_app_mock = _make_server_app_mock(port=9003)
        proc_mock = _make_proc_mock()
        captured_threads: list[dict] = []

        def capture_thread(**kwargs):
            t = MagicMock()
            captured_threads.append(kwargs)
            return t

        with ExitStack() as stack:
            _model_patches(stack, server_app_mock)
            stack.enter_context(patch("subprocess.Popen", return_value=proc_mock))
            stack.enter_context(patch("threading.Thread", side_effect=capture_thread))
            _main(["main.py"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        assert any(kw.get("daemon") is True for kw in captured_threads)

    def test_ws_server_join_not_called_in_default_mode(self) -> None:
        server_app_mock = _make_server_app_mock(port=9004)
        proc_mock = _make_proc_mock()

        with ExitStack() as stack:
            _model_patches(stack, server_app_mock)
            stack.enter_context(patch("subprocess.Popen", return_value=proc_mock))
            stack.enter_context(patch("threading.Thread", return_value=MagicMock()))
            _main(["main.py"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        server_app_mock._ws_server.join.assert_not_called()


class TestWatcherFunction:
    """The watcher closure: subprocess exit triggers server_app.stop() and proc.terminate()."""

    def _capture_watcher_target(self, server_app_mock: MagicMock, proc_mock: MagicMock):
        """Run _main() in default mode and return the watcher thread target function."""
        captured: list = []

        def capture_thread(target=None, daemon=None, name=None, **kwargs):
            if name and "watcher" in name.lower():
                captured.append(target)
            return MagicMock()

        with ExitStack() as stack:
            _model_patches(stack, server_app_mock)
            stack.enter_context(patch("subprocess.Popen", return_value=proc_mock))
            stack.enter_context(patch("threading.Thread", side_effect=capture_thread))
            _main(["main.py"], _FAKE_MODELS_DIR, _FAKE_LOGS_DIR, _FAKE_CONFIG_PATH)

        assert len(captured) == 1, f"Expected 1 watcher thread, got {len(captured)}"
        return captured[0]

    def test_watcher_calls_server_app_stop_when_subprocess_exits(self) -> None:
        server_app_mock = _make_server_app_mock()
        proc_mock = _make_proc_mock(exit_code=0)

        watcher = self._capture_watcher_target(server_app_mock, proc_mock)
        server_app_mock.stop.reset_mock()

        watcher()

        server_app_mock.stop.assert_called_once()

    def test_watcher_terminates_subprocess_after_server_stop(self) -> None:
        server_app_mock = _make_server_app_mock()
        proc_mock = _make_proc_mock(exit_code=1)

        watcher = self._capture_watcher_target(server_app_mock, proc_mock)

        watcher()

        proc_mock.terminate.assert_called()
