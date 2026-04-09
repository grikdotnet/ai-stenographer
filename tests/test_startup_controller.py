"""Tests for StartupController startup orchestration phases."""

import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, call

import pytest

from src.StartupArgs import StartupArgs
from src.StartupController import StartupController
from src.StartupController import TauriBinaryNotFoundError, MissingModelsError, DownloadCancelledError
from src.asr.ModelLoader import ModelLoadError
from src.asr.ModelManager import ModelManager


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


def _make_args(
    *,
    server_only: bool = False,
    download_model: bool = False,
    verbose: bool = False,
    input_file: str | None = None,
    port: int = 0,
) -> StartupArgs:
    return StartupArgs(
        verbose=verbose,
        server_only=server_only,
        download_model=download_model,
        input_file=input_file,
        port=port,
    )


def _make_server_app_mock(port: int = 9000) -> MagicMock:
    mock = MagicMock()
    mock.port = port
    mock._ws_server = MagicMock()
    mock._ws_server.join = MagicMock()
    mock._ws_server._thread = MagicMock()
    mock._ws_server._thread.is_alive.return_value = False
    return mock


def _make_path_resolver_mock(
    models_dir: Path = Path("/fake/models"),
    logs_dir: Path = Path("/fake/logs"),
    config_dir: Path = Path("/fake/config"),
    root_dir: Path = Path("/fake"),
) -> MagicMock:
    resolver = MagicMock()
    resolver.paths.models_dir = models_dir
    resolver.paths.logs_dir = logs_dir
    resolver.paths.config_dir = config_dir
    resolver.paths.root_dir = root_dir
    resolver.get_config_path.return_value = config_dir / "server_config.json"
    return resolver


def _patch_stack(stack: ExitStack, server_app_mock: MagicMock) -> None:
    """Apply all standard patches for a full-startup controller run."""
    stack.enter_context(patch("src.StartupController.setup_logging"))
    stack.enter_context(patch("builtins.open", mock_open(read_data=_FAKE_CONFIG)))
    stack.enter_context(patch("onnxruntime.SessionOptions", return_value=MagicMock()))
    stack.enter_context(patch("onnxruntime.GraphOptimizationLevel"))
    stack.enter_context(patch("onnx_asr.load_model", return_value=MagicMock()))
    stack.enter_context(patch(
        "src.StartupController.ExecutionProviderManager",
        return_value=MagicMock(),
    ))
    stack.enter_context(patch(
        "src.StartupController.SessionOptionsFactory",
        return_value=MagicMock(),
    ))
    stack.enter_context(patch(
        "src.StartupController.Recognizer", return_value=MagicMock(),
    ))
    stack.enter_context(patch(
        "src.StartupController.ApplicationState", return_value=MagicMock(),
    ))
    stack.enter_context(patch(
        "src.StartupController.ServerApp", return_value=server_app_mock,
    ))


def _make_model_manager(
    exists: bool = True,
    recheck: bool | None = None,
) -> MagicMock:
    mm = MagicMock(spec=ModelManager)
    calls = [0]

    def _side_effect() -> bool:
        result = exists if calls[0] == 0 else (recheck if recheck is not None else True)
        calls[0] += 1
        return result

    mm.model_exists.side_effect = _side_effect
    return mm


def _make_controller(
    args: StartupArgs | None = None,
    path_resolver: MagicMock | None = None,
    model_manager: MagicMock | None = None,
) -> StartupController:
    return StartupController(
        args=args or _make_args(server_only=True),
        path_resolver=path_resolver or _make_path_resolver_mock(),
        model_manager=model_manager if model_manager is not None else _make_model_manager(exists=True),
    )


class TestEnvironmentValidation:
    """Tauri binary check in default (non-server-only) mode."""

    def test_missing_tauri_binary_exits_1_in_default_mode(self, capsys) -> None:
        resolver = _make_path_resolver_mock()
        controller = StartupController(
            args=_make_args(server_only=False),
            path_resolver=resolver,
            model_manager=_make_model_manager(exists=True),
        )
        with (
            patch("src.StartupController.setup_logging"),
            patch("pathlib.Path.exists", return_value=False),
            pytest.raises(TauriBinaryNotFoundError),
        ):
            controller.run()
        assert "stt-tauri-client.exe" in capsys.readouterr().err

    def test_missing_tauri_binary_skipped_in_server_only_mode(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = StartupController(
            args=_make_args(server_only=True),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=True),
        )
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            controller.run()
        # no SystemExit — we got here

    def test_present_tauri_binary_does_not_exit(self) -> None:
        server_app_mock = _make_server_app_mock()
        proc_mock = MagicMock()
        proc_mock.wait.return_value = 0
        controller = StartupController(
            args=_make_args(server_only=False),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=True),
        )
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            stack.enter_context(patch("pathlib.Path.exists", return_value=True))
            stack.enter_context(patch("subprocess.Popen", return_value=proc_mock))
            controller.run()
        # no SystemExit


class TestLoggingSetup:
    def test_setup_logging_called_with_verbose_true(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = StartupController(
            args=_make_args(server_only=True, verbose=True),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=True),
        )
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            # Enter after _patch_stack so this mock wins (last in = on top)
            mock_setup = stack.enter_context(patch("src.StartupController.setup_logging"))
            controller.run()
        mock_setup.assert_called_once()
        _, kwargs = mock_setup.call_args
        assert kwargs["verbose"] is True

    def test_setup_logging_receives_logs_dir(self) -> None:
        logs_dir = Path("/custom/logs")
        server_app_mock = _make_server_app_mock()
        controller = StartupController(
            args=_make_args(server_only=True),
            path_resolver=_make_path_resolver_mock(logs_dir=logs_dir),
            model_manager=_make_model_manager(exists=True),
        )
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            mock_setup = stack.enter_context(patch("src.StartupController.setup_logging"))
            controller.run()
        args_positional, _ = mock_setup.call_args
        assert args_positional[0] == logs_dir


class TestEnsureModelsServerOnly:
    """--server-only exits with code 1 when models are missing."""

    def test_missing_models_exits_1_in_server_only(self) -> None:
        controller = _make_controller(args=_make_args(server_only=True), model_manager=_make_model_manager(exists=False))
        with (
            patch("src.StartupController.setup_logging"),
            patch("src.StartupController.sys.stdin.isatty", return_value=False),
            patch("src.StartupController.sys.stdout.isatty", return_value=False),
            pytest.raises(MissingModelsError),
        ):
            controller.run()

    def test_missing_model_names_printed_to_stderr(self, capsys) -> None:
        controller = _make_controller(args=_make_args(server_only=True), model_manager=_make_model_manager(exists=False))
        with (
            patch("src.StartupController.setup_logging"),
            patch("src.StartupController.sys.stdin.isatty", return_value=False),
            patch("src.StartupController.sys.stdout.isatty", return_value=False),
            pytest.raises(MissingModelsError),
        ):
            controller.run()
        assert "parakeet" in capsys.readouterr().err

    def test_no_server_app_created_when_models_missing(self) -> None:
        controller = _make_controller(args=_make_args(server_only=True), model_manager=_make_model_manager(exists=False))
        with (
            patch("src.StartupController.setup_logging"),
            patch("src.StartupController.sys.stdin.isatty", return_value=False),
            patch("src.StartupController.sys.stdout.isatty", return_value=False),
            patch("src.StartupController.ServerApp") as mock_server_cls,
            pytest.raises(MissingModelsError),
        ):
            controller.run()
        mock_server_cls.assert_not_called()

    def test_interactive_server_only_prompts_and_downloads(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = StartupController(
            args=_make_args(server_only=True),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=False, recheck=True),
        )
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            stack.enter_context(patch("src.StartupController.sys.stdin.isatty", return_value=True))
            stack.enter_context(patch("src.StartupController.sys.stdout.isatty", return_value=True))
            dialog_cls = stack.enter_context(patch("src.StartupController.ModelDownloadCliDialog"))
            downloader = stack.enter_context(patch.object(controller, "_model_downloader"))
            dialog_cls.return_value.confirm_download.return_value = True
            controller.run()
        downloader.download_parakeet.assert_called_once()

    def test_interactive_server_only_decline_raises_cancelled(self) -> None:
        controller = _make_controller(args=_make_args(server_only=True), model_manager=_make_model_manager(exists=False))
        with (
            patch("src.StartupController.setup_logging"),
            patch("src.StartupController.sys.stdin.isatty", return_value=True),
            patch("src.StartupController.sys.stdout.isatty", return_value=True),
            patch("src.StartupController.ModelDownloadCliDialog") as dialog_cls,
            pytest.raises(DownloadCancelledError),
        ):
            dialog_cls.return_value.confirm_download.return_value = False
            controller.run()

    def test_non_interactive_download_model_flag_downloads_without_prompt(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = StartupController(
            args=_make_args(server_only=True, download_model=True),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=False, recheck=True),
        )
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            stack.enter_context(patch("src.StartupController.sys.stdin.isatty", return_value=False))
            stack.enter_context(patch("src.StartupController.sys.stdout.isatty", return_value=False))
            dialog_cls = stack.enter_context(patch("src.StartupController.ModelDownloadCliDialog"))
            downloader = stack.enter_context(patch.object(controller, "_model_downloader"))
            controller.run()
        dialog_cls.assert_not_called()
        downloader.download_parakeet.assert_called_once()

    def test_non_interactive_without_flag_prints_guidance(self, capsys) -> None:
        controller = _make_controller(args=_make_args(server_only=True), model_manager=_make_model_manager(exists=False))
        with (
            patch("src.StartupController.setup_logging"),
            patch("src.StartupController.sys.stdin.isatty", return_value=False),
            patch("src.StartupController.sys.stdout.isatty", return_value=False),
            pytest.raises(MissingModelsError),
        ):
            controller.run()
        assert "--download-model" in capsys.readouterr().err

    def test_download_failure_raises_missing_models_error(self) -> None:
        controller = StartupController(
            args=_make_args(server_only=True, download_model=True),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=False, recheck=False),
        )
        with (
            patch("src.StartupController.setup_logging"),
            patch("src.StartupController.sys.stdin.isatty", return_value=False),
            patch("src.StartupController.sys.stdout.isatty", return_value=False),
            patch.object(
                controller._model_downloader,
                "download_parakeet",
                side_effect=RuntimeError("network error"),
            ),
            pytest.raises(MissingModelsError, match="Automatic download failed"),
        ):
            controller.run()

    def test_download_keyboard_interrupt_raises_cancelled(self, capsys) -> None:
        controller = StartupController(
            args=_make_args(server_only=True, download_model=True),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=False, recheck=False),
        )
        with (
            patch("src.StartupController.setup_logging"),
            patch("src.StartupController.sys.stdin.isatty", return_value=False),
            patch("src.StartupController.sys.stdout.isatty", return_value=False),
            patch.object(
                controller._model_downloader,
                "download_parakeet",
                side_effect=KeyboardInterrupt,
            ),
            patch.object(controller._model_downloader, "cleanup_partial_files") as mock_cleanup,
            pytest.raises(DownloadCancelledError, match="cancelled"),
        ):
            controller.run()
        mock_cleanup.assert_called_once_with(controller._paths.models_dir)
        assert capsys.readouterr().err == "\n"


class TestEnsureModelsDefaultMode:
    """Default mode model download flow."""

    def test_spawns_download_subprocess_when_models_missing(self) -> None:
        download_proc = MagicMock()
        download_proc.returncode = 1

        controller = StartupController(
            args=_make_args(server_only=False),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=False),
        )
        with (
            patch("src.StartupController.setup_logging"),
            patch("pathlib.Path.exists", return_value=True),
            patch.object(
                controller,
                "_download_missing_models_via_client",
                side_effect=DownloadCancelledError("cancelled"),
            ) as mock_download,
            pytest.raises(DownloadCancelledError),
        ):
            controller.run()
        mock_download.assert_called_once()

    def test_exits_0_when_download_cancelled(self) -> None:
        controller = StartupController(
            args=_make_args(server_only=False),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=False),
        )
        with (
            patch("src.StartupController.setup_logging"),
            patch("pathlib.Path.exists", return_value=True),
            patch.object(
                controller,
                "_download_missing_models_via_client",
                side_effect=DownloadCancelledError("cancelled"),
            ),
            pytest.raises(DownloadCancelledError),
        ):
            controller.run()


class TestCliProgressReporter:
    def test_reports_progress_in_megabytes(self) -> None:
        controller = _make_controller()
        reporter = controller._make_cli_progress_reporter()

        with patch("builtins.print") as mock_print:
            reporter(0.5, 5 * 1024 * 1024, 10 * 1024 * 1024)

        mock_print.assert_called_once_with(
            "Downloading parakeet... 50% (5.0 MB downloaded)",
            flush=True,
        )

    def test_exits_1_when_still_missing_after_download(self) -> None:
        controller = StartupController(
            args=_make_args(server_only=False),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=False, recheck=False),
        )
        with (
            patch("src.StartupController.setup_logging"),
            patch("pathlib.Path.exists", return_value=True),
            patch.object(
                controller,
                "_download_missing_models_via_client",
                return_value=None,
            ),
            pytest.raises(MissingModelsError),
        ):
            controller.run()

    def test_continues_startup_after_successful_download(self) -> None:
        server_app_mock = _make_server_app_mock(port=9005)
        client_proc = MagicMock()
        client_proc.wait.return_value = 0

        controller = StartupController(
            args=_make_args(server_only=False),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=False, recheck=True),
        )
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            stack.enter_context(patch("pathlib.Path.exists", return_value=True))
            stack.enter_context(patch.object(
                controller, "_download_missing_models_via_client", return_value=None,
            ))
            stack.enter_context(patch("subprocess.Popen", return_value=client_proc))
            controller.run()
        server_app_mock.start.assert_called_once()

    def test_download_missing_models_via_client_raises_when_cancelled(self) -> None:
        download_proc = MagicMock()
        download_proc.returncode = 1
        controller = StartupController(
            args=_make_args(server_only=False),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=False),
        )

        with (
            patch.object(
                controller,
                "_spawn_client_for_download",
                return_value=download_proc,
            ) as mock_spawn,
            pytest.raises(DownloadCancelledError),
        ):
            controller._download_missing_models_via_client()

        mock_spawn.assert_called_once()

    def test_download_missing_models_via_client_waits_for_process(self) -> None:
        download_proc = MagicMock()
        download_proc.returncode = 0
        controller = StartupController(
            args=_make_args(server_only=False),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=False),
        )

        with patch.object(
            controller,
            "_spawn_client_for_download",
            return_value=download_proc,
        ):
            controller._download_missing_models_via_client()

        download_proc.wait.assert_called_once()


class TestCreateAndStartServer:
    def test_server_app_start_called(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            controller.run()
        server_app_mock.start.assert_called_once()

    def test_explicit_port_passed_to_server_app(self) -> None:
        server_app_mock = _make_server_app_mock(port=62062)
        controller = _make_controller(args=_make_args(server_only=True, port=62062))
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            server_app_cls = stack.enter_context(
                patch("src.StartupController.ServerApp", return_value=server_app_mock)
            )
            controller.run()
        _, kwargs = server_app_cls.call_args
        assert kwargs["port"] == 62062

    def test_default_port_zero_passed_to_server_app(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller(args=_make_args(server_only=True, port=0))
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            server_app_cls = stack.enter_context(
                patch("src.StartupController.ServerApp", return_value=server_app_mock)
            )
            controller.run()
        _, kwargs = server_app_cls.call_args
        assert kwargs["port"] == 0

    def test_model_load_error_raises(self) -> None:
        controller = _make_controller()
        with (
            patch("src.StartupController.setup_logging"),
            patch("builtins.open", mock_open(read_data=_FAKE_CONFIG)),
            patch("onnxruntime.SessionOptions", return_value=MagicMock()),
            patch("onnxruntime.GraphOptimizationLevel"),
            patch("src.StartupController.ExecutionProviderManager", return_value=MagicMock()),
            patch("src.StartupController.SessionOptionsFactory", return_value=MagicMock()),
            patch("src.StartupController.load_asr_model",
                  side_effect=ModelLoadError("model not found")),
            pytest.raises(ModelLoadError),
        ):
            controller.run()

    def test_vad_model_path_derived_from_models_dir(self) -> None:
        models_dir = Path("/custom/models")
        server_app_mock = _make_server_app_mock()
        controller = _make_controller(
            path_resolver=_make_path_resolver_mock(models_dir=models_dir),
        )
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            server_app_cls = stack.enter_context(
                patch("src.StartupController.ServerApp", return_value=server_app_mock)
            )
            controller.run()
        _, kwargs = server_app_cls.call_args
        expected = models_dir / "silero_vad" / "silero_vad.onnx"
        assert kwargs["vad_model_path"] == expected


class TestRunLifecycleServerOnly:
    def test_server_only_prints_url(self, capsys) -> None:
        server_app_mock = _make_server_app_mock(port=9010)
        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            controller.run()
        assert "ws://127.0.0.1:9010" in capsys.readouterr().out

    def test_server_only_calls_print_qr_code(self) -> None:
        server_app_mock = _make_server_app_mock(port=9011)
        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            mock_qr = stack.enter_context(patch("src.StartupController.print_qr_code"))
            controller.run()
        mock_qr.assert_called_once_with("ws://127.0.0.1:9011")

    def test_server_only_joins_ws_server_thread(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            controller.run()
        server_app_mock._ws_server._thread.is_alive.assert_called()

    def test_server_only_calls_stop_in_finally(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            controller.run()
        server_app_mock.stop.assert_called_once()

    def test_server_only_no_subprocess_spawned(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            mock_popen = stack.enter_context(patch("subprocess.Popen"))
            controller.run()
        mock_popen.assert_not_called()


class TestRunLifecycleDefaultMode:
    def _run_default(
        self,
        server_app_mock: MagicMock,
        proc_mock: MagicMock,
        args: StartupArgs | None = None,
    ) -> MagicMock:
        controller = StartupController(
            args=args or _make_args(server_only=False),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=True),
        )
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            stack.enter_context(patch("pathlib.Path.exists", return_value=True))
            mock_popen = stack.enter_context(patch("subprocess.Popen", return_value=proc_mock))
            controller.run()
        return mock_popen

    def test_default_mode_spawns_tauri_client(self) -> None:
        server_app_mock = _make_server_app_mock(port=9020)
        proc_mock = MagicMock()
        proc_mock.wait.return_value = 0
        mock_popen = self._run_default(server_app_mock, proc_mock)
        mock_popen.assert_called_once()

    def test_default_mode_subprocess_args_contain_server_url(self) -> None:
        server_app_mock = _make_server_app_mock(port=9021)
        proc_mock = MagicMock()
        proc_mock.wait.return_value = 0
        mock_popen = self._run_default(server_app_mock, proc_mock)
        popen_args = mock_popen.call_args[0][0]
        assert any("--server-url=ws://127.0.0.1:9021" in str(a) for a in popen_args)

    def test_default_mode_subprocess_args_contain_tauri_binary(self) -> None:
        server_app_mock = _make_server_app_mock(port=9022)
        proc_mock = MagicMock()
        proc_mock.wait.return_value = 0
        mock_popen = self._run_default(server_app_mock, proc_mock)
        popen_args = mock_popen.call_args[0][0]
        assert any("stt-tauri-client.exe" in str(a) for a in popen_args)

    def test_default_mode_forwards_input_file(self) -> None:
        server_app_mock = _make_server_app_mock(port=9023)
        proc_mock = MagicMock()
        proc_mock.wait.return_value = 0
        mock_popen = self._run_default(
            server_app_mock, proc_mock,
            args=_make_args(server_only=False, input_file="/some/file.wav"),
        )
        popen_args = mock_popen.call_args[0][0]
        assert any("--input-file=/some/file.wav" in str(a) for a in popen_args)

    def test_default_mode_no_input_file_omits_flag(self) -> None:
        server_app_mock = _make_server_app_mock(port=9024)
        proc_mock = MagicMock()
        proc_mock.wait.return_value = 0
        mock_popen = self._run_default(
            server_app_mock, proc_mock,
            args=_make_args(server_only=False, input_file=None),
        )
        popen_args = mock_popen.call_args[0][0]
        assert not any("--input-file" in str(a) for a in popen_args)

    def test_stop_called_after_ws_server_exits(self) -> None:
        server_app_mock = _make_server_app_mock(port=9025)
        proc_mock = MagicMock()
        call_order: list[str] = []
        server_app_mock._ws_server._thread.is_alive.return_value = False
        server_app_mock.stop.side_effect = lambda: call_order.append("stop")
        self._run_default(server_app_mock, proc_mock)
        assert call_order == ["stop"]

    def test_keyboard_interrupt_still_calls_stop(self) -> None:
        server_app_mock = _make_server_app_mock(port=9026)
        proc_mock = MagicMock()
        server_app_mock._ws_server.join.side_effect = KeyboardInterrupt
        server_app_mock._ws_server._thread.is_alive.return_value = True

        controller = StartupController(
            args=_make_args(server_only=False),
            path_resolver=_make_path_resolver_mock(),
            model_manager=_make_model_manager(exists=True),
        )
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            stack.enter_context(patch("pathlib.Path.exists", return_value=True))
            stack.enter_context(patch("subprocess.Popen", return_value=proc_mock))
            controller.run()
        server_app_mock.stop.assert_called_once()
