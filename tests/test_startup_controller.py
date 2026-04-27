"""Tests for StartupController startup orchestration phases."""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from src.PathResolver import ResolvedPaths
from src.StartupArgs import StartupArgs
from src.StartupController import DownloadCancelledError, MissingModelsError, StartupController
from src.asr.ModelLoader import ModelLoadError
from src.asr.ModelManager import ModelManager
from src.asr.ModelRegistry import ModelRegistry


_FAKE_CONFIG = json.dumps({
    "server": {"host": "config-host"},
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
    download_model: bool = False,
    host: str | None = None,
    verbose: bool = False,
    port: int = 0,
) -> StartupArgs:
    return StartupArgs(
        verbose=verbose,
        download_model=download_model,
        host=host,
        port=port,
    )


def _make_server_app_mock(port: int = 9000) -> MagicMock:
    mock = MagicMock()
    mock.port = port
    mock.is_running.side_effect = [False]
    mock.join = MagicMock()
    return mock


def _make_paths(
    models_dir: Path = Path("/fake/models"),
    logs_dir: Path = Path("/fake/logs"),
    config_dir: Path = Path("/fake/config"),
    root_dir: Path = Path("/fake"),
) -> ResolvedPaths:
    return ResolvedPaths(
        app_dir=root_dir,
        internal_dir=root_dir,
        root_dir=root_dir,
        models_dir=models_dir,
        config_dir=config_dir,
        assets_dir=root_dir,
        logs_dir=logs_dir,
        environment="development",
    )


def _patch_stack(stack: ExitStack, server_app_mock: MagicMock) -> MagicMock:
    """Apply all standard patches for a full-startup controller run."""
    stack.enter_context(patch("src.StartupController.setup_logging"))
    stack.enter_context(patch("builtins.open", mock_open(read_data=_FAKE_CONFIG)))
    recognizer_factory = MagicMock()
    recognizer_factory.create_recognizer.return_value = MagicMock()
    stack.enter_context(
        patch(
            "src.StartupController.RecognizerFactory",
            return_value=recognizer_factory,
        )
    )
    stack.enter_context(patch(
        "src.StartupController.ApplicationState", return_value=MagicMock(),
    ))
    stack.enter_context(patch(
        "src.StartupController.ServerApp", return_value=server_app_mock,
    ))
    return recognizer_factory


def _make_model_manager(
    exists: bool = True,
    recheck: bool | None = None,
) -> MagicMock:
    mm = MagicMock(spec=ModelManager)
    registry = MagicMock(spec=ModelRegistry)
    asr_model = MagicMock()
    asr_model.name = "parakeet"
    asr_model.get_model_path.return_value = Path("/fake/models/parakeet")
    calls = [0]

    def _side_effect() -> bool:
        result = exists if calls[0] == 0 else (recheck if recheck is not None else True)
        calls[0] += 1
        return result

    mm.model_exists.side_effect = _side_effect
    mm.model_registry = registry
    mm.get_asr_model.return_value = asr_model
    return mm


def _make_controller(
    args: StartupArgs | None = None,
    paths: ResolvedPaths | None = None,
    model_manager: MagicMock | None = None,
) -> StartupController:
    return StartupController(
        args=args or _make_args(),
        paths=paths or _make_paths(),
        model_manager=model_manager if model_manager is not None else _make_model_manager(exists=True),
    )


class TestNoPythonGuiOwnership:
    def test_run_does_not_validate_or_spawn_tauri_client(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller()

        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            stack.enter_context(patch("pathlib.Path.exists", side_effect=AssertionError("unexpected binary check")))
            popen = stack.enter_context(patch("subprocess.Popen"))
            controller.run()

        popen.assert_not_called()

    def test_removed_tauri_helpers_are_not_exposed(self) -> None:
        controller = _make_controller()
        assert not hasattr(controller, "_validate_environment")
        assert not hasattr(controller, "_spawn_tauri_client")
        assert not hasattr(controller, "_start_client_exit_watcher")
        assert not hasattr(controller, "_tauri_binary_path")


class TestLoggingSetup:
    def test_setup_logging_called_with_verbose_true(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller(args=_make_args(verbose=True))
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            mock_setup = stack.enter_context(patch("src.StartupController.setup_logging"))
            controller.run()
        mock_setup.assert_called_once()
        _, kwargs = mock_setup.call_args
        assert kwargs["verbose"] is True

    def test_setup_logging_receives_logs_dir(self) -> None:
        logs_dir = Path("/custom/logs")
        server_app_mock = _make_server_app_mock()
        controller = _make_controller(paths=_make_paths(logs_dir=logs_dir))
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            mock_setup = stack.enter_context(patch("src.StartupController.setup_logging"))
            controller.run()
        args_positional, _ = mock_setup.call_args
        assert args_positional[0] == logs_dir


class TestMissingModelStartup:
    def test_missing_models_in_non_interactive_startup_starts_waiting_server(self) -> None:
        server_app_mock = _make_server_app_mock(port=9005)
        app_state_mock = MagicMock()
        with patch("src.StartupController.ApplicationState", return_value=app_state_mock):
            controller = _make_controller(model_manager=_make_model_manager(exists=False))

        with ExitStack() as stack:
            recognizer_factory = _patch_stack(stack, server_app_mock)
            stack.enter_context(patch("src.StartupController.sys.stdin.isatty", return_value=False))
            stack.enter_context(patch("src.StartupController.sys.stdout.isatty", return_value=False))
            controller.run()

        app_state_mock.set_state.assert_called_once_with("waiting_for_model")
        server_app_mock.start.assert_called_once()
        recognizer_factory.create_recognizer.assert_not_called()
        server_app_mock.attach_recognizer.assert_not_called()

    def test_missing_models_in_non_interactive_startup_does_not_prompt(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller(model_manager=_make_model_manager(exists=False))

        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            stack.enter_context(patch("src.StartupController.sys.stdin.isatty", return_value=False))
            stack.enter_context(patch("src.StartupController.sys.stdout.isatty", return_value=False))
            dialog_cls = stack.enter_context(patch(
                "src.StartupController.ModelDownloadCliDialog",
                side_effect=AssertionError("unexpected prompt"),
            ))
            controller.run()

        dialog_cls.assert_not_called()

    def test_missing_models_in_interactive_startup_prompts_before_download(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller(model_manager=_make_model_manager(exists=False, recheck=True))
        dialog = MagicMock()
        dialog.confirm_download.return_value = True

        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            stack.enter_context(patch("src.StartupController.sys.stdin.isatty", return_value=True))
            stack.enter_context(patch("src.StartupController.sys.stdout.isatty", return_value=True))
            dialog_cls = stack.enter_context(
                patch("src.StartupController.ModelDownloadCliDialog", return_value=dialog)
            )
            controller.run()

        dialog_cls.assert_called_once_with(["parakeet"])
        dialog.confirm_download.assert_called_once_with()
        controller._asr_model.download.assert_called_once()
        server_app_mock.attach_recognizer.assert_called_once()

    def test_missing_models_in_interactive_startup_decline_cancels_download(self) -> None:
        controller = _make_controller(model_manager=_make_model_manager(exists=False))
        dialog = MagicMock()
        dialog.confirm_download.return_value = False

        with (
            patch("src.StartupController.setup_logging"),
            patch("src.StartupController.sys.stdin.isatty", return_value=True),
            patch("src.StartupController.sys.stdout.isatty", return_value=True),
            patch("src.StartupController.ModelDownloadCliDialog", return_value=dialog) as dialog_cls,
            pytest.raises(DownloadCancelledError, match="cancelled"),
        ):
            controller.run()

        dialog_cls.assert_called_once_with(["parakeet"])
        controller._asr_model.download.assert_not_called()

    def test_download_model_flag_downloads_without_prompt(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller(
            args=_make_args(download_model=True),
            model_manager=_make_model_manager(exists=False, recheck=True),
        )

        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            asr_model = stack.enter_context(patch.object(controller, "_asr_model"))
            controller.run()

        asr_model.download.assert_called_once()

    def test_download_failure_raises_missing_models_error(self) -> None:
        controller = _make_controller(
            args=_make_args(download_model=True),
            model_manager=_make_model_manager(exists=False, recheck=False),
        )
        with (
            patch("src.StartupController.setup_logging"),
            patch.object(
                controller._asr_model,
                "download",
                side_effect=RuntimeError("network error"),
            ),
            pytest.raises(MissingModelsError, match="Automatic download failed"),
        ):
            controller.run()

    def test_download_keyboard_interrupt_raises_cancelled(self, capsys) -> None:
        controller = _make_controller(
            args=_make_args(download_model=True),
            model_manager=_make_model_manager(exists=False, recheck=False),
        )
        with (
            patch("src.StartupController.setup_logging"),
            patch.object(
                controller._asr_model,
                "download",
                side_effect=KeyboardInterrupt,
            ),
            patch.object(controller._asr_model, "cleanup_partial_files") as mock_cleanup,
            pytest.raises(DownloadCancelledError, match="cancelled"),
        ):
            controller.run()
        mock_cleanup.assert_called_once_with()
        assert capsys.readouterr().err == "\n"


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
        controller = _make_controller(args=_make_args(port=62062))
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
        controller = _make_controller(args=_make_args(port=0))
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            server_app_cls = stack.enter_context(
                patch("src.StartupController.ServerApp", return_value=server_app_mock)
            )
            controller.run()
        _, kwargs = server_app_cls.call_args
        assert kwargs["port"] == 0

    def test_cli_host_passed_to_server_app(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller(args=_make_args(host="cli-host"))
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            server_app_cls = stack.enter_context(
                patch("src.StartupController.ServerApp", return_value=server_app_mock)
            )
            controller.run()
        _, kwargs = server_app_cls.call_args
        assert kwargs["host"] == "cli-host"

    def test_config_host_used_when_cli_host_absent(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            server_app_cls = stack.enter_context(
                patch("src.StartupController.ServerApp", return_value=server_app_mock)
            )
            controller.run()
        _, kwargs = server_app_cls.call_args
        assert kwargs["host"] == "config-host"

    def test_default_host_used_when_cli_and_config_host_absent(self) -> None:
        server_app_mock = _make_server_app_mock()
        config_without_host = json.dumps({
            "audio": {"sample_rate": 16000},
            "vad": {"frame_duration_ms": 32, "threshold": 0.5},
            "windowing": {"max_speech_duration_ms": 3000, "max_window_duration": 7.0},
        })
        controller = _make_controller()
        with ExitStack() as stack:
            stack.enter_context(patch("src.StartupController.setup_logging"))
            stack.enter_context(patch("builtins.open", mock_open(read_data=config_without_host)))
            recognizer_factory = MagicMock()
            recognizer_factory.create_recognizer.return_value = MagicMock()
            stack.enter_context(
                patch(
                    "src.StartupController.RecognizerFactory",
                    return_value=recognizer_factory,
                )
            )
            stack.enter_context(patch(
                "src.StartupController.ApplicationState", return_value=MagicMock(),
            ))
            server_app_cls = stack.enter_context(
                patch("src.StartupController.ServerApp", return_value=server_app_mock)
            )
            controller.run()
        _, kwargs = server_app_cls.call_args
        assert kwargs["host"] == "127.0.0.1"

    def test_model_load_error_raises(self) -> None:
        controller = _make_controller()
        recognizer_factory = MagicMock()
        recognizer_factory.create_recognizer.side_effect = ModelLoadError("model not found")
        with (
            patch("src.StartupController.setup_logging"),
            patch("builtins.open", mock_open(read_data=_FAKE_CONFIG)),
            patch(
                "src.StartupController.RecognizerFactory",
                return_value=recognizer_factory,
            ),
            patch("src.StartupController.ServerApp", return_value=_make_server_app_mock()),
            pytest.raises(ModelLoadError),
        ):
            controller.run()

    def test_vad_model_path_derived_from_models_dir(self) -> None:
        model_registry = MagicMock(spec=ModelRegistry)
        server_app_mock = _make_server_app_mock()
        controller = _make_controller()
        controller._model_registry = model_registry
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            server_app_cls = stack.enter_context(
                patch("src.StartupController.ServerApp", return_value=server_app_mock)
            )
            controller.run()
        _, kwargs = server_app_cls.call_args
        assert kwargs["model_registry"] is model_registry

    def test_server_app_receives_model_registry(self) -> None:
        server_app_mock = _make_server_app_mock()
        model_manager = _make_model_manager()
        controller = _make_controller(model_manager=model_manager)
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            server_app_cls = stack.enter_context(
                patch("src.StartupController.ServerApp", return_value=server_app_mock)
            )
            controller.run()
        _, kwargs = server_app_cls.call_args
        assert kwargs["model_registry"] is model_manager.model_registry

    def test_recognizer_factory_object_passed_to_server_app(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            server_app_cls = stack.enter_context(
                patch("src.StartupController.ServerApp", return_value=server_app_mock)
            )
            controller.run()
        _, kwargs = server_app_cls.call_args
        recognizer_factory = kwargs["recognizer_factory"]
        assert recognizer_factory is controller._recognizer_factory
        assert recognizer_factory is not None

    def test_config_loaded_from_paths_config_dir(self) -> None:
        server_app_mock = _make_server_app_mock()
        paths = _make_paths(config_dir=Path("/custom/config"))
        controller = _make_controller(paths=paths)
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            mock_open_fn = stack.enter_context(patch("builtins.open", mock_open(read_data=_FAKE_CONFIG)))
            controller.run()
        mock_open_fn.assert_called_once_with(paths.config_dir / "server_config.json")

    def test_controller_owned_app_state_passed_to_server_app(self) -> None:
        server_app_mock = _make_server_app_mock()
        app_state_mock = MagicMock()
        with patch("src.StartupController.ApplicationState", return_value=app_state_mock) as app_state_cls:
            controller = _make_controller()

        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            server_app_cls = stack.enter_context(
                patch("src.StartupController.ServerApp", return_value=server_app_mock)
            )
            controller.run()

        app_state_cls.assert_called_once_with()
        _, kwargs = server_app_cls.call_args
        assert kwargs["app_state"] is app_state_mock

    def test_controller_owned_app_state_passed_to_recognizer_factory(self) -> None:
        server_app_mock = _make_server_app_mock()
        app_state_mock = MagicMock()
        with patch("src.StartupController.ApplicationState", return_value=app_state_mock):
            controller = _make_controller()

        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            recognizer_factory_cls = stack.enter_context(
                patch("src.StartupController.RecognizerFactory")
            )
            recognizer_factory = MagicMock()
            recognizer_factory.create_recognizer.return_value = MagicMock()
            recognizer_factory_cls.return_value = recognizer_factory
            controller.run()

        _, kwargs = recognizer_factory_cls.call_args
        assert kwargs["app_state"] is app_state_mock

    def test_running_state_set_after_attach_recognizer(self) -> None:
        server_app_mock = _make_server_app_mock()
        app_state_mock = MagicMock()
        recognizer_mock = MagicMock()
        call_order: list[str] = []
        server_app_mock.attach_recognizer.side_effect = lambda recognizer: call_order.append(
            f"attach:{id(recognizer)}"
        )
        app_state_mock.set_state.side_effect = lambda state: call_order.append(state)
        with patch("src.StartupController.ApplicationState", return_value=app_state_mock):
            controller = _make_controller()

        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            recognizer_factory_cls = stack.enter_context(
                patch("src.StartupController.RecognizerFactory")
            )
            recognizer_factory = MagicMock()
            recognizer_factory.create_recognizer.return_value = recognizer_mock
            recognizer_factory_cls.return_value = recognizer_factory
            controller.run()

        assert call_order[:2] == [
            f"attach:{id(recognizer_mock)}",
            "running",
        ]
        server_app_mock.attach_recognizer.assert_called_once_with(recognizer_mock)
        app_state_mock.set_state.assert_called_once_with("running")


class TestRunLifecycle:
    def test_prints_parseable_server_url(self, capsys) -> None:
        server_app_mock = _make_server_app_mock(port=9010)
        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            controller.run()
        assert "Server listening on ws://config-host:9010" in capsys.readouterr().out

    def test_calls_print_qr_code(self) -> None:
        server_app_mock = _make_server_app_mock(port=9011)
        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            mock_qr = stack.enter_context(patch("src.StartupController.print_qr_code"))
            controller.run()
        mock_qr.assert_called_once_with("ws://config-host:9011")

    def test_joins_server_app_while_running(self) -> None:
        server_app_mock = _make_server_app_mock()
        server_app_mock.is_running.side_effect = [True, False]
        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            controller.run()
        server_app_mock.join.assert_called_once_with(timeout=0.5)

    def test_calls_stop_in_finally(self) -> None:
        server_app_mock = _make_server_app_mock()
        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            controller.run()
        server_app_mock.stop.assert_called_once()

    def test_keyboard_interrupt_still_calls_stop(self) -> None:
        server_app_mock = _make_server_app_mock()
        server_app_mock.join.side_effect = KeyboardInterrupt
        server_app_mock.is_running.side_effect = [True]

        controller = _make_controller()
        with ExitStack() as stack:
            _patch_stack(stack, server_app_mock)
            controller.run()
        assert server_app_mock.stop.call_count >= 1


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
