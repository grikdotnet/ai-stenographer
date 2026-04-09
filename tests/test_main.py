"""Tests for main.py _main() thin entry point.

Behavioral coverage has moved to test_startup_args.py and test_startup_controller.py.
These tests verify that _main() correctly wires StartupArgs and StartupController.
"""

import pytest
from unittest.mock import patch, MagicMock

from main import _main
from src.StartupController import TauriBinaryNotFoundError, MissingModelsError, DownloadCancelledError
from src.asr.ModelLoader import ModelLoadError


class TestMainWiring:
    """_main() delegates to StartupArgs.from_argv and StartupController.run."""

    def test_creates_startup_args_from_argv(self) -> None:
        argv = ["main.py", "--server-only"]
        with (
            patch("main.StartupArgs.from_argv", return_value=MagicMock()) as mock_from_argv,
            patch("main.ModelManager"),
            patch("main.StartupController") as mock_controller_cls,
        ):
            mock_controller_cls.return_value.run.return_value = None
            _main(argv)
        mock_from_argv.assert_called_once_with(argv)

    def test_creates_controller_with_args_and_path_resolver(self) -> None:
        fake_args = MagicMock()
        with (
            patch("main.StartupArgs.from_argv", return_value=fake_args),
            patch("main.ModelManager"),
            patch("main.StartupController") as mock_controller_cls,
            patch("main.path_resolver") as mock_resolver,
        ):
            mock_controller_cls.return_value.run.return_value = None
            _main(["main.py"])
        call_args = mock_controller_cls.call_args
        assert call_args[0][0] is fake_args
        assert call_args[0][1] is mock_resolver

    def test_calls_run_on_controller(self) -> None:
        with (
            patch("main.StartupArgs.from_argv", return_value=MagicMock()),
            patch("main.ModelManager"),
            patch("main.StartupController") as mock_controller_cls,
        ):
            mock_run = mock_controller_cls.return_value.run
            _main(["main.py"])
        mock_run.assert_called_once()

    def test_startup_args_system_exit_propagates(self) -> None:
        with (
            patch("main.StartupArgs.from_argv", side_effect=SystemExit(1)),
            pytest.raises(SystemExit) as exc_info,
        ):
            _main(["main.py"])
        assert exc_info.value.code == 1


class TestMainExceptionMapping:
    """_main() maps StartupController exceptions to sys.exit() codes."""

    def _run_with_exception(self, exc: Exception) -> "pytest.ExceptionInfo[SystemExit]":
        with (
            patch("main.StartupArgs.from_argv", return_value=MagicMock()),
            patch("main.ModelManager"),
            patch("main.StartupController") as mock_cls,
            pytest.raises(SystemExit) as exc_info,
        ):
            mock_cls.return_value.run.side_effect = exc
            _main(["main.py"])
        return exc_info

    def test_missing_models_error_exits_1(self) -> None:
        exc_info = self._run_with_exception(MissingModelsError("missing"))
        assert exc_info.value.code == 1

    def test_download_cancelled_exits_0(self) -> None:
        exc_info = self._run_with_exception(DownloadCancelledError("cancelled"))
        assert exc_info.value.code == 0

    def test_tauri_binary_not_found_exits_1(self) -> None:
        exc_info = self._run_with_exception(TauriBinaryNotFoundError("no binary"))
        assert exc_info.value.code == 1

    def test_model_load_error_exits_1(self) -> None:
        exc_info = self._run_with_exception(ModelLoadError("oops"))
        assert exc_info.value.code == 1

    def test_model_manager_instance_passed_to_controller(self) -> None:
        fake_mm = MagicMock()
        with (
            patch("main.StartupArgs.from_argv", return_value=MagicMock()),
            patch("main.ModelManager", return_value=fake_mm),
            patch("main.StartupController") as mock_cls,
        ):
            mock_cls.return_value.run.return_value = None
            _main(["main.py"])
        assert mock_cls.call_args[0][2] is fake_mm
