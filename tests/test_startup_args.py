"""Tests for StartupArgs CLI argument parsing."""

from dataclasses import FrozenInstanceError

import pytest

from src.StartupArgs import StartupArgs


class TestDefaults:
    """No flags -> safe defaults."""

    def test_no_args_gives_defaults(self) -> None:
        args = StartupArgs.from_argv(["main.py"])
        assert args.verbose is False
        assert args.server_only is False
        assert args.download_model is False
        assert args.input_file is None
        assert args.port == 0

    def test_empty_argv_gives_defaults(self) -> None:
        args = StartupArgs.from_argv([])
        assert args.verbose is False
        assert args.server_only is False
        assert args.download_model is False
        assert args.input_file is None
        assert args.port == 0


class TestVerboseFlag:
    def test_verbose_short_flag(self) -> None:
        assert StartupArgs.from_argv(["main.py", "-v"]).verbose is True

    def test_verbose_not_set_without_flag(self) -> None:
        assert StartupArgs.from_argv(["main.py"]).verbose is False

    def test_verbose_flag_position_independence(self) -> None:
        assert StartupArgs.from_argv(["-v", "main.py"]).verbose is True
        assert StartupArgs.from_argv(["main.py", "--server-only", "-v"]).verbose is True


class TestServerOnlyFlag:
    def test_server_only_flag_sets_server_only(self) -> None:
        assert StartupArgs.from_argv(["main.py", "--server-only"]).server_only is True

    def test_server_only_not_set_by_default(self) -> None:
        assert StartupArgs.from_argv(["main.py"]).server_only is False

    def test_server_only_combined_with_verbose(self) -> None:
        args = StartupArgs.from_argv(["main.py", "-v", "--server-only"])
        assert args.verbose is True
        assert args.server_only is True


class TestDownloadModelFlag:
    def test_download_model_flag_sets_download_model(self) -> None:
        args = StartupArgs.from_argv(["main.py", "--server-only", "--download-model"])
        assert args.download_model is True

    def test_download_model_not_set_by_default(self) -> None:
        assert StartupArgs.from_argv(["main.py", "--server-only"]).download_model is False

    def test_download_model_requires_server_only(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            StartupArgs.from_argv(["main.py", "--download-model"])
        assert exc_info.value.code == 1
        assert "--server-only" in capsys.readouterr().err


class TestInputFileArg:
    def test_input_file_parsed_correctly(self) -> None:
        args = StartupArgs.from_argv(["main.py", "--input-file=/path/to/file.wav"])
        assert args.input_file == "/path/to/file.wav"

    def test_input_file_with_equals_in_path(self) -> None:
        args = StartupArgs.from_argv(["main.py", "--input-file=/path/a=b.wav"])
        assert args.input_file == "/path/a=b.wav"

    def test_input_file_absent_gives_none(self) -> None:
        assert StartupArgs.from_argv(["main.py"]).input_file is None


class TestPortArg:
    def test_default_port_is_zero(self) -> None:
        assert StartupArgs.from_argv(["main.py"]).port == 0

    def test_explicit_port_parsed(self) -> None:
        assert StartupArgs.from_argv(["main.py", "--port=8080"]).port == 8080

    def test_port_boundary_zero(self) -> None:
        assert StartupArgs.from_argv(["main.py", "--port=0"]).port == 0

    def test_port_boundary_max(self) -> None:
        assert StartupArgs.from_argv(["main.py", "--port=65535"]).port == 65535

    def test_port_too_high_exits_1(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            StartupArgs.from_argv(["main.py", "--port=65536"])
        assert exc_info.value.code == 1
        assert "65536" in capsys.readouterr().err

    def test_port_negative_exits_1(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            StartupArgs.from_argv(["main.py", "--port=-1"])
        assert exc_info.value.code == 1

    def test_port_non_integer_exits_1(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            StartupArgs.from_argv(["main.py", "--port=abc"])
        assert exc_info.value.code == 1
        assert "abc" in capsys.readouterr().err


class TestImmutability:
    def test_frozen_dataclass_rejects_mutation(self) -> None:
        args = StartupArgs.from_argv(["main.py"])
        with pytest.raises(FrozenInstanceError):
            args.verbose = True  # type: ignore[misc]
