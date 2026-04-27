"""Tests for StartupArgs CLI argument parsing."""

from dataclasses import FrozenInstanceError

import pytest

from src.StartupArgs import StartupArgs


class TestDefaults:
    """No flags -> safe defaults."""

    @pytest.mark.parametrize("argv", [["main.py"], []])
    def test_defaults_are_applied(self, argv: list[str]) -> None:
        args = StartupArgs.from_argv(argv)
        assert args.verbose is False
        assert args.download_model is False
        assert args.host is None
        assert args.port == 0
        assert not hasattr(args, "server_only")
        assert not hasattr(args, "input_file")


class TestVerboseFlag:
    def test_verbose_short_flag(self) -> None:
        assert StartupArgs.from_argv(["main.py", "-v"]).verbose is True

    def test_verbose_not_set_without_flag(self) -> None:
        assert StartupArgs.from_argv(["main.py"]).verbose is False

    def test_verbose_flag_position_independence(self) -> None:
        assert StartupArgs.from_argv(["-v", "main.py"]).verbose is True
        assert StartupArgs.from_argv(["main.py", "--server-only", "-v"]).verbose is True


class TestServerOnlyFlag:
    def test_server_only_flag_is_accepted_and_ignored(self) -> None:
        args = StartupArgs.from_argv(["main.py", "--server-only"])
        assert not hasattr(args, "server_only")

    def test_server_only_combined_with_verbose(self) -> None:
        args = StartupArgs.from_argv(["main.py", "-v", "--server-only"])
        assert args.verbose is True
        assert not hasattr(args, "server_only")


class TestDownloadModelFlag:
    def test_download_model_flag_sets_download_model(self) -> None:
        args = StartupArgs.from_argv(["main.py", "--download-model"])
        assert args.download_model is True

    def test_download_model_not_set_by_default(self) -> None:
        assert StartupArgs.from_argv(["main.py"]).download_model is False

    def test_download_model_still_works_with_ignored_server_only_flag(self) -> None:
        args = StartupArgs.from_argv(["main.py", "--server-only", "--download-model"])
        assert args.download_model is True


class TestInputFileArg:
    def test_input_file_is_not_part_of_python_startup_args(self) -> None:
        args = StartupArgs.from_argv(["main.py", "--input-file=/path/to/file.wav"])
        assert not hasattr(args, "input_file")

    def test_input_file_with_equals_in_path_is_ignored(self) -> None:
        args = StartupArgs.from_argv(["main.py", "--input-file=/path/a=b.wav"])
        assert not hasattr(args, "input_file")

    def test_input_file_absent_still_has_no_field(self) -> None:
        assert not hasattr(StartupArgs.from_argv(["main.py"]), "input_file")


class TestHostArg:
    def test_host_absent_gives_none(self) -> None:
        assert StartupArgs.from_argv(["main.py"]).host is None

    def test_host_parsed_correctly(self) -> None:
        args = StartupArgs.from_argv(["main.py", "--host=localhost"])
        assert args.host == "localhost"


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
