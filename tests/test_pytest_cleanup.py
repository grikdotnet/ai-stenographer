"""Tests for pytest session artifact cleanup."""

from pathlib import Path

from conftest import cleanup_test_artifact_dirs


def test_cleanup_test_artifact_dirs_removes_only_known_artifacts(tmp_path: Path) -> None:
    root_path = tmp_path / "repo"
    cache_dir = root_path / ".pytest_cache"
    tmp_dir = root_path / ".pytest_tmp"
    keeper_dir = root_path / "logs"

    cache_dir.mkdir(parents=True)
    tmp_dir.mkdir()
    keeper_dir.mkdir()

    cleanup_test_artifact_dirs(root_path)

    assert not cache_dir.exists()
    assert not tmp_dir.exists()
    assert keeper_dir.exists()
