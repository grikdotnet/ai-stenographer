"""Tests for downloader progress and partial-file behavior."""

from pathlib import Path

import pytest

from src.downloader.ModelDownloader import _calculate_overall_progress, _partial_download_path


class TestModelDownloaderProgress:
    @pytest.mark.parametrize(
        ("completed_downloaded", "current_downloaded", "overall_total", "expected_progress"),
        [
            (0, 5, 50, 0.1),
            (100, 10, 100, 1.0),
        ],
    )
    def test_progress_calculation(
        self,
        completed_downloaded: int,
        current_downloaded: int,
        overall_total: int,
        expected_progress: float,
    ) -> None:
        progress = _calculate_overall_progress(
            completed_downloaded=completed_downloaded,
            current_downloaded=current_downloaded,
            overall_total=overall_total,
        )

        assert progress == expected_progress


class TestPartialDownloadPath:
    def test_partial_download_path_appends_partial_suffix(self) -> None:
        target = Path("/fake/models/parakeet/config.json")

        assert _partial_download_path(target) == Path("/fake/models/parakeet/config.json.partial")
