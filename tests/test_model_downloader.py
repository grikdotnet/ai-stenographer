"""Tests for downloader progress and partial-file behavior."""

from pathlib import Path

from src.downloader.ModelDownloader import _calculate_overall_progress, _partial_download_path


class TestModelDownloaderProgress:
    def test_progress_is_weighted_by_downloaded_bytes(self) -> None:
        progress = _calculate_overall_progress(
            completed_downloaded=0,
            current_downloaded=5,
            overall_total=50,
        )

        assert progress == 0.1

    def test_progress_is_clamped_to_one(self) -> None:
        progress = _calculate_overall_progress(
            completed_downloaded=100,
            current_downloaded=10,
            overall_total=100,
        )

        assert progress == 1.0


class TestPartialDownloadPath:
    def test_partial_download_path_appends_partial_suffix(self) -> None:
        target = Path("/fake/models/parakeet/config.json")

        assert _partial_download_path(target) == Path("/fake/models/parakeet/config.json.partial")
