"""Tests for downloader progress and partial-file behavior."""

from pathlib import Path
import threading

import pytest

from src.downloader import ModelDownloader as downloader_module
from src.downloader.ModelDownloader import (
    DownloadCancelledError,
    _calculate_overall_progress,
    _partial_download_path,
    download_parakeet,
)


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


class _StreamingResponse:
    """Minimal requests response double for streaming downloader tests."""

    headers = {"Content-Length": "12"}

    def __init__(self, cancel_event: threading.Event) -> None:
        self._cancel_event = cancel_event

    def raise_for_status(self) -> None:
        return

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield b"first"
        self._cancel_event.set()
        yield b"second"

    def close(self) -> None:
        return


class TestDownloaderCancellation:
    def test_cancels_streaming_download_and_removes_partial_artifacts(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        cancel_event = threading.Event()
        monkeypatch.setattr(downloader_module, "PARAKEET_FILES", ["config.json"])
        monkeypatch.setattr(downloader_module, "_get_remote_file_size", lambda _url: 12)
        monkeypatch.setattr(
            downloader_module.requests,
            "get",
            lambda *_args, **_kwargs: _StreamingResponse(cancel_event),
        )

        with pytest.raises(DownloadCancelledError):
            download_parakeet(
                tmp_path,
                lambda _progress, _downloaded, _total: None,
                cancel_event=cancel_event,
            )

        parakeet_dir = tmp_path / "parakeet"
        assert not list(parakeet_dir.glob("*.partial"))
        assert not (parakeet_dir / "manifest.json").exists()
