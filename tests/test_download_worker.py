"""Tests for background model download worker cancellation."""

from __future__ import annotations

import threading

from src.downloader.DownloadWorker import DownloadWorker
from src.downloader.ModelDownloader import DownloadCancelledError


class _CancellableModel:
    """Downloadable model fake that blocks until cancellation is requested."""

    name = "parakeet"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancel_event: threading.Event | None = None

    def download(self, progress_callback, cancel_event=None) -> None:
        del progress_callback
        self.cancel_event = cancel_event
        self.started.set()
        assert cancel_event is not None
        cancel_event.wait(timeout=2)
        raise DownloadCancelledError("cancelled")


class TestDownloadWorkerCancellation:
    def test_cancel_returns_false_when_no_download_is_running(self) -> None:
        worker = DownloadWorker()

        assert worker.cancel() is False

    def test_cancel_sets_event_and_routes_cancelled_callback(self) -> None:
        worker = DownloadWorker()
        model = _CancellableModel()
        cancelled: list[str] = []
        errors: list[tuple[str, Exception]] = []

        started = worker.start(
            model=model,
            progress_callback=lambda _progress, _downloaded, _total: None,
            on_success=lambda _model_name: None,
            on_error=lambda model_name, exc: errors.append((model_name, exc)),
            on_cancelled=lambda model_name: cancelled.append(model_name),
        )

        assert started is True
        assert model.started.wait(timeout=2)
        assert worker.cancel() is True

        for _ in range(100):
            if not worker.is_downloading():
                break
            threading.Event().wait(0.01)

        assert model.cancel_event is not None
        assert model.cancel_event.is_set()
        assert cancelled == ["parakeet"]
        assert errors == []
