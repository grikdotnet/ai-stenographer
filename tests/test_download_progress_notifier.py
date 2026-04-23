"""Tests for server-side download progress notification throttling and terminals."""

from __future__ import annotations

from src.network.types import WsDownloadProgress
from src.server.DownloadProgressNotifier import DownloadProgressNotifier


class FakeBroadcaster:
    """Capture broadcast server messages for assertions."""

    def __init__(self) -> None:
        self.messages: list[WsDownloadProgress] = []

    def broadcast(self, message: WsDownloadProgress) -> None:
        self.messages.append(message)


class TestDownloadProgressNotifier:
    def test_throttles_small_progress_updates(self, monkeypatch) -> None:
        broadcaster = FakeBroadcaster()
        notifier = DownloadProgressNotifier(broadcaster)
        times = iter([100.0, 100.1, 100.6])
        monkeypatch.setattr(
            "src.server.DownloadProgressNotifier.time.monotonic",
            lambda: next(times),
        )

        notifier.on_progress("parakeet", 0.10, 10, 100)
        notifier.on_progress("parakeet", 0.105, 11, 100)
        notifier.on_progress("parakeet", 0.12, 12, 100)

        assert len(broadcaster.messages) == 2
        assert broadcaster.messages[0] == WsDownloadProgress(
            model_name="parakeet",
            status="downloading",
            progress=0.10,
            downloaded_bytes=10,
            total_bytes=100,
        )
        assert broadcaster.messages[1] == WsDownloadProgress(
            model_name="parakeet",
            status="downloading",
            progress=0.12,
            downloaded_bytes=12,
            total_bytes=100,
        )

    def test_complete_emits_and_resets_throttle_state(self, monkeypatch) -> None:
        broadcaster = FakeBroadcaster()
        notifier = DownloadProgressNotifier(broadcaster)
        times = iter([200.0, 200.1, 200.2])
        monkeypatch.setattr(
            "src.server.DownloadProgressNotifier.time.monotonic",
            lambda: next(times),
        )

        notifier.on_progress("parakeet", 0.10, 10, 100)
        notifier.on_complete("parakeet")
        notifier.on_progress("parakeet", 0.1005, 11, 100)

        assert broadcaster.messages == [
            WsDownloadProgress(
                model_name="parakeet",
                status="downloading",
                progress=0.10,
                downloaded_bytes=10,
                total_bytes=100,
            ),
            WsDownloadProgress(
                model_name="parakeet",
                status="complete",
                progress=1.0,
            ),
            WsDownloadProgress(
                model_name="parakeet",
                status="downloading",
                progress=0.1005,
                downloaded_bytes=11,
                total_bytes=100,
            ),
        ]

    def test_error_emits_error_message_and_resets_throttle_state(
        self,
        monkeypatch,
    ) -> None:
        broadcaster = FakeBroadcaster()
        notifier = DownloadProgressNotifier(broadcaster)
        times = iter([300.0, 300.1, 300.2])
        monkeypatch.setattr(
            "src.server.DownloadProgressNotifier.time.monotonic",
            lambda: next(times),
        )

        notifier.on_progress("parakeet", 0.10, 10, 100)
        notifier.on_error("parakeet", RuntimeError("disk full"))
        notifier.on_progress("parakeet", 0.1005, 11, 100)

        assert broadcaster.messages == [
            WsDownloadProgress(
                model_name="parakeet",
                status="downloading",
                progress=0.10,
                downloaded_bytes=10,
                total_bytes=100,
            ),
            WsDownloadProgress(
                model_name="parakeet",
                status="error",
                error_message="disk full",
            ),
            WsDownloadProgress(
                model_name="parakeet",
                status="downloading",
                progress=0.1005,
                downloaded_bytes=11,
                total_bytes=100,
            ),
        ]
