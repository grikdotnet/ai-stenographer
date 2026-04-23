"""Broadcast throttled download progress updates to active server sessions."""

from __future__ import annotations

import logging
import threading
import time

from src.network.types import WsDownloadProgress
from src.server.protocols import IServerMessageBroadcaster

logger = logging.getLogger(__name__)

_THROTTLE_INTERVAL_SECONDS = 0.5
_THROTTLE_PROGRESS_DELTA = 0.01


class DownloadProgressNotifier:
    """Publish model download progress updates through the server message broadcaster.

    Responsibilities:
    - throttle transient ``downloading`` updates per model
    - emit terminal ``complete`` and ``error`` updates immediately
    - reset throttling state after terminal updates
    """

    def __init__(self, broadcaster: IServerMessageBroadcaster) -> None:
        """Create a notifier backed by a server message broadcaster.

        Args:
            broadcaster: Server-wide broadcaster that encodes and fans out wire messages.
        """
        self._broadcaster = broadcaster
        self._lock = threading.Lock()
        self._last_emit_time: dict[str, float] = {}
        self._last_emit_progress: dict[str, float] = {}

    def on_progress(
        self,
        model_name: str,
        progress: float,
        downloaded_bytes: int,
        total_bytes: int,
    ) -> None:
        """Publish a throttled in-progress download update.

        Args:
            model_name: Downloaded model name.
            progress: Fractional completion value in the range ``[0.0, 1.0]``.
            downloaded_bytes: Bytes downloaded so far.
            total_bytes: Total bytes expected for the download.
        """
        if not self._should_emit(model_name, progress):
            return

        self._broadcaster.broadcast(
            WsDownloadProgress(
                model_name=model_name,
                status="downloading",
                progress=progress,
                downloaded_bytes=downloaded_bytes,
                total_bytes=total_bytes,
            )
        )

    def on_complete(self, model_name: str) -> None:
        """Publish a terminal completion update.

        Args:
            model_name: Downloaded model name.
        """
        self._clear_model_state(model_name)
        self._broadcaster.broadcast(
            WsDownloadProgress(
                model_name=model_name,
                status="complete",
                progress=1.0,
            )
        )

    def on_error(self, model_name: str, exc: Exception) -> None:
        """Publish a terminal failure update.

        Args:
            model_name: Downloaded model name.
            exc: Exception raised by the download worker.
        """
        self._clear_model_state(model_name)
        self._broadcaster.broadcast(
            WsDownloadProgress(
                model_name=model_name,
                status="error",
                error_message=str(exc),
            )
        )

    def _should_emit(self, model_name: str, progress: float) -> bool:
        """Return whether a throttled progress update should be emitted."""
        now = time.monotonic()
        with self._lock:
            last_time = self._last_emit_time.get(model_name)
            last_progress = self._last_emit_progress.get(model_name, -1.0)
            if (
                last_time is not None
                and now - last_time < _THROTTLE_INTERVAL_SECONDS
                and progress - last_progress < _THROTTLE_PROGRESS_DELTA
            ):
                return False

            self._last_emit_time[model_name] = now
            self._last_emit_progress[model_name] = progress
            return True

    def _clear_model_state(self, model_name: str) -> None:
        """Forget the last throttling state for one model."""
        with self._lock:
            self._last_emit_time.pop(model_name, None)
            self._last_emit_progress.pop(model_name, None)
