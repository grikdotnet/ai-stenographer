"""Background download thread with throttled progress callback.

Owns the daemon thread and progress throttling. No broadcasting — callers
inject on_success and on_error callbacks.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from src.downloader.ModelDownloader import ModelDownloader

logger = logging.getLogger(__name__)

_THROTTLE_INTERVAL = 0.5
_THROTTLE_DELTA = 0.01


class DownloadWorker:
    """Runs a model download on a daemon thread with throttled progress reporting.

    Thread-safety: ``_downloading`` flag is guarded by ``_lock``. ``start()``
    is the only public mutating method and is safe to call from any thread.

    Args:
        models_dir: Root models directory passed to download_parakeet.
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._downloader = ModelDownloader(models_dir)
        self._downloading = False
        self._lock = threading.Lock()

    def is_downloading(self) -> bool:
        """Return True while a download thread is running.

        Returns:
            True if a download is in progress.
        """
        with self._lock:
            return self._downloading

    def start(
        self,
        model_name: str,
        progress_callback: Callable[[float, int, int], None],
        on_success: Callable[[str], None],
        on_error: Callable[[str, Exception], None],
    ) -> bool:
        """Spawn a daemon download thread for model_name.

        Args:
            model_name: Model identifier (e.g. ``"parakeet"``).
            progress_callback: Called with ``(progress, downloaded_bytes, total_bytes)``
                after each chunk; will be wrapped with throttle logic.
            on_success: Called with ``model_name`` on successful completion.
            on_error: Called with ``(model_name, exception)`` on failure.

        Returns:
            True if the thread was started; False if a download is already running.
        """
        with self._lock:
            if self._downloading:
                return False
            self._downloading = True

        throttled = self._make_throttled_callback(progress_callback)
        thread = threading.Thread(
            target=self._thread_body,
            args=(model_name, throttled, on_success, on_error),
            daemon=True,
            name=f"ModelDownload-{model_name}",
        )
        thread.start()
        return True

    def _make_throttled_callback(
        self,
        raw: Callable[[float, int, int], None],
    ) -> Callable[[float, int, int], None]:
        """Wrap raw with interval + delta throttling.

        Algorithm:
            - Always passes through the first call.
            - Always passes through progress == 1.0.
            - Otherwise passes through only when elapsed >= _THROTTLE_INTERVAL
              AND delta >= _THROTTLE_DELTA since the last broadcast.

        Args:
            raw: Unthrottled callback accepting ``(progress, downloaded_bytes, total_bytes)``.

        Returns:
            Throttled callback with the same signature.
        """
        last_time: list[float] = [0.0]
        last_progress: list[float] = [-1.0]
        first_call: list[bool] = [True]

        def throttled(progress: float, downloaded_bytes: int, total_bytes: int) -> None:
            now = time.monotonic()

            if progress >= 1.0:
                raw(1.0, downloaded_bytes, total_bytes)
                last_time[0] = now
                last_progress[0] = 1.0
                first_call[0] = False
                return

            elapsed = now - last_time[0]
            delta = progress - last_progress[0]

            if first_call[0] or (elapsed >= _THROTTLE_INTERVAL and delta >= _THROTTLE_DELTA):
                raw(progress, downloaded_bytes, total_bytes)
                last_time[0] = now
                last_progress[0] = progress
                first_call[0] = False

        return throttled

    def _thread_body(
        self,
        model_name: str,
        progress_callback: Callable[[float, int, int], None],
        on_success: Callable[[str], None],
        on_error: Callable[[str, Exception], None],
    ) -> None:
        """Download thread entry point.

        Args:
            model_name: Model being downloaded.
            progress_callback: Already-throttled progress callback.
            on_success: Called with model_name on success.
            on_error: Called with (model_name, exception) on failure.
        """
        try:
            self._downloader.download_parakeet(progress_callback)
            on_success(model_name)
        except Exception as exc:
            logger.exception("DownloadWorker: download failed for %s: %s", model_name, exc)
            on_error(model_name, exc)
        finally:
            with self._lock:
                self._downloading = False
