"""Background download thread with throttled progress callback."""

import logging
import threading
import time
from typing import Callable

from src.asr.ModelDefinitions import IModelDefinition

logger = logging.getLogger(__name__)

_THROTTLE_INTERVAL = 0.5
_THROTTLE_DELTA = 0.01


class DownloadWorker:
    """Run a model download on a daemon thread with throttled progress.

    Thread-safety: ``_downloading`` is guarded by ``_lock``. ``start()`` is the
    only public mutating method and is safe to call from any thread.
    """

    def __init__(self) -> None:
        self._downloading = False
        self._current_model_name: str | None = None
        self._lock = threading.Lock()

    def is_downloading(self) -> bool:
        """Return True while a download thread is running."""
        with self._lock:
            return self._downloading

    def current_model_name(self) -> str | None:
        """Return the name of the model currently being downloaded, if any."""
        with self._lock:
            return self._current_model_name

    def start(
        self,
        model: IModelDefinition,
        progress_callback: Callable[[float, int, int], None],
        on_success: Callable[[str], None],
        on_error: Callable[[str, Exception], None],
    ) -> bool:
        """Spawn a daemon download thread for the requested model.

        Args:
            model: Downloadable model definition.
            progress_callback: Called with ``(progress, downloaded_bytes, total_bytes)``.
            on_success: Called with the model name after a successful download.
            on_error: Called with ``(model_name, exception)`` after a failed download.

        Returns:
            True if a thread was started, False if another download is in progress.
        """
        with self._lock:
            if self._downloading:
                return False
            self._downloading = True
            self._current_model_name = model.name

        throttled = self._make_throttled_callback(progress_callback)
        thread = threading.Thread(
            target=self._thread_body,
            args=(model, throttled, on_success, on_error),
            daemon=True,
            name=f"ModelDownload-{model.name}",
        )
        thread.start()
        return True

    def _make_throttled_callback(
        self,
        raw: Callable[[float, int, int], None],
    ) -> Callable[[float, int, int], None]:
        """Wrap the raw callback with interval and delta throttling."""
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
        model: IModelDefinition,
        progress_callback: Callable[[float, int, int], None],
        on_success: Callable[[str], None],
        on_error: Callable[[str, Exception], None],
    ) -> None:
        """Download thread entry point."""
        try:
            model.download(progress_callback)
            on_success(model.name)
        except Exception as exc:
            logger.exception("DownloadWorker: download failed for %s: %s", model.name, exc)
            on_error(model.name, exc)
        finally:
            with self._lock:
                self._downloading = False
                self._current_model_name = None
