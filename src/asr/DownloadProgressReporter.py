"""
DownloadProgressReporter captures download progress from huggingface_hub.

This module provides custom tqdm classes that intercept progress from
huggingface_hub's snapshot_download() and hf_hub_download() functions.
"""
import logging
import threading
from typing import Callable, Optional, Dict, Tuple
from tqdm.auto import tqdm


class ProgressAggregator:
    """
    Thread-safe aggregator for tracking progress across multiple files.

    Used when snapshot_download() downloads multiple files concurrently
    (max_workers=8). Each file gets its own DownloadProgressReporter,
    and this class aggregates their progress into overall statistics.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._file_progress: Dict[str, Tuple[int, int]] = {}  # file_id -> (downloaded, total)

    def update_file_progress(self, file_id: str, downloaded: int, total: int):
        """
        Update progress for a specific file.

        Args:
            file_id: Unique identifier for the file (e.g., file path)
            downloaded: Bytes downloaded so far
            total: Total bytes to download
        """
        with self._lock:
            self._file_progress[file_id] = (downloaded, total)

    def get_overall_progress(self) -> Tuple[int, int, float]:
        """
        Get aggregated progress across all files.

        Returns:
            Tuple of (total_downloaded, total_size, percentage)
            percentage is 0.0 to 1.0
        """
        with self._lock:
            if not self._file_progress:
                return (0, 0, 0.0)

            total_downloaded = sum(downloaded for downloaded, _ in self._file_progress.values())
            total_size = sum(total for _, total in self._file_progress.values())

            percentage = total_downloaded / total_size if total_size > 0 else 0.0

            return (total_downloaded, total_size, percentage)


class DownloadProgressReporter(tqdm):
    """
    Custom tqdm class that reports download progress to a callback.

    Inherits from tqdm.auto.tqdm to satisfy huggingface_hub's requirements
    for the tqdm_class parameter. Intercepts update() calls to track
    download progress and forward it to a callback function.
    """

    # Class variables to store parameters (set by BoundProgressReporter subclass)
    _callback = None
    _model_name = None
    _aggregator = None

    def __init__(
        self,
        callback: Optional[Callable[[str, float, str, int, int], None]] = None,
        model_name: Optional[str] = None,
        aggregator: Optional[ProgressAggregator] = None,
        *args,
        **kwargs
    ):
        """
        Initialize progress reporter.

        Args:
            callback: Function to call with progress updates.
                Signature: callback(model_name, progress, status, downloaded_bytes, total_bytes)
                - model_name: Name of the model being downloaded
                - progress: 0.0 to 1.0 (percentage)
                - status: 'downloading' | 'complete' | 'error'
                - downloaded_bytes: Cumulative bytes downloaded
                - total_bytes: Total bytes to download
            model_name: Name of the model (e.g., 'parakeet', 'silero_vad')
            aggregator: ProgressAggregator for multi-file tracking
            *args, **kwargs: Passed to tqdm.__init__
        """
        # Use provided parameters or fall back to class variables (set by subclass)
        self.callback = callback if callback is not None else self.__class__._callback
        self.model_name = model_name if model_name is not None else self.__class__._model_name
        self.aggregator = aggregator if aggregator is not None else self.__class__._aggregator

        logging.debug(f"DownloadProgressReporter.__init__: model={self.model_name}, callback={self.callback is not None}, aggregator={self.aggregator is not None}")

        self.file_id = f"{self.model_name}_{id(self)}"
        self.downloaded_bytes = 0

        # Filter out 'name' parameter which huggingface_hub passes and tqdm doesn't accept
        tqdm_kwargs = {k: v for k, v in kwargs.items() if k != 'name'}

        # IMPORTANT: Force disable=True to prevent terminal output
        tqdm_kwargs['disable'] = True

        # BUGFIX: Handle None iterable passed by huggingface_hub
        # If first positional arg is None, replace it with an empty list
        # We can't just remove it because tqdm needs SOMETHING to iterate over
        if args and args[0] is None:
            # Replace None with empty list - tqdm can iterate over empty list without errors
            args = ([], *args[1:])
            logging.debug(f"DownloadProgressReporter.__init__: Replaced None iterable with empty list for {self.model_name}")

        # Call tqdm's __init__ with filtered kwargs
        super().__init__(*args, **tqdm_kwargs)

    def __iter__(self):
        """
        Override tqdm's __iter__ to handle None iterable gracefully.
        """
        if self.iterable is None:
            logging.debug(f"DownloadProgressReporter.__iter__: iterable is None, returning empty iterator for {self.model_name}")
            return iter([])  # instead of crashing
        return super().__iter__()

    def update(self, n: int = 1):
        """
        Called by huggingface_hub when bytes are downloaded.

        Args:
            n: Number of bytes downloaded since last update
        """
        # Track cumulative bytes
        self.downloaded_bytes += n

        # Update aggregator with this file's progress
        total_bytes = self.total if self.total is not None else 0
        if self.aggregator:
            self.aggregator.update_file_progress(
                self.file_id,
                self.downloaded_bytes,
                total_bytes
            )

            overall_downloaded, overall_total, overall_percentage = self.aggregator.get_overall_progress()

            # Call progress callback only if we have a reasonable total
            MIN_REASONABLE_TOTAL = 100 * 1024 * 1024  # 100 MB

            logging.debug(f"Progress update: {self.model_name} {overall_percentage:.1%} ({overall_downloaded}/{overall_total})")
            logging.debug(f"  callback={self.callback}, callable={callable(self.callback) if self.callback else False}, total={overall_total}, threshold={MIN_REASONABLE_TOTAL}")

            if self.callback and callable(self.callback) and overall_total >= MIN_REASONABLE_TOTAL:
                try:
                    self.callback(
                        self.model_name,
                        overall_percentage,
                        "downloading",
                        overall_downloaded,
                        overall_total
                    )
                except Exception as e:
                    logging.error(f"  Callback failed: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
        else:
            # No aggregator available - just update tqdm and exit early
            logging.debug(f"DownloadProgressReporter.update() called without aggregator for {self.model_name}")
            super().update(n)
            return

        # Update tqdm's internal state AFTER our tracking to avoid double-counting
        super().update(n)

    def close(self):
        """Called when download completes."""
        super().close()

        if not self.aggregator:
            # No aggregator - this shouldn't happen in normal operation
            logging.debug(f"DownloadProgressReporter.close() called without aggregator for {self.model_name}")
            return

        total_bytes = self.total if self.total is not None else 0
        self.aggregator.update_file_progress(
            self.file_id,
            self.downloaded_bytes,
            total_bytes
        )

        overall_downloaded, overall_total, overall_percentage = self.aggregator.get_overall_progress()

        if self.callback and callable(self.callback):
            try:
                self.callback(
                    self.model_name,
                    overall_percentage,
                    "downloading",  # Still downloading, model may have multiple files
                    overall_downloaded,
                    overall_total
                )
            except Exception as e:
                logging.error(f"DownloadProgressReporter.close() callback error: {e}")


def create_bound_progress_reporter(
    callback: Callable[[str, float, str, int, int], None],
    model_name: str,
    aggregator: ProgressAggregator
) -> type:
    """
    Factory function that creates a DownloadProgressReporter subclass with bound variables.

    huggingface_hub instantiates the tqdm_class internally, so we cannot pass instance parameters directly. 
    Instead, we create a subclass with class variables set to the desired values.

    Args:
        callback: Progress callback function (5 parameters)
            Signature: callback(model_name, progress, status, downloaded_bytes, total_bytes)
        model_name: Model being downloaded (e.g., 'parakeet', 'silero_vad')
        aggregator: ProgressAggregator for multi-file tracking

    Returns:
        A new class (subclass of DownloadProgressReporter) with bound variables

    Example:
        >>> aggregator = ProgressAggregator()
        >>> BoundReporter = create_bound_progress_reporter(my_callback, 'parakeet', aggregator)
        >>> # Pass to huggingface_hub
        >>> snapshot_download(..., tqdm_class=BoundReporter)
    """
    class BoundProgressReporter(DownloadProgressReporter):
        """Subclass with bound callback, model_name, and aggregator."""
        _callback = callback
        _model_name = model_name
        _aggregator = aggregator

    return BoundProgressReporter
