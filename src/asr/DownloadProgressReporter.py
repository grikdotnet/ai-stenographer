"""Compatibility re-export — canonical location is src.client.tk.asr.DownloadProgressReporter."""
from src.client.tk.asr.DownloadProgressReporter import (
    ProgressAggregator,
    DownloadProgressReporter,
    create_bound_progress_reporter,
)
__all__ = ['ProgressAggregator', 'DownloadProgressReporter', 'create_bound_progress_reporter']
