"""
Tests for DownloadProgressReporter - progress tracking for model downloads.
"""
import pytest
from unittest.mock import MagicMock
from src.DownloadProgressReporter import (
    ProgressAggregator,
    DownloadProgressReporter,
    create_bound_progress_reporter
)


class TestProgressAggregator:
    """Test suite for ProgressAggregator."""

    def test_initial_state_is_empty(self):
        """Empty aggregator returns zero progress."""
        aggregator = ProgressAggregator()
        downloaded, total, percentage = aggregator.get_overall_progress()

        assert downloaded == 0
        assert total == 0
        assert percentage == 0.0

    def test_single_file_progress(self):
        """Tracks progress for single file."""
        aggregator = ProgressAggregator()
        aggregator.update_file_progress("file1", 50, 100)

        downloaded, total, percentage = aggregator.get_overall_progress()

        assert downloaded == 50
        assert total == 100
        assert percentage == 0.5

    def test_multiple_files_aggregation(self):
        """Aggregates progress across multiple files."""
        aggregator = ProgressAggregator()
        aggregator.update_file_progress("file1", 50, 100)
        aggregator.update_file_progress("file2", 25, 50)

        downloaded, total, percentage = aggregator.get_overall_progress()

        assert downloaded == 75  # 50 + 25
        assert total == 150      # 100 + 50
        assert percentage == 0.5  # 75/150

    def test_file_progress_updates(self):
        """Updates progress for existing file."""
        aggregator = ProgressAggregator()
        aggregator.update_file_progress("file1", 50, 100)
        aggregator.update_file_progress("file1", 100, 100)

        downloaded, total, percentage = aggregator.get_overall_progress()

        assert downloaded == 100
        assert total == 100
        assert percentage == 1.0


class TestBoundProgressReporter:
    """Test suite for create_bound_progress_reporter factory."""

    def test_creates_subclass_with_bound_variables(self):
        """Factory creates subclass with bound callback, model_name, aggregator."""
        callback = MagicMock()
        aggregator = ProgressAggregator()

        BoundReporter = create_bound_progress_reporter(
            callback=callback,
            model_name='parakeet',
            aggregator=aggregator
        )

        # Verify class variables are set
        assert BoundReporter._callback is callback
        assert BoundReporter._model_name == 'parakeet'
        assert BoundReporter._aggregator is aggregator

    def test_bound_reporter_inherits_from_download_progress_reporter(self):
        """Bound reporter is a subclass of DownloadProgressReporter."""
        callback = MagicMock()
        aggregator = ProgressAggregator()

        BoundReporter = create_bound_progress_reporter(
            callback=callback,
            model_name='parakeet',
            aggregator=aggregator
        )

        assert issubclass(BoundReporter, DownloadProgressReporter)

    def test_instance_uses_bound_variables(self):
        """Instance created from bound class uses bound variables."""
        callback = MagicMock()
        aggregator = ProgressAggregator()

        BoundReporter = create_bound_progress_reporter(
            callback=callback,
            model_name='parakeet',
            aggregator=aggregator
        )

        # Create instance (simulating huggingface_hub instantiation)
        instance = BoundReporter(total=1000)

        assert instance.callback is callback
        assert instance.model_name == 'parakeet'
        assert instance.aggregator is aggregator


class TestDownloadProgressReporter:
    """Test suite for DownloadProgressReporter."""

    def test_update_tracks_progress(self):
        """update() tracks cumulative bytes and updates aggregator."""
        callback = MagicMock()
        aggregator = ProgressAggregator()

        reporter = DownloadProgressReporter(
            callback=callback,
            model_name='parakeet',
            aggregator=aggregator,
            total=200 * 1024 * 1024  # 200 MB (exceeds MIN_REASONABLE_TOTAL)
        )

        # Simulate download progress
        reporter.update(50 * 1024 * 1024)  # 50 MB

        assert reporter.downloaded_bytes == 50 * 1024 * 1024

        # Verify aggregator was updated
        downloaded, total, percentage = aggregator.get_overall_progress()
        assert downloaded == 50 * 1024 * 1024
        assert total == 200 * 1024 * 1024

    def test_filters_name_kwarg(self):
        """__init__ filters out 'name' kwarg that tqdm doesn't accept."""
        callback = MagicMock()
        aggregator = ProgressAggregator()

        # This should not raise error (name kwarg filtered)
        reporter = DownloadProgressReporter(
            callback=callback,
            model_name='parakeet',
            aggregator=aggregator,
            name="file.onnx",  # This would break tqdm
            total=1000
        )

        assert reporter.model_name == 'parakeet'

    def test_forces_disable_true(self):
        """__init__ forces disable=True to prevent terminal output."""
        callback = MagicMock()
        aggregator = ProgressAggregator()

        reporter = DownloadProgressReporter(
            callback=callback,
            model_name='parakeet',
            aggregator=aggregator,
            total=1000,
            disable=False  # This will be overridden
        )

        # tqdm's disable attribute should be True
        assert reporter.disable is True

    def test_callback_receives_progress_updates(self):
        """Callback receives progress updates with 5 parameters."""
        callback = MagicMock()
        aggregator = ProgressAggregator()

        reporter = DownloadProgressReporter(
            callback=callback,
            model_name='parakeet',
            aggregator=aggregator,
            total=200 * 1024 * 1024  # 200 MB (exceeds MIN_REASONABLE_TOTAL)
        )

        # Simulate download progress
        reporter.update(50 * 1024 * 1024)  # 50 MB

        # Callback should be called with 5 parameters
        assert callback.call_count >= 1
        call_args = callback.call_args[0]
        assert len(call_args) == 5
        model_name, progress, status, downloaded_bytes, total_bytes = call_args
        assert model_name == 'parakeet'
        assert status == 'downloading'
        assert downloaded_bytes == 50 * 1024 * 1024
        assert total_bytes == 200 * 1024 * 1024

    def test_init_with_none_iterable(self):
        """__init__ handles None iterable without crashing (PRIMARY FIX)."""
        # When using BoundProgressReporter (created by create_bound_progress_reporter),
        # the class variables are already set, so we only need to pass args
        aggregator = ProgressAggregator()
        callback = MagicMock()

        # Create a bound reporter class (simulating create_bound_progress_reporter)
        BoundReporter = create_bound_progress_reporter(
            callback=callback,
            model_name='parakeet',
            aggregator=aggregator
        )

        # Simulate huggingface_hub calling tqdm_class(None)
        # This is what causes the actual bug!
        reporter = BoundReporter(None)  # None as first positional arg

        # Should initialize successfully without crashing
        assert reporter.model_name == 'parakeet'
        assert reporter.aggregator is aggregator

    def test_close_with_none_aggregator(self):
        """close() handles None aggregator without crashing."""
        callback = MagicMock()

        # Create reporter WITHOUT aggregator
        reporter = DownloadProgressReporter(
            callback=callback,
            model_name='parakeet',
            aggregator=None,
            total=1000
        )

        # This should NOT crash
        reporter.close()

        # Callback should NOT be called when aggregator is None
        assert callback.call_count == 0

    def test_update_with_none_aggregator(self):
        """update() handles None aggregator without crashing."""
        callback = MagicMock()

        # Create reporter WITHOUT aggregator
        reporter = DownloadProgressReporter(
            callback=callback,
            model_name='parakeet',
            aggregator=None,
            total=1000
        )

        # This should NOT crash
        reporter.update(100)

        # Callback should NOT be called when aggregator is None
        assert callback.call_count == 0

    def test_iter_with_none_iterable(self):
        """__iter__ handles None iterable without crashing."""
        aggregator = ProgressAggregator()
        callback = MagicMock()

        # Create a bound reporter class
        BoundReporter = create_bound_progress_reporter(
            callback=callback,
            model_name='parakeet',
            aggregator=aggregator
        )

        # Create reporter with None iterable (this is what causes the real bug)
        reporter = BoundReporter(None)

        # Try to iterate over the reporter (this is what crashes in the real bug)
        # Should return empty iterator, not crash
        result = list(reporter)
        assert result == []
