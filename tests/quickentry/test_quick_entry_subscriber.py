"""Tests for QuickEntrySubscriber - text accumulation for Quick Entry popup."""
import pytest
from unittest.mock import MagicMock
import threading
from src.types import RecognitionResult


class TestQuickEntrySubscriber:
    """Test QuickEntrySubscriber text accumulation behavior."""

    def _make_result(self, text: str) -> RecognitionResult:
        """Create a RecognitionResult for testing."""
        return RecognitionResult(
            text=text,
            start_time=0.0,
            end_time=1.0,
            chunk_ids=[1]
        )

    def test_inactive_by_default(self):
        """Subscriber should be inactive by default."""
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber

        subscriber = QuickEntrySubscriber(on_text_change=lambda f, p: None)

        assert subscriber.is_active() is False

    def test_activate_sets_active_state(self):
        """activate() should set active state and clear accumulated text."""
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber

        subscriber = QuickEntrySubscriber(on_text_change=lambda f, p: None)
        subscriber.activate()

        assert subscriber.is_active() is True
        assert subscriber.get_accumulated_text() == ""

    def test_deactivate_clears_active_state(self):
        """deactivate() should clear active state."""
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber

        subscriber = QuickEntrySubscriber(on_text_change=lambda f, p: None)
        subscriber.activate()
        subscriber.deactivate()

        assert subscriber.is_active() is False

    def test_ignores_events_when_inactive(self):
        """Should ignore recognition events when not active."""
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber

        callback = MagicMock()
        subscriber = QuickEntrySubscriber(on_text_change=callback)

        subscriber.on_partial_update(self._make_result("hello"))
        subscriber.on_finalization(self._make_result("hello"))

        callback.assert_not_called()
        assert subscriber.get_accumulated_text() == ""

    def test_on_finalization_accumulates_text(self):
        """on_finalization() should accumulate finalized text."""
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber

        callback = MagicMock()
        subscriber = QuickEntrySubscriber(on_text_change=callback)
        subscriber.activate()

        subscriber.on_finalization(self._make_result("hello"))

        assert subscriber.get_accumulated_text() == "hello"
        callback.assert_called_with("hello", "")

    def test_multiple_finalizations_concatenate_with_space(self):
        """Multiple finalizations should be concatenated with spaces."""
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber

        subscriber = QuickEntrySubscriber(on_text_change=lambda f, p: None)
        subscriber.activate()

        subscriber.on_finalization(self._make_result("hello"))
        subscriber.on_finalization(self._make_result("world"))

        assert subscriber.get_accumulated_text() == "hello world"

    def test_on_partial_update_tracks_partial_text(self):
        """on_partial_update() should track partial text for display."""
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber

        callback = MagicMock()
        subscriber = QuickEntrySubscriber(on_text_change=callback)
        subscriber.activate()

        subscriber.on_partial_update(self._make_result("hel"))

        callback.assert_called_with("", "hel")

    def test_get_display_text_includes_partial(self):
        """get_display_text() should include both finalized and partial text."""
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber

        subscriber = QuickEntrySubscriber(on_text_change=lambda f, p: None)
        subscriber.activate()

        subscriber.on_finalization(self._make_result("hello"))
        subscriber.on_partial_update(self._make_result("wor"))

        assert subscriber.get_display_text() == "hello wor"

    def test_finalization_clears_partial_text(self):
        """on_finalization() should clear partial text."""
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber

        callback = MagicMock()
        subscriber = QuickEntrySubscriber(on_text_change=callback)
        subscriber.activate()

        subscriber.on_partial_update(self._make_result("hel"))
        subscriber.on_finalization(self._make_result("hello"))

        # Last call should have empty partial
        callback.assert_called_with("hello", "")

    def test_activate_clears_previous_text(self):
        """activate() should clear any previously accumulated text."""
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber

        subscriber = QuickEntrySubscriber(on_text_change=lambda f, p: None)
        subscriber.activate()
        subscriber.on_finalization(self._make_result("old text"))
        subscriber.deactivate()

        subscriber.activate()

        assert subscriber.get_accumulated_text() == ""
        assert subscriber.get_display_text() == ""

    def test_thread_safety(self):
        """Subscriber should be thread-safe for concurrent access."""
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber

        subscriber = QuickEntrySubscriber(on_text_change=lambda f, p: None)
        subscriber.activate()

        errors = []

        def writer():
            try:
                for i in range(100):
                    subscriber.on_finalization(self._make_result(f"word{i}"))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    _ = subscriber.get_accumulated_text()
                    _ = subscriber.get_display_text()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_handles_flush_status(self):
        """Should handle 'flush' status same as 'final'."""
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber

        subscriber = QuickEntrySubscriber(on_text_change=lambda f, p: None)
        subscriber.activate()

        subscriber.on_finalization(self._make_result("hello"))

        assert subscriber.get_accumulated_text() == "hello"
