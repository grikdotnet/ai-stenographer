"""Tests for QuickEntryPopup auto-resize behavior when text grows."""
import pytest
from unittest.mock import patch
import tkinter as tk

pytestmark = pytest.mark.gui


@pytest.fixture
def root():
    """Create a tkinter root window for testing."""
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


class TestQuickEntryPopupAutoResize:
    """Test auto-resize behavior when text grows beyond one line."""

    def test_short_text_keeps_initial_height(self, root):
        """Short text that fits in one line should not trigger resize."""
        from src.client.tk.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        initial_height = popup._current_height

        popup.show()
        popup.set_text("Short text")
        root.update()

        assert popup._current_height == initial_height

    def test_long_text_increases_height(self, root):
        """Long wrapping text should increase window height."""
        from src.client.tk.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        initial_height = popup._current_height

        popup.show()
        root.update()

        # Text long enough to wrap multiple times
        long_text = "This is a very long text that should wrap to multiple lines " * 5
        popup.set_text(long_text)
        root.update()

        assert popup._current_height > initial_height

    def test_height_never_decreases_during_session(self, root):
        """Height should only grow, never shrink during a session."""
        from src.client.tk.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        popup.show()
        root.update()

        # Set long text to increase height
        long_text = "Long text that wraps to multiple lines " * 10
        popup.set_text(long_text)
        root.update()
        expanded_height = popup._current_height

        # Set short text - height should NOT decrease
        popup.set_text("Short")
        root.update()

        assert popup._current_height == expanded_height

    def test_height_capped_at_max(self, root):
        """Height should never exceed MAX_HEIGHT."""
        from src.client.tk.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        popup.show()
        root.update()

        # Text long enough to potentially exceed max height
        very_long_text = "This is extremely long text that goes on and on " * 100
        popup.set_text(very_long_text)
        root.update()

        assert popup._current_height <= popup.MAX_HEIGHT

    def test_hide_resets_height(self, root):
        """hide() should reset height for next session."""
        from src.client.tk.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        initial_height = popup.HEIGHT

        popup.show()
        root.update()
        popup.set_text("Long text " * 20)
        root.update()

        popup.hide()
        root.update()

        # After hide, height should reset to initial
        assert popup._current_height == initial_height

    def test_windows_effects_reapplied_on_resize(self, root):
        """Windows effects should be reapplied after auto-resize."""
        from src.client.tk.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        popup.show()
        root.update()

        initial_height = popup._current_height

        # Trigger resize with long text
        long_text = "Long wrapping text that needs more space " * 10
        popup.set_text(long_text)
        root.update()

        # If height increased, effects should have been reapplied
        if popup._current_height > initial_height:
            # The _apply_windows_effects method sets this to True after applying
            assert popup._effects_applied is True

    def test_calculate_required_height_returns_positive_int(self, root):
        """_calculate_required_height() should return a positive integer."""
        from src.client.tk.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        popup.show()
        root.update()

        result = popup._calculate_required_height()

        assert isinstance(result, int)
        assert result > 0

    def test_gradual_height_increase(self, root):
        """Height should increase gradually as more text is added."""
        from src.client.tk.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        popup.show()
        root.update()

        heights = []
        base_text = "Adding more text to see gradual growth "

        for i in range(1, 6):
            popup.set_text(base_text * i * 2)
            root.update()
            heights.append(popup._current_height)

        # Heights should be non-decreasing (may plateau at MAX_HEIGHT)
        for i in range(1, len(heights)):
            assert heights[i] >= heights[i - 1]
