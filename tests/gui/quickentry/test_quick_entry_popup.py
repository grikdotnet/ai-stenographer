"""Tests for QuickEntryPopup - floating popup window for Quick Entry."""
import pytest
from unittest.mock import MagicMock, patch
import tkinter as tk

pytestmark = pytest.mark.gui


@pytest.fixture
def root():
    """Create a tkinter root window for testing."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


class TestQuickEntryPopup:
    """Test QuickEntryPopup GUI behavior."""

    def test_starts_hidden(self, root):
        """Popup should start in hidden state."""
        from src.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)

        assert popup.is_visible() is False

    def test_show_makes_visible(self, root):
        """show() should make popup visible."""
        from src.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        popup.show()
        root.update()

        assert popup.is_visible() is True

    def test_hide_makes_invisible(self, root):
        """hide() should hide the popup."""
        from src.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        popup.show()
        root.update()
        popup.hide()
        root.update()

        assert popup.is_visible() is False

    def test_set_text_updates_display(self, root):
        """set_text() should update displayed text."""
        from src.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        popup.show()
        popup.set_text("Hello world")
        root.update()

        assert popup.get_text() == "Hello world"

    def test_enter_key_triggers_submit(self, root):
        """Pressing Enter should trigger on_submit callback."""
        from src.quickentry.QuickEntryPopup import QuickEntryPopup

        on_submit = MagicMock()
        popup = QuickEntryPopup(root, on_submit=on_submit, on_cancel=lambda: None)
        popup.show()
        root.update()

        # Simulate Enter key press
        popup._window.event_generate('<Return>')
        root.update()

        on_submit.assert_called_once()

    def test_escape_key_triggers_cancel(self, root):
        """Pressing Escape should trigger on_cancel callback."""
        from src.quickentry.QuickEntryPopup import QuickEntryPopup

        on_cancel = MagicMock()
        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=on_cancel)
        popup.show()
        root.update()

        # Simulate Escape key press
        popup._window.event_generate('<Escape>')
        root.update()

        on_cancel.assert_called_once()

    def test_window_is_topmost(self, root):
        """Popup window should have topmost attribute."""
        from src.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)

        # Check topmost attribute
        assert popup._window.attributes('-topmost') == 1

    def test_window_is_borderless(self, root):
        """Popup window should be borderless (overrideredirect)."""
        from src.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)

        assert popup._window.overrideredirect() is True

    def test_displays_full_long_text(self, root):
        """Long text should be displayed in full (multiline), get_text returns original."""
        from src.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        long_text = "a" * 100  # Very long text
        popup.set_text(long_text)

        # get_text() returns original text (for submission)
        assert popup.get_text() == long_text

        # Display label shows full text with cursor
        displayed = popup._text_label.cget('text')
        assert displayed == long_text + "_"

    def test_clear_text(self, root):
        """Setting empty text should clear display."""
        from src.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        popup.set_text("Hello")
        popup.set_text("")

        assert popup.get_text() == ""

    def test_destroy_cleans_up(self, root):
        """destroy() should clean up window resources."""
        from src.quickentry.QuickEntryPopup import QuickEntryPopup

        popup = QuickEntryPopup(root, on_submit=lambda: None, on_cancel=lambda: None)
        popup.destroy()

        # Window should no longer exist
        assert popup._window is None or not popup._window.winfo_exists()
