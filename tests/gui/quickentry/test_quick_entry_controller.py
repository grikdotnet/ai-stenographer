"""Tests for QuickEntryController - orchestrates Quick Entry workflow."""
import pytest
from unittest.mock import MagicMock, patch, call
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


@pytest.fixture
def mock_components():
    """Create mock components for controller testing."""
    return {
        'subscriber': MagicMock(),
        'focus_tracker': MagicMock(),
        'keyboard': MagicMock(),
    }


class TestQuickEntryController:
    """Test QuickEntryController workflow orchestration."""

    def test_on_hotkey_schedules_toggle_on_main_thread(self, root, mock_components):
        """on_hotkey() should schedule _toggle_popup via root.after()."""
        from src.quickentry.QuickEntryController import QuickEntryController

        controller = QuickEntryController(
            subscriber=mock_components['subscriber'],
            focus_tracker=mock_components['focus_tracker'],
            keyboard_simulator=mock_components['keyboard'],
            root=root
        )

        # Mock root.after to verify it's called
        with patch.object(root, 'after') as mock_after:
            controller.on_hotkey()
            mock_after.assert_called_once()
            # First arg should be 0 (immediate scheduling)
            assert mock_after.call_args[0][0] == 0

    def test_show_popup_saves_focus_and_activates_subscriber(self, root, mock_components):
        """Showing popup should save focus and activate subscriber."""
        from src.quickentry.QuickEntryController import QuickEntryController

        controller = QuickEntryController(
            subscriber=mock_components['subscriber'],
            focus_tracker=mock_components['focus_tracker'],
            keyboard_simulator=mock_components['keyboard'],
            root=root
        )

        controller._show_popup()
        root.update()

        mock_components['focus_tracker'].save_focus.assert_called_once()
        mock_components['subscriber'].activate.assert_called_once()

    def test_show_popup_creates_and_shows_popup(self, root, mock_components):
        """Showing popup should create popup window and show it."""
        from src.quickentry.QuickEntryController import QuickEntryController

        controller = QuickEntryController(
            subscriber=mock_components['subscriber'],
            focus_tracker=mock_components['focus_tracker'],
            keyboard_simulator=mock_components['keyboard'],
            root=root
        )

        controller._show_popup()
        root.update()

        assert controller._popup is not None
        assert controller._is_showing is True

    def test_toggle_hides_if_already_showing(self, root, mock_components):
        """toggle_popup() should hide popup if already showing."""
        from src.quickentry.QuickEntryController import QuickEntryController

        controller = QuickEntryController(
            subscriber=mock_components['subscriber'],
            focus_tracker=mock_components['focus_tracker'],
            keyboard_simulator=mock_components['keyboard'],
            root=root
        )

        # Show first
        controller._show_popup()
        root.update()

        # Toggle should hide
        controller._toggle_popup()
        root.update()

        assert controller._is_showing is False
        mock_components['subscriber'].deactivate.assert_called()

    def test_on_submit_gets_text_and_inserts(self, root, mock_components):
        """on_submit should get accumulated text and insert it."""
        from src.quickentry.QuickEntryController import QuickEntryController

        mock_components['subscriber'].get_accumulated_text.return_value = "hello world"
        mock_components['focus_tracker'].restore_focus.return_value = True

        controller = QuickEntryController(
            subscriber=mock_components['subscriber'],
            focus_tracker=mock_components['focus_tracker'],
            keyboard_simulator=mock_components['keyboard'],
            root=root
        )

        controller._show_popup()
        root.update()
        controller._on_submit()

        mock_components['subscriber'].get_accumulated_text.assert_called()
        mock_components['subscriber'].deactivate.assert_called()
        mock_components['focus_tracker'].restore_focus.assert_called()

    def test_on_submit_types_text_after_delay(self, root, mock_components):
        """on_submit should type text after delay for focus restoration."""
        from src.quickentry.QuickEntryController import QuickEntryController

        mock_components['subscriber'].get_accumulated_text.return_value = "hello"

        controller = QuickEntryController(
            subscriber=mock_components['subscriber'],
            focus_tracker=mock_components['focus_tracker'],
            keyboard_simulator=mock_components['keyboard'],
            root=root
        )

        controller._show_popup()
        root.update()
        controller._on_submit()

        # Process the delayed callback
        root.after(100, root.quit)
        root.mainloop()

        mock_components['keyboard'].type_text.assert_called_with("hello")

    def test_on_submit_skips_empty_text(self, root, mock_components):
        """on_submit should not type if accumulated text is empty."""
        from src.quickentry.QuickEntryController import QuickEntryController

        mock_components['subscriber'].get_accumulated_text.return_value = "   "

        controller = QuickEntryController(
            subscriber=mock_components['subscriber'],
            focus_tracker=mock_components['focus_tracker'],
            keyboard_simulator=mock_components['keyboard'],
            root=root
        )

        controller._show_popup()
        root.update()
        controller._on_submit()

        # Process any pending callbacks
        root.after(100, root.quit)
        root.mainloop()

        mock_components['keyboard'].type_text.assert_not_called()

    def test_on_cancel_hides_without_insertion(self, root, mock_components):
        """on_cancel should hide popup without inserting text."""
        from src.quickentry.QuickEntryController import QuickEntryController

        mock_components['subscriber'].get_accumulated_text.return_value = "hello"

        controller = QuickEntryController(
            subscriber=mock_components['subscriber'],
            focus_tracker=mock_components['focus_tracker'],
            keyboard_simulator=mock_components['keyboard'],
            root=root
        )

        controller._show_popup()
        root.update()
        controller._on_cancel()

        mock_components['subscriber'].deactivate.assert_called()
        mock_components['focus_tracker'].restore_focus.assert_called()
        mock_components['keyboard'].type_text.assert_not_called()
        assert controller._is_showing is False

    def test_on_text_change_schedules_display_update(self, root, mock_components):
        """on_text_change should schedule popup text update."""
        from src.quickentry.QuickEntryController import QuickEntryController

        controller = QuickEntryController(
            subscriber=mock_components['subscriber'],
            focus_tracker=mock_components['focus_tracker'],
            keyboard_simulator=mock_components['keyboard'],
            root=root
        )

        controller._show_popup()
        root.update()

        # Simulate text change callback
        with patch.object(root, 'after') as mock_after:
            controller.on_text_change("hello", "wor")
            mock_after.assert_called()

    def test_display_update_sets_popup_text(self, root, mock_components):
        """_update_display should set text on popup."""
        from src.quickentry.QuickEntryController import QuickEntryController

        controller = QuickEntryController(
            subscriber=mock_components['subscriber'],
            focus_tracker=mock_components['focus_tracker'],
            keyboard_simulator=mock_components['keyboard'],
            root=root
        )

        controller._show_popup()
        root.update()

        controller._update_display("hello world")
        root.update()

        assert controller._popup.get_text() == "hello world"

    def test_is_showing_property(self, root, mock_components):
        """is_showing property should reflect popup state."""
        from src.quickentry.QuickEntryController import QuickEntryController

        controller = QuickEntryController(
            subscriber=mock_components['subscriber'],
            focus_tracker=mock_components['focus_tracker'],
            keyboard_simulator=mock_components['keyboard'],
            root=root
        )

        assert controller.is_showing is False

        controller._show_popup()
        root.update()
        assert controller.is_showing is True

        controller._on_cancel()
        root.update()
        assert controller.is_showing is False
