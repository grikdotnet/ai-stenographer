"""
Tests for ControlPanel - Pause/Resume button widget.
"""
import unittest
import pytest
from unittest.mock import Mock, patch
try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

pytestmark = pytest.mark.gui


@unittest.skipUnless(TKINTER_AVAILABLE, "tkinter not available")
class TestControlPanel(unittest.TestCase):
    """Test ControlPanel GUI widget."""

    def setUp(self):
        # Import here to allow test discovery
        try:
            from src.client.tk.gui.ControlPanel import ControlPanel
            self.ControlPanel = ControlPanel
        except ImportError:
            self.skipTest("ControlPanel not implemented yet")

        try:
            self.root = tk.Tk()
            self.root.withdraw()
        except Exception as e:
            self.skipTest(f"tkinter initialization failed: {e}")

        # mock parent frame
        self.parent = tk.Frame(self.root)

        self.mock_app_state = Mock()
        self.mock_app_state.get_state = Mock(return_value='running')
        self.mock_app_state.register_gui_observer = Mock()

        self.mock_controller = Mock()

    def tearDown(self):
        """Clean up tkinter resources."""
        if hasattr(self, 'root'):
            try:
                self.root.destroy()
            except:
                pass

    def test_registers_as_gui_observer_on_init(self):
        """Test that ControlPanel registers itself as a GUI observer."""
        panel = self.ControlPanel(
            self.parent,
            self.mock_app_state,
            self.mock_controller
        )

        self.mock_app_state.register_gui_observer.assert_called_once()

    def test_button_enabled_when_state_running(self):
        """Test that button is enabled when state is 'running'."""
        panel = self.ControlPanel(
            self.parent,
            self.mock_app_state,
            self.mock_controller
        )

        panel._on_state_change('starting', 'running')

        # Button should be enabled
        self.assertEqual(str(panel.button['state']), 'normal')
        self.assertEqual(panel.button['text'], 'Pause')

    def test_button_enabled_when_state_paused(self):
        """Test that button is enabled when state is 'paused'."""
        panel = self.ControlPanel(
            self.parent,
            self.mock_app_state,
            self.mock_controller
        )

        panel._on_state_change('running', 'paused')

        # Button should be enabled
        self.assertEqual(str(panel.button['state']), 'normal')
        self.assertEqual(panel.button['text'], 'Resume')

    def test_button_disabled_when_state_starting(self):
        """Test that button is disabled when state is 'starting'."""
        self.mock_app_state.get_state.return_value = 'starting'
        panel = self.ControlPanel(
            self.parent,
            self.mock_app_state,
            self.mock_controller
        )

        panel._on_state_change('shutdown', 'starting')

        # Button should be disabled
        self.assertEqual(str(panel.button['state']), 'disabled')

    def test_button_disabled_when_state_shutdown(self):
        """Test that button is disabled when state is 'shutdown'."""
        panel = self.ControlPanel(
            self.parent,
            self.mock_app_state,
            self.mock_controller
        )

        panel._on_state_change('running', 'shutdown')

        # Button should be disabled
        self.assertEqual(str(panel.button['state']), 'disabled')

    def test_button_click_calls_controller_toggle(self):
        """Test that clicking button calls controller.toggle()."""
        panel = self.ControlPanel(
            self.parent,
            self.mock_app_state,
            self.mock_controller
        )

        panel._on_button_click()

        # Controller toggle should be called
        self.mock_controller.toggle.assert_called_once()

    def test_button_disabled_during_click_to_prevent_rapid_clicks(self):
        """Test that button is disabled immediately on click to prevent rapid clicking."""
        panel = self.ControlPanel(
            self.parent,
            self.mock_app_state,
            self.mock_controller
        )

        panel._on_state_change('starting', 'running')
        self.assertEqual(str(panel.button['state']), 'normal')

        panel._on_button_click()

        # Button should be disabled immediately
        self.assertEqual(str(panel.button['state']), 'disabled')

    def test_button_text_updates_correctly(self):
        """Test that button text updates based on state transitions."""
        panel = self.ControlPanel(
            self.parent,
            self.mock_app_state,
            self.mock_controller
        )

        panel._on_state_change('running', 'paused')
        self.assertEqual(panel.button['text'], 'Resume')

        panel._on_state_change('paused', 'running')
        self.assertEqual(panel.button['text'], 'Pause')


if __name__ == '__main__':
    unittest.main()
