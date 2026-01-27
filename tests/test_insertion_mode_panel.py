import unittest
from unittest.mock import Mock

try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


@unittest.skipUnless(TKINTER_AVAILABLE, "tkinter not available")
class TestInsertionModePanel(unittest.TestCase):
    """Test InsertionModePanel GUI widget."""

    def setUp(self):
        try:
            from src.gui.InsertionModePanel import InsertionModePanel
            self.InsertionModePanel = InsertionModePanel
        except ImportError:
            self.skipTest("InsertionModePanel not implemented yet")

        try:
            self.root = tk.Tk()
            self.root.withdraw()
        except Exception as e:
            self.skipTest(f"tkinter initialization failed: {e}")

        self.parent = tk.Frame(self.root)

        self.mock_controller = Mock()
        self.mock_controller.is_enabled = Mock(return_value=False)

    def tearDown(self):
        if hasattr(self, 'root'):
            try:
                self.root.destroy()
            except:
                pass

    def test_button_initial_text_shows_off(self):
        panel = self.InsertionModePanel(self.parent, self.mock_controller)

        self.assertIn('OFF', panel.button['text'])

    def test_button_click_calls_controller_toggle(self):
        panel = self.InsertionModePanel(self.parent, self.mock_controller)

        panel._on_button_click()

        self.mock_controller.toggle.assert_called_once()

    def test_button_shows_on_when_enabled(self):
        self.mock_controller.is_enabled.return_value = True
        panel = self.InsertionModePanel(self.parent, self.mock_controller)

        panel._update_button_state()

        self.assertIn('ON', panel.button['text'])

    def test_button_shows_off_when_disabled(self):
        self.mock_controller.is_enabled.return_value = False
        panel = self.InsertionModePanel(self.parent, self.mock_controller)

        panel._update_button_state()

        self.assertIn('OFF', panel.button['text'])

    def test_button_updates_after_toggle(self):
        panel = self.InsertionModePanel(self.parent, self.mock_controller)

        self.assertIn('OFF', panel.button['text'])

        # Simulate toggle enabling
        self.mock_controller.is_enabled.return_value = True
        panel._on_button_click()

        # Should now show ON
        self.assertIn('ON', panel.button['text'])


if __name__ == '__main__':
    unittest.main()
