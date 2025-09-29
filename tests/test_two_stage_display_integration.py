import unittest
import queue
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import tkinter as tk
    from tkinter import scrolledtext
    from src.TwoStageDisplayHandler import TwoStageDisplayHandler
    from src.GuiWindow import create_stt_window
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
except Exception:
    # Catch any tkinter initialization errors
    TKINTER_AVAILABLE = False


@unittest.skipUnless(TKINTER_AVAILABLE, "tkinter not available or properly configured")
class TestTwoStageDisplayIntegration(unittest.TestCase):
    """
    Integration test for TwoStageDisplayHandler using real tkinter GUI window.

    Tests actual text widget content after preliminary and final text operations
    to verify correct display behavior and prevent text duplication issues.
    """

    def setUp(self) -> None:
        """Set up real GUI window and display handler for testing."""
        try:
            self.root, self.text_widget = create_stt_window()
            self.root.withdraw()  # Hide window to avoid GUI popup during testing
            self.final_queue = queue.Queue()
            self.partial_queue = queue.Queue()
            self.display = TwoStageDisplayHandler(
                self.final_queue,
                self.partial_queue,
                self.text_widget,
                self.root
            )
            self.root.lift()
        except Exception as e:
            self.skipTest(f"Could not initialize tkinter GUI: {e}")

    def tearDown(self) -> None:
        """Clean up GUI window after test."""
        if hasattr(self, 'root'):
            try:
                self.root.after(1000)
                self.root.destroy()
            except:
                pass

    def get_widget_content(self) -> str:
        """Get the complete text content of the widget."""
        return self.text_widget.get("1.0", tk.END).rstrip('\n')

    def test_preliminary_text_replacement(self) -> None:
        """Test that preliminary text gets replaced correctly without duplication."""
        # Add preliminary text
        self.display.update_preliminary_text("hello")
        content = self.get_widget_content()
        self.assertEqual(content, "hello")

        # Replace with longer preliminary text
        self.display.update_preliminary_text("hello world")
        content = self.get_widget_content()
        self.assertEqual(content, "hello world")

        # Replace with shorter preliminary text
        self.display.update_preliminary_text("hi")
        content = self.get_widget_content()
        self.assertEqual(content, "hi")

    def test_finalization_replaces_preliminary(self) -> None:
        """Test that finalization correctly replaces preliminary text."""
        # Add preliminary text
        self.display.update_preliminary_text("hello world ho")

        # Finalize with different text
        self.display.finalize_text("hello world")
        content = self.get_widget_content()
        self.assertEqual(content, "hello world ")

    def test_multiple_preliminary_updates_before_finalization(self) -> None:
        """Test multiple preliminary updates followed by finalization."""
        # Multiple preliminary updates
        self.display.update_preliminary_text("h")
        self.display.update_preliminary_text("he")
        self.display.update_preliminary_text("hel")
        self.display.update_preliminary_text("hello")

        content = self.get_widget_content()
        self.assertEqual(content, "hello")

        # Finalize
        self.display.finalize_text("hello")
        content = self.get_widget_content()
        self.assertEqual(content, "hello ")

    def test_multiple_finalization_cycles(self) -> None:
        """Test multiple complete cycles of preliminary → final text."""
        # First cycle
        self.display.update_preliminary_text("hello")
        self.display.finalize_text("hello")

        # Second cycle
        self.display.update_preliminary_text("world")
        self.display.finalize_text("world")

        content = self.get_widget_content()
        self.assertEqual(content, "hello world ")

    def test_empty_preliminary_text(self) -> None:
        """Test handling of empty preliminary text."""
        self.display.update_preliminary_text("")
        content = self.get_widget_content()
        self.assertEqual(content, "")

        self.display.finalize_text("final")
        content = self.get_widget_content()
        self.assertEqual(content, "final ")

    def test_text_styling_applied(self) -> None:
        """Test that correct text styling is applied to preliminary and final text."""
        # Add preliminary text and check it has preliminary tag
        self.display.update_preliminary_text("preliminary")

        # Check that preliminary text has the correct tag
        preliminary_tags = self.text_widget.tag_names("1.0")
        self.assertIn("preliminary", preliminary_tags)

        # Finalize and check final tag
        self.display.finalize_text("final")

        # Check that finalized text has the correct tag
        final_tags = self.text_widget.tag_names("1.0")
        self.assertIn("final", final_tags)



if __name__ == '__main__':
    unittest.main()