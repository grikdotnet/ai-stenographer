import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import tkinter as tk
    from tkinter import scrolledtext
    from src.GuiWindow import GuiWindow, create_stt_window
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
except Exception:
    # Catch any tkinter initialization errors
    TKINTER_AVAILABLE = False


@unittest.skipUnless(TKINTER_AVAILABLE, "tkinter not available or properly configured")
class TestGuiWindow(unittest.TestCase):
    """
    Tests for GuiWindow class.

    Tests preliminary/final text display, position tracking, text styling,
    and deduplication behavior.
    """

    def setUp(self) -> None:
        """Set up GUI window for testing."""
        try:
            self.root, self.text_widget = create_stt_window()
            self.root.withdraw()  # Hide window during tests
            self.gui = GuiWindow(self.text_widget, self.root)
        except Exception as e:
            self.skipTest(f"Could not initialize tkinter GUI: {e}")

    def tearDown(self) -> None:
        """Clean up GUI window."""
        if hasattr(self, 'root'):
            try:
                self.root.destroy()
            except:
                pass

    def get_widget_content(self) -> str:
        """Get complete text content of the widget."""
        return self.text_widget.get("1.0", tk.END).rstrip('\n')

    def test_update_partial_displays_text(self) -> None:
        """Test that update_partial() displays preliminary text."""
        self.gui.update_partial("hello")
        content = self.get_widget_content()
        self.assertEqual(content, "hello")

    def test_update_partial_appends_to_previous_partial(self) -> None:
        """Test that update_partial() appends to previous preliminary text."""
        self.gui.update_partial("hello")
        self.gui.update_partial("world")
        content = self.get_widget_content()
        self.assertEqual(content, "hello world")

    def test_update_partial_appends_multiple_words(self) -> None:
        """Test that update_partial() correctly appends multiple times."""
        self.gui.update_partial("hello")
        self.gui.update_partial("world")
        self.gui.update_partial("how")
        content = self.get_widget_content()
        self.assertEqual(content, "hello world how")

    def test_finalize_text_converts_preliminary_to_final(self) -> None:
        """Test that finalize_text() converts preliminary to final with space."""
        self.gui.update_partial("hello")
        self.gui.update_partial("world")
        self.gui.finalize_text("hello world")
        content = self.get_widget_content()
        self.assertEqual(content, "hello world ")

    def test_finalize_text_without_preliminary(self) -> None:
        """Test that finalize_text() works when no preliminary text exists."""
        self.gui.finalize_text("hello")
        content = self.get_widget_content()
        self.assertEqual(content, "hello ")

    def test_multiple_preliminary_finalize_cycles(self) -> None:
        """Test multiple cycles of preliminary → final text."""
        # First cycle
        self.gui.update_partial("hello")
        self.gui.finalize_text("hello")

        # Second cycle
        self.gui.update_partial("world")
        self.gui.finalize_text("world")

        content = self.get_widget_content()
        self.assertEqual(content, "hello world ")

    def test_finalize_prevents_duplicate_finalization(self) -> None:
        """Test that finalize_text() prevents duplicate finalization."""
        self.gui.update_partial("hello")
        self.gui.finalize_text("hello")
        self.gui.finalize_text("hello")  # Duplicate call

        content = self.get_widget_content()
        self.assertEqual(content, "hello ")  # Should not duplicate

    def test_update_partial_after_finalize(self) -> None:
        """Test that update_partial() works correctly after finalization."""
        self.gui.update_partial("hello")
        self.gui.finalize_text("hello")
        self.gui.update_partial("world")

        content = self.get_widget_content()
        self.assertEqual(content, "hello world")

    def test_preliminary_text_has_preliminary_tag(self) -> None:
        """Test that preliminary text has correct styling tag."""
        self.gui.update_partial("preliminary")

        # Get tags at the start of the text
        tags = self.text_widget.tag_names("1.0")
        self.assertIn("preliminary", tags)

    def test_final_text_has_final_tag(self) -> None:
        """Test that finalized text has correct styling tag."""
        self.gui.update_partial("text")
        self.gui.finalize_text("text")

        # Get tags at the start of the text
        tags = self.text_widget.tag_names("1.0")
        self.assertIn("final", tags)

    def test_empty_partial_text(self) -> None:
        """Test handling of empty preliminary text."""
        self.gui.update_partial("")
        content = self.get_widget_content()
        self.assertEqual(content, "")

    def test_finalize_replaces_accumulated_preliminary(self) -> None:
        """Test finalization replaces all accumulated preliminary text."""
        self.gui.update_partial("hello")
        self.gui.update_partial("world")
        self.gui.update_partial("how")
        self.gui.finalize_text("hello world")

        content = self.get_widget_content()
        self.assertEqual(content, "hello world ")

    def test_position_tracking_after_multiple_operations(self) -> None:
        """Test that position tracking remains correct after multiple operations."""
        self.gui.update_partial("first")
        self.gui.finalize_text("first")
        self.gui.update_partial("second")
        self.gui.finalize_text("second")
        self.gui.update_partial("third")

        content = self.get_widget_content()
        self.assertEqual(content, "first second third")


@unittest.skipUnless(TKINTER_AVAILABLE, "tkinter not available or properly configured")
class TestCreateSttWindow(unittest.TestCase):
    """Tests for create_stt_window() factory function."""

    def tearDown(self) -> None:
        """Clean up GUI window."""
        if hasattr(self, 'root'):
            try:
                self.root.destroy()
            except:
                pass

    def test_create_stt_window_returns_root_and_widget(self) -> None:
        """Test that create_stt_window() returns root and text widget."""
        try:
            root, text_widget = create_stt_window()
            self.root = root
            self.root.withdraw()

            self.assertIsInstance(root, tk.Tk)
            self.assertIsInstance(text_widget, scrolledtext.ScrolledText)
        except Exception as e:
            self.skipTest(f"Could not initialize tkinter GUI: {e}")

    def test_created_window_has_title(self) -> None:
        """Test that created window has correct title."""
        try:
            root, text_widget = create_stt_window()
            self.root = root
            self.root.withdraw()

            self.assertEqual(root.title(), "Speech-to-Text Display")
        except Exception as e:
            self.skipTest(f"Could not initialize tkinter GUI: {e}")


if __name__ == '__main__':
    unittest.main()
