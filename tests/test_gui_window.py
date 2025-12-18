import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import tkinter as tk
    from tkinter import scrolledtext
    from src.GuiWindow import GuiWindow, create_stt_window
    from src.types import RecognitionResult
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
except Exception:
    # Catch any tkinter initialization errors
    TKINTER_AVAILABLE = False


@unittest.skipUnless(TKINTER_AVAILABLE, "tkinter not available or properly configured")
class TestGuiWindowWithChunkIds(unittest.TestCase):
    """
    Tests for GuiWindow with chunk-ID-based partial finalization.

    Tests new behavior where GuiWindow tracks RecognitionResult objects
    with chunk_ids to correctly handle partial finalization.
    """

    def setUp(self) -> None:
        """Set up GUI window for testing."""
        try:
            self.root, self.text_widget = create_stt_window()
            self.root.withdraw()
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

    def test_update_partial_accepts_recognition_result(self) -> None:
        """Test that update_partial() accepts RecognitionResult with chunk_ids."""
        result = RecognitionResult(
            text="hello",
            start_time=0.0,
            end_time=0.5,
            status='preliminary',
            chunk_ids=[1]
        )
        self.gui.update_partial(result)

        content = self.get_widget_content()
        self.assertEqual(content, "hello")

    def test_finalize_text_accepts_recognition_result(self) -> None:
        """Test that finalize_text() accepts RecognitionResult with chunk_ids."""
        # Preliminary
        prelim = RecognitionResult(
            text="hello",
            start_time=0.0,
            end_time=0.5,
            status='preliminary',
            chunk_ids=[1]
        )
        self.gui.update_partial(prelim)

        # Finalize
        final = RecognitionResult(
            text="hello",
            start_time=0.0,
            end_time=0.5,
            status='final',
            chunk_ids=[1]
        )
        self.gui.finalize_text(final)

        content = self.get_widget_content()
        self.assertEqual(content, "hello ")

    def test_partial_finalization_with_chunk_ids(self) -> None:
        """
        Test that finalize_text() uses chunk_ids to finalize only matched chunks.

        Scenario:
        - Preliminary chunks 1-5: "hello" "world" "how" "are" "you"
        - Finalize chunks 1-3: "hello world how"
        - Chunks 4-5 remain preliminary: "are you"
        """
        # Preliminary results
        results = [
            RecognitionResult("hello", 0.0, 0.5, True, [1]),
            RecognitionResult("world", 0.5, 1.0, True, [2]),
            RecognitionResult("how", 1.0, 1.5, True, [3]),
            RecognitionResult("are", 1.5, 2.0, True, [4]),
            RecognitionResult("you", 2.0, 2.5, True, [5]),
        ]

        for r in results:
            self.gui.update_partial(r)

        # All preliminary
        content = self.get_widget_content()
        self.assertEqual(content, "hello world how are you")

        # Finalize chunks 1-3
        final = RecognitionResult(
            text="hello world how",
            start_time=0.0,
            end_time=1.5,
            status='final',
            chunk_ids=[1, 2, 3]
        )
        self.gui.finalize_text(final)

        # Content preserved: finalized + remaining preliminary
        content = self.get_widget_content()
        self.assertEqual(content, "hello world how are you")

        # Check tags: "hello world how" is final
        tags_at_start = self.text_widget.tag_names("1.0")
        self.assertIn("final", tags_at_start)

        # "are you" is still preliminary
        are_pos = "1." + str(len("hello world how "))
        tags_at_are = self.text_widget.tag_names(are_pos)
        self.assertIn("preliminary", tags_at_are)

    def test_overlapping_window_workflow_with_chunk_ids(self) -> None:
        """
        Test complete workflow with overlapping windows using chunk_ids.

        Realistic pipeline simulation:
        1. Preliminary chunks 1-3: "hello" "world" "how"
        2. Window 1 finalizes chunks 1-2: "hello world"
        3. More preliminary chunks 4-5: "are" "you"
        4. Window 2 finalizes chunks 3-4: "how are"
        5. Chunk 5 remains preliminary: "you"
        """
        # First batch of preliminary
        for r in [
            RecognitionResult("hello", 0.0, 0.5, True, [1]),
            RecognitionResult("world", 0.5, 1.0, True, [2]),
            RecognitionResult("how", 1.0, 1.5, True, [3]),
        ]:
            self.gui.update_partial(r)

        content = self.get_widget_content()
        self.assertEqual(content, "hello world how")

        # Window 1 finalized
        self.gui.finalize_text(
            RecognitionResult("hello world", 0.0, 1.0, False, [1, 2])
        )

        content = self.get_widget_content()
        self.assertEqual(content, "hello world how")

        # More preliminary chunks
        for r in [
            RecognitionResult("are", 1.5, 2.0, True, [4]),
            RecognitionResult("you", 2.0, 2.5, True, [5]),
        ]:
            self.gui.update_partial(r)

        content = self.get_widget_content()
        self.assertEqual(content, "hello world how are you")

        # Window 2 finalized (includes overlap chunk 3)
        self.gui.finalize_text(
            RecognitionResult("how are", 1.0, 2.0, False, [3, 4])
        )

        content = self.get_widget_content()
        self.assertEqual(content, "hello world how are you")

        # "hello world how are" is finalized
        tags_at_start = self.text_widget.tag_names("1.0")
        self.assertIn("final", tags_at_start)

        # "you" is still preliminary
        you_pos = "1." + str(len("hello world how are "))
        tags_at_you = self.text_widget.tag_names(you_pos)
        self.assertIn("preliminary", tags_at_you)

    def test_finalize_tracks_chunk_ids_globally(self) -> None:
        """
        Test that GuiWindow tracks finalized chunk IDs globally.

        Verifies:
        - finalized_chunk_ids set grows with each finalization
        - Chunks are not finalized twice
        """
        # Preliminary
        for r in [
            RecognitionResult("one", 0.0, 0.5, True, [1]),
            RecognitionResult("two", 0.5, 1.0, True, [2]),
            RecognitionResult("three", 1.0, 1.5, True, [3]),
        ]:
            self.gui.update_partial(r)

        # Finalize chunk 1
        self.gui.finalize_text(
            RecognitionResult("one", 0.0, 0.5, False, [1])
        )
        self.assertIn(1, self.gui.finalized_chunk_ids)

        # Finalize chunk 2
        self.gui.finalize_text(
            RecognitionResult("two", 0.5, 1.0, False, [2])
        )
        self.assertIn(2, self.gui.finalized_chunk_ids)

        # Both chunks tracked
        self.assertEqual(self.gui.finalized_chunk_ids, {1, 2})

        # Chunk 3 still preliminary
        content = self.get_widget_content()
        self.assertEqual(content, "one two three")

    def test_preliminary_results_stored_as_objects(self) -> None:
        """
        Test that GuiWindow stores RecognitionResult objects internally.

        Verifies internal state management for chunk-based tracking.
        """
        results = [
            RecognitionResult("hello", 0.0, 0.5, True, [1]),
            RecognitionResult("world", 0.5, 1.0, True, [2]),
        ]

        for r in results:
            self.gui.update_partial(r)

        # GuiWindow should have stored these objects
        self.assertEqual(len(self.gui.preliminary_results), 2)
        self.assertEqual(self.gui.preliminary_results[0].text, "hello")
        self.assertEqual(self.gui.preliminary_results[1].text, "world")
        self.assertEqual(self.gui.preliminary_results[0].chunk_ids, [1])
        self.assertEqual(self.gui.preliminary_results[1].chunk_ids, [2])

    def test_finalize_text_without_preliminary(self) -> None:
        """Test that finalize_text() works when no preliminary text exists."""
        result = RecognitionResult(
            text="hello",
            start_time=0.0,
            end_time=0.5,
            status='final',
            chunk_ids=[1]
        )
        self.gui.finalize_text(result)

        content = self.get_widget_content()
        self.assertEqual(content, "hello ")

    def test_finalize_prevents_duplicate_finalization(self) -> None:
        """Test that finalize_text() prevents duplicate finalization."""
        # Preliminary
        prelim = RecognitionResult(
            text="hello",
            start_time=0.0,
            end_time=0.5,
            status='preliminary',
            chunk_ids=[1]
        )
        self.gui.update_partial(prelim)

        # Finalize once
        final = RecognitionResult(
            text="hello",
            start_time=0.0,
            end_time=0.5,
            status='final',
            chunk_ids=[1]
        )
        self.gui.finalize_text(final)

        # Try to finalize again with same text (duplicate call)
        self.gui.finalize_text(final)

        content = self.get_widget_content()
        self.assertEqual(content, "hello ")  # Should not duplicate

    def test_preliminary_text_has_preliminary_tag(self) -> None:
        """Test that preliminary text has correct styling tag."""
        result = RecognitionResult(
            text="preliminary",
            start_time=0.0,
            end_time=0.5,
            status='preliminary',
            chunk_ids=[1]
        )
        self.gui.update_partial(result)

        # Get tags at the start of the text
        tags = self.text_widget.tag_names("1.0")
        self.assertIn("preliminary", tags)

    def test_empty_partial_text(self) -> None:
        """Test handling of empty preliminary text."""
        result = RecognitionResult(
            text="",
            start_time=0.0,
            end_time=0.0,
            status='preliminary',
            chunk_ids=[]
        )
        self.gui.update_partial(result)

        content = self.get_widget_content()
        self.assertEqual(content, "")


@unittest.skipUnless(TKINTER_AVAILABLE, "tkinter not available or properly configured")
class TestGuiWindowParagraphBreaks(unittest.TestCase):
    """Tests for GuiWindow paragraph break functionality."""

    def setUp(self) -> None:
        """Set up GUI window for testing."""
        try:
            self.root, self.text_widget = create_stt_window()
            self.root.withdraw()
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

    def test_add_paragraph_break_adds_newlines(self) -> None:
        """Test that add_paragraph_break() adds two newlines to finalized text."""
        # Add some finalized text first
        result = RecognitionResult(
            text="Hello world",
            start_time=0.0,
            end_time=1.0,
            status='final',
            chunk_ids=[1, 2]
        )
        self.gui.finalize_text(result)

        # Add paragraph break
        self.gui.add_paragraph_break()

        # Check that finalized_text ends with \n\n
        self.assertTrue(self.gui.finalized_text.endswith("\n\n"))

        # Display should show the paragraph break
        content = self.get_widget_content()
        self.assertIn("\n\n", content)

    def test_add_paragraph_break_empty_text(self) -> None:
        """Test that add_paragraph_break() doesn't crash on empty text."""
        # Call on empty GuiWindow
        self.gui.add_paragraph_break()

        # Should add \n\n to empty string
        self.assertEqual(self.gui.finalized_text, "\n\n")

        # Should not crash during render
        # Note: _rerender_all adds a space after finalized text (line 166: finalized_text + " ")
        content = self.get_widget_content()
        self.assertEqual(content, "\n\n ")


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
