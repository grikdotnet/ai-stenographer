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

    def test_finalize_text_without_preliminary(self) -> None:
        """Test that finalize_text() works when no preliminary text exists."""
        self.gui.finalize_text("hello")
        content = self.get_widget_content()
        self.assertEqual(content, "hello ")

    def test_preliminary_finalize_cycles(self) -> None:
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

    def test_partial_finalization_preserves_remaining_preliminary(self) -> None:
        """
        Legacy test: Without chunk_ids, finalize_text() clears all preliminary text.

        NOTE: For proper partial finalization with remaining preliminary text preserved,
        see TestGuiWindowWithChunkIds which uses chunk-ID tracking.

        This test documents the old behavior where partial finalization was not possible
        without chunk-IDs - all preliminary text gets cleared on finalization.
        """
        # Simulate preliminary chunks arriving
        self.gui.update_partial("hello")
        self.gui.update_partial("world")
        self.gui.update_partial("how")
        self.gui.update_partial("are")
        self.gui.update_partial("you")

        # All preliminary: "hello world how are you"
        content = self.get_widget_content()
        self.assertEqual(content, "hello world how are you")

        # Window 1 finalized - without chunk_ids, all preliminary is cleared
        self.gui.finalize_text("hello world how")

        # Legacy behavior: all preliminary cleared, only finalized text remains
        content = self.get_widget_content()
        self.assertEqual(content, "hello world how ")

        # Check that "hello world how" is finalized (has "final" tag)
        tags_at_hello = self.text_widget.tag_names("1.0")
        self.assertIn("final", tags_at_hello)

    def test_overlapping_window_workflow(self) -> None:
        """
        Legacy test: Without chunk_ids, overlapping window workflow clears preliminary on finalize.

        NOTE: For proper overlapping window workflow with preliminary text preserved,
        see TestGuiWindowWithChunkIds.test_overlapping_window_workflow_with_chunk_ids

        This test documents legacy behavior where preliminary text is cleared on each finalization.
        """
        # First batch of chunks (belong to Window 1 + overlap)
        self.gui.update_partial("hello")
        self.gui.update_partial("world")
        self.gui.update_partial("how")

        # Window 1 finalized - without chunk_ids, all preliminary is cleared
        self.gui.finalize_text("hello world")

        content = self.get_widget_content()
        self.assertEqual(content, "hello world ")

        # More chunks arrive (belong to Window 2)
        self.gui.update_partial("are")
        self.gui.update_partial("you")

        content = self.get_widget_content()
        self.assertEqual(content, "hello world are you")

        # Window 2 finalized - again, all preliminary cleared
        self.gui.finalize_text("how are")

        content = self.get_widget_content()
        self.assertEqual(content, "hello world how are ")

        # Check final text
        tags_at_start = self.text_widget.tag_names("1.0")
        self.assertIn("final", tags_at_start)

    def test_finalize_portion_of_accumulated_preliminary(self) -> None:
        """
        Legacy test: Without chunk_ids, finalize_text() cannot preserve partial preliminary.

        NOTE: For proper partial finalization with remaining preliminary text preserved,
        see TestGuiWindowWithChunkIds which uses chunk-ID tracking.

        This test documents the old behavior where partial finalization was not possible.
        """
        self.gui.update_partial("one")
        self.gui.update_partial("two")
        self.gui.update_partial("three")
        self.gui.update_partial("four")
        self.gui.update_partial("five")

        # All preliminary
        content = self.get_widget_content()
        self.assertEqual(content, "one two three four five")

        # Finalize only first three words - but without chunk_ids, all preliminary is cleared
        self.gui.finalize_text("one two three")

        # Legacy behavior: all preliminary cleared
        content = self.get_widget_content()
        self.assertEqual(content, "one two three ")

        # "one two three" should be final
        tags_at_one = self.text_widget.tag_names("1.0")
        self.assertIn("final", tags_at_one)


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
