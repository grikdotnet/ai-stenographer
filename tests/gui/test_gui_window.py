import unittest
import sys
import os
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import tkinter as tk
    from tkinter import scrolledtext
    from src.gui.TextFormatter import TextFormatter
    from src.gui.TextDisplayWidget import TextDisplayWidget
    from src.gui.ApplicationWindow import ApplicationWindow
    from src.types import RecognitionResult
    from src.ApplicationState import ApplicationState
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
except Exception:
    # Catch any tkinter initialization errors
    TKINTER_AVAILABLE = False

pytestmark = pytest.mark.gui


@unittest.skipUnless(TKINTER_AVAILABLE, "tkinter not available or properly configured")
class TestGuiWindowWithChunkIds(unittest.TestCase):
    """
    Tests for TextFormatter + TextDisplayWidget with chunk-ID-based partial finalization.

    Tests MVC architecture where TextFormatter (controller) calculates formatting logic
    and TextDisplayWidget (view) renders to tkinter.
    """

    def setUp(self) -> None:
        """Set up TextFormatter + TextDisplayWidget for testing."""
        try:
            # Create minimal config and ApplicationState for testing
            test_config = {'audio': {}}  # Minimal config needed by ApplicationState
            test_app_state = ApplicationState(config=test_config)
            self.app_window = ApplicationWindow(test_app_state, test_config)
            self.root = self.app_window.get_root()

            # Get MVC components from ApplicationWindow
            self.formatter = self.app_window.get_formatter()
            self.display = self.app_window.get_display()
            self.text_widget = self.display.text_widget  # Access internal widget for test assertions

            self.root.withdraw()
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
        result = RecognitionResult(text="hello", start_time=0.0, end_time=0.5, chunk_ids=[1])
        self.formatter.partial_update(result)

        content = self.get_widget_content()
        self.assertEqual(content, "hello")

    def test_finalize_text_accepts_recognition_result(self) -> None:
        """Test that finalize_text() accepts RecognitionResult with chunk_ids."""
        # Preliminary
        prelim = RecognitionResult(text="hello", start_time=0.0, end_time=0.5, chunk_ids=[1])
        self.formatter.partial_update(prelim)

        # Finalize
        final = RecognitionResult(text="hello", start_time=0.0, end_time=0.5, chunk_ids=[1])
        self.formatter.finalization(final)

        content = self.get_widget_content()
        self.assertEqual(content, "hello ")

    def test_incremental_partial_then_finalize(self) -> None:
        """
        Test incremental architecture: partial_update replaces, finalization appends.

        Scenario:
        - Partial update with growing text: "hello" → "hello world how are you"
        - Finalize "hello world how"
        - New partial: "are you"
        """
        # Incremental partial updates (each replaces previous)
        self.formatter.partial_update(
            RecognitionResult(text="hello world how are you", start_time=0.0, end_time=2.5, chunk_ids=[1, 2, 3, 4, 5])
        )

        content = self.get_widget_content()
        self.assertEqual(content, "hello world how are you")

        # Finalize stable prefix
        self.formatter.finalization(
            RecognitionResult(text="hello world how", start_time=0.0, end_time=1.5, chunk_ids=[1, 2, 3])
        )

        content = self.get_widget_content()
        self.assertEqual(content, "hello world how ")

        # Check tags: finalized text has "final" tag
        tags_at_start = self.text_widget.tag_names("1.0")
        self.assertIn("final", tags_at_start)

    def test_progressive_finalization_workflow(self) -> None:
        """
        Test progressive finalization with incremental architecture.

        Realistic pipeline simulation:
        1. Partial update: "hello world how" (growing window)
        2. Finalize: "hello world"
        3. Partial update: "how are you" (new unstable tail)
        4. Finalize: "how are"
        """
        # Growing window shows preliminary text
        self.formatter.partial_update(
            RecognitionResult(text="hello world how", start_time=0.0, end_time=1.5, chunk_ids=[1, 2, 3])
        )
        content = self.get_widget_content()
        self.assertEqual(content, "hello world how")

        # First finalization
        self.formatter.finalization(
            RecognitionResult(text="hello world", start_time=0.0, end_time=1.0, chunk_ids=[1, 2])
        )
        content = self.get_widget_content()
        self.assertEqual(content, "hello world ")

        # New partial update replaces preliminary
        self.formatter.partial_update(
            RecognitionResult(text="how are you", start_time=1.0, end_time=2.5, chunk_ids=[3, 4, 5])
        )
        content = self.get_widget_content()
        self.assertEqual(content, "hello world how are you")

        # Second finalization
        self.formatter.finalization(
            RecognitionResult(text="how are", start_time=1.0, end_time=2.0, chunk_ids=[3, 4])
        )
        content = self.get_widget_content()
        self.assertEqual(content, "hello world how are ")

        tags_at_start = self.text_widget.tag_names("1.0")
        self.assertIn("final", tags_at_start)

    def test_finalized_text_accumulates(self) -> None:
        """
        Test that finalized_text buffer accumulates across multiple finalizations.

        Verifies:
        - Each finalization appends to finalized_text
        - Duplicate finalization is prevented
        """
        # Progressive finalization
        self.formatter.finalization(
            RecognitionResult(text="one", start_time=0.0, end_time=0.5, chunk_ids=[1])
        )
        self.assertEqual(self.formatter.finalized_text, "one")

        self.formatter.finalization(
            RecognitionResult(text="two", start_time=0.5, end_time=1.0, chunk_ids=[2])
        )
        self.assertEqual(self.formatter.finalized_text, "one two")

        # Duplicate prevention
        self.formatter.finalization(
            RecognitionResult(text="two", start_time=0.5, end_time=1.0, chunk_ids=[2])
        )
        self.assertEqual(self.formatter.finalized_text, "one two")

        content = self.get_widget_content()
        self.assertEqual(content, "one two ")

    def test_current_preliminary_replaced_on_partial_update(self) -> None:
        """
        Test that partial_update replaces current_preliminary (not appends).

        Verifies the incremental architecture: each partial_update replaces the previous one.
        """
        result1 = RecognitionResult(text="hello", start_time=0.0, end_time=0.5, chunk_ids=[1])
        result2 = RecognitionResult(text="hello world", start_time=0.0, end_time=1.0, chunk_ids=[1, 2])

        self.formatter.partial_update(result1)
        self.assertEqual(self.formatter.current_preliminary.text, "hello")

        self.formatter.partial_update(result2)
        self.assertEqual(self.formatter.current_preliminary.text, "hello world")

    def test_finalize_text_without_preliminary(self) -> None:
        """Test that finalize_text() works when no preliminary text exists."""
        result = RecognitionResult(text="hello", start_time=0.0, end_time=0.5, chunk_ids=[1])
        self.formatter.finalization(result)

        content = self.get_widget_content()
        self.assertEqual(content, "hello ")

    def test_finalize_prevents_duplicate_finalization(self) -> None:
        """Test that finalize_text() prevents duplicate finalization."""
        # Preliminary
        prelim = RecognitionResult(text="hello", start_time=0.0, end_time=0.5, chunk_ids=[1])
        self.formatter.partial_update(prelim)

        # Finalize once
        final = RecognitionResult(text="hello", start_time=0.0, end_time=0.5, chunk_ids=[1])
        self.formatter.finalization(final)

        # Try to finalize again with same text (duplicate call)
        self.formatter.finalization(final)

        content = self.get_widget_content()
        self.assertEqual(content, "hello ")  # Should not duplicate

    def test_preliminary_text_has_preliminary_tag(self) -> None:
        """Test that preliminary text has correct styling tag."""
        result = RecognitionResult(text="preliminary", start_time=0.0, end_time=0.5, chunk_ids=[1])
        self.formatter.partial_update(result)

        # Get tags at the start of the text
        tags = self.text_widget.tag_names("1.0")
        self.assertIn("preliminary", tags)

    def test_empty_partial_text(self) -> None:
        """Test handling of empty preliminary text."""
        result = RecognitionResult(text="", start_time=0.0, end_time=0.0, chunk_ids=[])
        self.formatter.partial_update(result)

        content = self.get_widget_content()
        self.assertEqual(content, "")


@unittest.skipUnless(TKINTER_AVAILABLE, "tkinter not available or properly configured")
class TestGuiWindowAudioParagraphBreaks(unittest.TestCase):
    """
    Tests for automatic paragraph breaks based on audio timing pauses.

    Tests the feature where TextFormatter automatically inserts paragraph breaks
    when there's a pause >= PARAGRAPH_PAUSE_THRESHOLD (2.0s) between a finalized
    segment's end and the next preliminary segment's start.
    """

    def setUp(self) -> None:
        """Set up TextFormatter + TextDisplayWidget for testing."""
        try:
            # Create minimal config and ApplicationState for testing
            test_config = {'audio': {}}  # Minimal config needed by ApplicationState
            test_app_state = ApplicationState(config=test_config)
            self.app_window = ApplicationWindow(test_app_state, test_config)
            self.root = self.app_window.get_root()

            # Get MVC components from ApplicationWindow
            self.formatter = self.app_window.get_formatter()
            self.display = self.app_window.get_display()
            self.text_widget = self.display.text_widget  # Access internal widget for test assertions

            self.root.withdraw()
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

    def test_pause_detection_inserts_paragraph_break(self) -> None:
        """
        Test that a pause >= 2.5s between finalized and preliminary inserts paragraph break.

        Scenario:
        - Finalize "hello world" (end_time=1.0)
        - Pause for 3 seconds
        - Preliminary "how" arrives (start_time=4.0)
        - Expected: paragraph break inserted automatically
        """
        # Finalize "hello world"
        final_result = RecognitionResult(text="hello world", start_time=0.0, end_time=1.0, chunk_ids=[1, 2])
        self.formatter.finalization(final_result)

        # Verify initial state
        self.assertEqual(self.formatter.finalized_text, "hello world")
        self.assertEqual(self.formatter.last_finalized_end_time, 1.0)

        # Preliminary after 3-second pause
        prelim_result = RecognitionResult(text="how", start_time=4.0, end_time=4.5, chunk_ids=[3])
        self.formatter.partial_update(prelim_result)

        # Paragraph break should be inserted (single newline)
        self.assertTrue(self.formatter.finalized_text.endswith("\n"))
        self.assertEqual(self.formatter.finalized_text, "hello world\n")

        # Widget should show the paragraph break
        content = self.get_widget_content()
        self.assertIn("\n", content)
        self.assertIn("how", content)

    def test_short_pause_no_paragraph_break(self) -> None:
        """
        Test that a pause < 2.5s does NOT insert paragraph break.

        Scenario:
        - Finalize "hello" (end_time=1.0)
        - Pause for 1 second (< 2.5s threshold)
        - Preliminary "world" arrives (start_time=2.0)
        - Expected: NO paragraph break
        """
        # Finalize "hello"
        final_result = RecognitionResult(text="hello", start_time=0.0, end_time=1.0, chunk_ids=[1])
        self.formatter.finalization(final_result)

        # Preliminary after 1-second pause (too short)
        prelim_result = RecognitionResult(text="world", start_time=2.0, end_time=2.5, chunk_ids=[2])
        self.formatter.partial_update(prelim_result)

        # No paragraph break should be inserted
        self.assertFalse(self.formatter.finalized_text.endswith("\n"))
        self.assertEqual(self.formatter.finalized_text, "hello")

        # Widget should show text without paragraph break
        content = self.get_widget_content()
        self.assertNotIn("\n", content)
        self.assertIn("hello", content)
        self.assertIn("world", content)

    def test_no_duplicate_paragraph_breaks(self) -> None:
        """
        Test that multiple preliminaries after a pause don't create duplicate breaks.

        Scenario:
        - Finalize "hello" (end_time=1.0)
        - Preliminary "how" arrives (start_time=4.0) → inserts break
        - Preliminary "are" arrives (start_time=4.5) → should NOT insert duplicate
        - Expected: Only one paragraph break
        """
        # Finalize "hello"
        final_result = RecognitionResult(text="hello", start_time=0.0, end_time=1.0, chunk_ids=[1])
        self.formatter.finalization(final_result)

        # First preliminary after pause (inserts break)
        prelim1 = RecognitionResult(text="how", start_time=4.0, end_time=4.5, chunk_ids=[2])
        self.formatter.partial_update(prelim1)

        # Verify paragraph break inserted
        self.assertEqual(self.formatter.finalized_text, "hello\n")

        # Second preliminary after same pause (should NOT insert duplicate)
        prelim2 = RecognitionResult(text="how are", start_time=4.0, end_time=5.0, chunk_ids=[2, 3])
        self.formatter.partial_update(prelim2)

        # Should still have only one paragraph break
        self.assertEqual(self.formatter.finalized_text, "hello\n")
        self.assertEqual(self.formatter.finalized_text.count("\n"), 1)

        # Widget should show current preliminary (replaces previous)
        content = self.get_widget_content()
        self.assertIn("how are", content)

    def test_no_pause_detection_without_finalization(self) -> None:
        """
        Test that pause detection doesn't trigger without prior finalization.

        Scenario:
        - No finalized text yet (last_finalized_end_time = 0)
        - Preliminary arrives
        - Expected: NO paragraph break (check skipped)
        """
        # Verify initial state
        self.assertEqual(self.formatter.last_finalized_end_time, 0.0)
        self.assertEqual(self.formatter.finalized_text, "")

        # Preliminary without any prior finalization
        prelim_result = RecognitionResult(text="hello", start_time=5.0, end_time=5.5, chunk_ids=[1])
        self.formatter.partial_update(prelim_result)

        # No paragraph break should be inserted
        self.assertEqual(self.formatter.finalized_text, "")

        # Widget should just show the preliminary text
        content = self.get_widget_content()
        self.assertEqual(content, "hello")
        self.assertNotIn("\n", content)

    def test_multiple_pauses_multiple_breaks(self) -> None:
        """
        Test that multiple pauses across different finalization cycles each insert breaks.

        Scenario:
        - Finalize "one" (end_time=1.0)
        - Preliminary "two" (start_time=4.0) → first break
        - Finalize "two three" (end_time=7.0)
        - Preliminary "four" (start_time=10.0) → second break
        - Expected: Two paragraph breaks in finalized text
        """
        # First finalization
        final1 = RecognitionResult(text="one", start_time=0.0, end_time=1.0, chunk_ids=[1])
        self.formatter.finalization(final1)
        self.assertEqual(self.formatter.finalized_text, "one")

        # First pause - preliminary triggers first break
        prelim1 = RecognitionResult(text="two", start_time=4.0, end_time=5.0, chunk_ids=[2])
        self.formatter.partial_update(prelim1)
        self.assertEqual(self.formatter.finalized_text, "one\n")

        # Second finalization (includes "two" from preliminary)
        final2 = RecognitionResult(text="two three", start_time=4.0, end_time=7.0, chunk_ids=[2, 3])
        self.formatter.finalization(final2)
        # finalized_text should be "one\n" + " " + "two three"
        self.assertEqual(self.formatter.finalized_text, "one\n two three")
        self.assertEqual(self.formatter.last_finalized_end_time, 7.0)

        # Second pause - preliminary triggers second break
        prelim2 = RecognitionResult(text="four", start_time=10.0, end_time=10.5, chunk_ids=[4])
        self.formatter.partial_update(prelim2)

        # Should have two paragraph breaks now
        self.assertTrue(self.formatter.finalized_text.endswith("\n"))
        self.assertEqual(self.formatter.finalized_text, "one\n two three\n")

        # Widget should show all text with both breaks
        content = self.get_widget_content()
        self.assertIn("one", content)
        self.assertIn("two three", content)
        self.assertIn("four", content)

    def test_paragraph_break_with_empty_finalized_text(self) -> None:
        """
        Test edge case: pause detection with empty finalized_text doesn't crash.

        Scenario:
        - finalized_text is empty (edge case)
        - Preliminary arrives with large pause
        - Expected: No crash, no paragraph break (guard: self.finalized_text and ...)
        """
        # Artificially set last_finalized_end_time without finalized text (edge case)
        self.formatter.last_finalized_end_time = 1.0
        self.assertEqual(self.formatter.finalized_text, "")

        # Preliminary with pause
        prelim_result = RecognitionResult(text="hello", start_time=4.0, end_time=4.5, chunk_ids=[1])

        # Should not crash
        self.formatter.partial_update(prelim_result)

        # No paragraph break inserted (empty finalized_text guard)
        self.assertEqual(self.formatter.finalized_text, "")

        # Widget should show preliminary text only
        content = self.get_widget_content()
        self.assertEqual(content, "hello")


if __name__ == '__main__':
    unittest.main()
