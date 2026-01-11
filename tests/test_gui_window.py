import unittest
import sys
import os

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
        result = RecognitionResult(
            text="hello",
            start_time=0.0,
            end_time=0.5,
            status='preliminary',
            chunk_ids=[1]
        )
        self.formatter.partial_update(result)

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
        self.formatter.partial_update(prelim)

        # Finalize
        final = RecognitionResult(
            text="hello",
            start_time=0.0,
            end_time=0.5,
            status='final',
            chunk_ids=[1]
        )
        self.formatter.finalization(final)

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
            self.formatter.partial_update(r)

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
        self.formatter.finalization(final)

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
            self.formatter.partial_update(r)

        content = self.get_widget_content()
        self.assertEqual(content, "hello world how")

        # Window 1 finalized
        self.formatter.finalization(
            RecognitionResult("hello world", 0.0, 1.0, False, [1, 2])
        )

        content = self.get_widget_content()
        self.assertEqual(content, "hello world how")

        # More preliminary chunks
        for r in [
            RecognitionResult("are", 1.5, 2.0, True, [4]),
            RecognitionResult("you", 2.0, 2.5, True, [5]),
        ]:
            self.formatter.partial_update(r)

        content = self.get_widget_content()
        self.assertEqual(content, "hello world how are you")

        # Window 2 finalized (includes overlap chunk 3)
        self.formatter.finalization(
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
            self.formatter.partial_update(r)

        # Finalize chunk 1
        self.formatter.finalization(
            RecognitionResult("one", 0.0, 0.5, False, [1])
        )
        self.assertIn(1, self.formatter.finalized_chunk_ids)

        # Finalize chunk 2
        self.formatter.finalization(
            RecognitionResult("two", 0.5, 1.0, False, [2])
        )
        self.assertIn(2, self.formatter.finalized_chunk_ids)

        # Both chunks tracked
        self.assertEqual(self.formatter.finalized_chunk_ids, {1, 2})

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
            self.formatter.partial_update(r)

        # GuiWindow should have stored these objects
        self.assertEqual(len(self.formatter.preliminary_results), 2)
        self.assertEqual(self.formatter.preliminary_results[0].text, "hello")
        self.assertEqual(self.formatter.preliminary_results[1].text, "world")
        self.assertEqual(self.formatter.preliminary_results[0].chunk_ids, [1])
        self.assertEqual(self.formatter.preliminary_results[1].chunk_ids, [2])

    def test_finalize_text_without_preliminary(self) -> None:
        """Test that finalize_text() works when no preliminary text exists."""
        result = RecognitionResult(
            text="hello",
            start_time=0.0,
            end_time=0.5,
            status='final',
            chunk_ids=[1]
        )
        self.formatter.finalization(result)

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
        self.formatter.partial_update(prelim)

        # Finalize once
        final = RecognitionResult(
            text="hello",
            start_time=0.0,
            end_time=0.5,
            status='final',
            chunk_ids=[1]
        )
        self.formatter.finalization(final)

        # Try to finalize again with same text (duplicate call)
        self.formatter.finalization(final)

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
        self.formatter.partial_update(result)

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
        final_result = RecognitionResult(
            text="hello world",
            start_time=0.0,
            end_time=1.0,
            status='final',
            chunk_ids=[1, 2]
        )
        self.formatter.finalization(final_result)

        # Verify initial state
        self.assertEqual(self.formatter.finalized_text, "hello world")
        self.assertEqual(self.formatter.last_finalized_end_time, 1.0)

        # Preliminary after 3-second pause
        prelim_result = RecognitionResult(
            text="how",
            start_time=4.0,  # 3.0s pause (4.0 - 1.0)
            end_time=4.5,
            status='preliminary',
            chunk_ids=[3]
        )
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
        final_result = RecognitionResult(
            text="hello",
            start_time=0.0,
            end_time=1.0,
            status='final',
            chunk_ids=[1]
        )
        self.formatter.finalization(final_result)

        # Preliminary after 1-second pause (too short)
        prelim_result = RecognitionResult(
            text="world",
            start_time=2.0,  # 1.0s pause (2.0 - 1.0) < 2.5s
            end_time=2.5,
            status='preliminary',
            chunk_ids=[2]
        )
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
        final_result = RecognitionResult(
            text="hello",
            start_time=0.0,
            end_time=1.0,
            status='final',
            chunk_ids=[1]
        )
        self.formatter.finalization(final_result)

        # First preliminary after pause (inserts break)
        prelim1 = RecognitionResult(
            text="how",
            start_time=4.0,  # 3.0s pause
            end_time=4.5,
            status='preliminary',
            chunk_ids=[2]
        )
        self.formatter.partial_update(prelim1)

        # Verify paragraph break inserted
        self.assertEqual(self.formatter.finalized_text, "hello\n")

        # Second preliminary after same pause (should NOT insert duplicate)
        prelim2 = RecognitionResult(
            text="are",
            start_time=4.5,  # Still 3.5s from finalized end (4.5 - 1.0)
            end_time=5.0,
            status='preliminary',
            chunk_ids=[3]
        )
        self.formatter.partial_update(prelim2)

        # Should still have only one paragraph break
        self.assertEqual(self.formatter.finalized_text, "hello\n")
        self.assertEqual(self.formatter.finalized_text.count("\n"), 1)

        # Widget should show both preliminaries without duplicate break
        content = self.get_widget_content()
        self.assertIn("how", content)
        self.assertIn("are", content)

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
        prelim_result = RecognitionResult(
            text="hello",
            start_time=5.0,  # Large start time doesn't matter
            end_time=5.5,
            status='preliminary',
            chunk_ids=[1]
        )
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
        final1 = RecognitionResult(
            text="one",
            start_time=0.0,
            end_time=1.0,
            status='final',
            chunk_ids=[1]
        )
        self.formatter.finalization(final1)
        self.assertEqual(self.formatter.finalized_text, "one")

        # First pause - preliminary triggers first break
        prelim1 = RecognitionResult(
            text="two",
            start_time=4.0,  # 3.0s pause
            end_time=5.0,
            status='preliminary',
            chunk_ids=[2]
        )
        self.formatter.partial_update(prelim1)
        self.assertEqual(self.formatter.finalized_text, "one\n")

        # Second finalization (includes "two" from preliminary)
        final2 = RecognitionResult(
            text="two three",
            start_time=4.0,
            end_time=7.0,
            status='final',
            chunk_ids=[2, 3]
        )
        self.formatter.finalization(final2)
        # finalized_text should be "one\n" + " " + "two three"
        self.assertEqual(self.formatter.finalized_text, "one\n two three")
        self.assertEqual(self.formatter.last_finalized_end_time, 7.0)

        # Second pause - preliminary triggers second break
        prelim2 = RecognitionResult(
            text="four",
            start_time=10.0,  # 3.0s pause (10.0 - 7.0)
            end_time=10.5,
            status='preliminary',
            chunk_ids=[4]
        )
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
        prelim_result = RecognitionResult(
            text="hello",
            start_time=4.0,  # 3.0s pause
            end_time=4.5,
            status='preliminary',
            chunk_ids=[1]
        )

        # Should not crash
        self.formatter.partial_update(prelim_result)

        # No paragraph break inserted (empty finalized_text guard)
        self.assertEqual(self.formatter.finalized_text, "")

        # Widget should show preliminary text only
        content = self.get_widget_content()
        self.assertEqual(content, "hello")


if __name__ == '__main__':
    unittest.main()
