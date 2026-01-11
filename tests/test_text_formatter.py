"""Unit tests for TextFormatter.

Tests verify:
- Partial update logic (spacing, state tracking)
- Finalization logic (chunk tracking, deduplication)
- Paragraph break detection (pause threshold)
- State inspection (remaining preliminary results)

All tests use a MockDisplay to capture DisplayInstructions without tkinter.
"""

import unittest
from src.gui.TextFormatter import TextFormatter
from src.types import RecognitionResult, DisplayInstructions


class MockDisplay:

    def __init__(self):
        self.instructions_history = []

    def apply_instructions(self, instructions: DisplayInstructions) -> None:
        """Capture instructions."""
        self.instructions_history.append(instructions)

    def get_last_instructions(self) -> DisplayInstructions:
        return self.instructions_history[-1] if self.instructions_history else None

    def clear_history(self):
        self.instructions_history = []


class TestTextFormatterBasicPartialUpdates(unittest.TestCase):
    """Group A: Basic Partial Updates"""

    def setUp(self):
        self.display = MockDisplay()
        self.formatter = TextFormatter(display=self.display, verbose=False)

    def test_first_partial_no_space(self):
        """First partial update should NOT prepend space."""
        result = RecognitionResult(
            text="Hello",
            start_time=1.0,
            end_time=1.5,
            status='preliminary',
            chunk_ids=[1]
        )

        self.formatter.partial_update(result)

        instructions = self.display.get_last_instructions()
        self.assertEqual(instructions.action, 'update_partial')
        self.assertEqual(instructions.text_to_append, "Hello")
        self.assertFalse(instructions.text_to_append.startswith(" "))

    def test_second_partial_adds_space(self):
        """Second partial update should prepend space."""
        result1 = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='preliminary', chunk_ids=[1])
        result2 = RecognitionResult(text="world", start_time=1.6, end_time=2.0, status='preliminary', chunk_ids=[2])

        self.formatter.partial_update(result1)
        self.formatter.partial_update(result2)

        instructions = self.display.get_last_instructions()
        self.assertEqual(instructions.action, 'update_partial')
        self.assertEqual(instructions.text_to_append, " world")
        self.assertTrue(instructions.text_to_append.startswith(" "))

    def test_partial_stores_result_objects(self):
        """Partial updates should store RecognitionResult objects."""
        result1 = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='preliminary', chunk_ids=[1])
        result2 = RecognitionResult(text="world", start_time=1.6, end_time=2.0, status='preliminary', chunk_ids=[2])

        self.formatter.partial_update(result1)
        self.formatter.partial_update(result2)

        self.assertEqual(len(self.formatter.preliminary_results), 2)
        self.assertEqual(self.formatter.preliminary_results[0].text, "Hello")
        self.assertEqual(self.formatter.preliminary_results[1].text, "world")

    def test_empty_preliminary_text(self):
        """Empty preliminary text should be handled gracefully."""
        result = RecognitionResult(text="", start_time=1.0, end_time=1.5, status='preliminary', chunk_ids=[1])

        self.formatter.partial_update(result)

        instructions = self.display.get_last_instructions()
        self.assertEqual(instructions.action, 'update_partial')
        self.assertEqual(instructions.text_to_append, "")
        self.assertFalse(instructions.text_to_append.startswith(" "))

    def test_partial_after_finalization_resets_last_text(self):
        """Partial update after finalization should reset last_finalized_text."""
        final_result = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='final', chunk_ids=[1])
        partial_result = RecognitionResult(text="world", start_time=1.6, end_time=2.0, status='preliminary', chunk_ids=[2])

        self.formatter.finalization(final_result)
        self.assertEqual(self.formatter.last_finalized_text, "Hello")

        self.formatter.partial_update(partial_result)
        self.assertEqual(self.formatter.last_finalized_text, "")


class TestTextFormatterFinalizationChunkTracking(unittest.TestCase):
    """Group B: Finalization & Chunk Tracking (6 tests)"""

    def setUp(self):
        self.display = MockDisplay()
        self.formatter = TextFormatter(display=self.display, verbose=False)

    def test_finalize_updates_chunk_ids(self):
        """Finalization should update finalized_chunk_ids set."""
        result = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='final', chunk_ids=[1, 2, 3])

        self.formatter.finalization(result)

        self.assertEqual(self.formatter.finalized_chunk_ids, {1, 2, 3})

    def test_finalize_filters_preliminary_by_chunk_ids(self):
        """Finalization should filter out preliminary results with overlapping chunk IDs."""
        # Add preliminary results
        partial1 = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='preliminary', chunk_ids=[1])
        partial2 = RecognitionResult(text="world", start_time=1.6, end_time=2.0, status='preliminary', chunk_ids=[2])
        partial3 = RecognitionResult(text="test", start_time=2.1, end_time=2.5, status='preliminary', chunk_ids=[3])

        self.formatter.partial_update(partial1)
        self.formatter.partial_update(partial2)
        self.formatter.partial_update(partial3)

        # Finalize chunks 1-2
        final_result = RecognitionResult(text="Hello world", start_time=1.0, end_time=2.0, status='final', chunk_ids=[1, 2])
        self.formatter.finalization(final_result)

        instructions = self.display.get_last_instructions()
        self.assertEqual(instructions.action, 'finalize')
        self.assertEqual(len(instructions.preliminary_segments), 1)
        self.assertEqual(instructions.preliminary_segments[0].text, "test")

    def test_finalize_duplicate_prevention(self):
        """Duplicate finalization of same text should return None (no display update)."""
        result = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='final', chunk_ids=[1])

        # First finalization
        self.formatter.finalization(result)
        self.assertEqual(len(self.display.instructions_history), 1)

        # Duplicate finalization
        self.formatter.finalization(result)
        # Should still be 1 (no new instruction added)
        self.assertEqual(len(self.display.instructions_history), 1)

    def test_finalize_without_preliminary(self):
        """Finalization without preliminary results should work correctly."""
        result = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='final', chunk_ids=[1])

        self.formatter.finalization(result)

        instructions = self.display.get_last_instructions()
        self.assertEqual(instructions.action, 'finalize')
        self.assertEqual(instructions.finalized_text, "Hello")
        self.assertEqual(len(instructions.preliminary_segments), 0)

    def test_multiple_finalizations_append_with_space(self):
        """Multiple finalizations should append with space separator."""
        result1 = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='final', chunk_ids=[1])
        result2 = RecognitionResult(text="world", start_time=1.6, end_time=2.0, status='final', chunk_ids=[2])

        self.formatter.finalization(result1)
        self.formatter.finalization(result2)

        self.assertEqual(self.formatter.finalized_text, "Hello world")

    def test_finalize_tracks_timing(self):
        """Finalization should track end_time for pause detection."""
        result = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='final', chunk_ids=[1])

        self.formatter.finalization(result)

        self.assertEqual(self.formatter.last_finalized_end_time, 1.5)


class TestTextFormatterParagraphBreakLogic(unittest.TestCase):
    """Group C: Paragraph Break Logic (7 tests)"""

    def setUp(self):
        self.display = MockDisplay()
        self.formatter = TextFormatter(display=self.display, verbose=False)

    def test_pause_detection_triggers_paragraph_break(self):
        """Pause >= threshold should trigger paragraph break and rerender."""
        # Finalize first text
        final1 = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='final', chunk_ids=[1])
        self.formatter.finalization(final1)

        # Add partial after long pause (2.0s threshold)
        partial = RecognitionResult(text="world", start_time=3.6, end_time=4.0, status='preliminary', chunk_ids=[2])
        self.formatter.partial_update(partial)

        instructions = self.display.get_last_instructions()
        self.assertEqual(instructions.action, 'rerender_all')
        self.assertTrue(instructions.paragraph_break_inserted)
        self.assertTrue(self.formatter.finalized_text.endswith('\n'))

    def test_short_pause_no_paragraph_break(self):
        """Pause < threshold should NOT trigger paragraph break."""
        final1 = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='final', chunk_ids=[1])
        self.formatter.finalization(final1)

        # Short pause (1.9s < 2.0s threshold)
        partial = RecognitionResult(text="world", start_time=3.4, end_time=4.0, status='preliminary', chunk_ids=[2])
        self.formatter.partial_update(partial)

        instructions = self.display.get_last_instructions()
        self.assertEqual(instructions.action, 'update_partial')
        self.assertFalse(instructions.paragraph_break_inserted)

    def test_no_duplicate_paragraph_breaks(self):
        """Multiple pauses should not insert duplicate paragraph breaks."""
        final1 = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='final', chunk_ids=[1])
        self.formatter.finalization(final1)

        # First pause - inserts paragraph break
        partial1 = RecognitionResult(text="world", start_time=3.6, end_time=4.0, status='preliminary', chunk_ids=[2])
        self.formatter.partial_update(partial1)
        self.assertTrue(self.formatter.finalized_text.endswith('\n'))

        # Second partial (already has paragraph break)
        partial2 = RecognitionResult(text="test", start_time=4.1, end_time=4.5, status='preliminary', chunk_ids=[3])
        self.formatter.partial_update(partial2)

        instructions = self.display.get_last_instructions()
        self.assertEqual(instructions.action, 'update_partial')
        # Should NOT insert another paragraph break
        self.assertFalse(instructions.paragraph_break_inserted)

    def test_pause_detection_without_finalization(self):
        """Pause detection requires prior finalization (last_finalized_end_time != 0)."""
        # No finalization yet
        partial = RecognitionResult(text="Hello", start_time=10.0, end_time=10.5, status='preliminary', chunk_ids=[1])
        self.formatter.partial_update(partial)

        instructions = self.display.get_last_instructions()
        self.assertEqual(instructions.action, 'update_partial')
        self.assertFalse(instructions.paragraph_break_inserted)

    def test_pause_detection_with_empty_finalized_text(self):
        """Pause detection requires non-empty finalized_text."""
        # Set timing but no finalized text
        self.formatter.last_finalized_end_time = 1.5
        self.formatter.finalized_text = ""

        partial = RecognitionResult(text="Hello", start_time=3.6, end_time=4.0, status='preliminary', chunk_ids=[1])
        self.formatter.partial_update(partial)

        instructions = self.display.get_last_instructions()
        self.assertEqual(instructions.action, 'update_partial')
        self.assertFalse(instructions.paragraph_break_inserted)

    def test_paragraph_break_triggers_rerender(self):
        """Paragraph break should trigger rerender_all action."""
        final1 = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='final', chunk_ids=[1])
        self.formatter.finalization(final1)

        partial = RecognitionResult(text="world", start_time=3.6, end_time=4.0, status='preliminary', chunk_ids=[2])
        self.formatter.partial_update(partial)

        instructions = self.display.get_last_instructions()
        self.assertEqual(instructions.action, 'rerender_all')
        self.assertEqual(instructions.finalized_text, "Hello\n")
        self.assertEqual(len(instructions.preliminary_segments), 1)

    def test_multiple_pauses_multiple_breaks(self):
        """Multiple pauses should insert multiple paragraph breaks."""
        final1 = RecognitionResult(text="First", start_time=1.0, end_time=1.5, status='final', chunk_ids=[1])
        self.formatter.finalization(final1)

        # First pause
        partial1 = RecognitionResult(text="Second", start_time=3.6, end_time=4.0, status='preliminary', chunk_ids=[2])
        self.formatter.partial_update(partial1)
        self.assertEqual(self.formatter.finalized_text, "First\n")

        # Finalize second
        final2 = RecognitionResult(text="Second", start_time=3.6, end_time=4.0, status='final', chunk_ids=[2])
        self.formatter.finalization(final2)

        # Second pause
        partial2 = RecognitionResult(text="Third", start_time=6.1, end_time=6.5, status='preliminary', chunk_ids=[3])
        self.formatter.partial_update(partial2)

        self.assertEqual(self.formatter.finalized_text, "First\n Second\n")


class TestTextFormatterStateInspection(unittest.TestCase):
    """Group D: State Inspection (3 tests)"""

    def setUp(self):
        self.display = MockDisplay()
        self.formatter = TextFormatter(display=self.display, verbose=False)

    def test_get_remaining_preliminary_results(self):
        """_get_remaining_preliminary_results should return non-finalized results."""
        partial1 = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='preliminary', chunk_ids=[1])
        partial2 = RecognitionResult(text="world", start_time=1.6, end_time=2.0, status='preliminary', chunk_ids=[2])
        partial3 = RecognitionResult(text="test", start_time=2.1, end_time=2.5, status='preliminary', chunk_ids=[3])

        self.formatter.partial_update(partial1)
        self.formatter.partial_update(partial2)
        self.formatter.partial_update(partial3)

        # Finalize chunks 1-2
        self.formatter.finalized_chunk_ids.update([1, 2])

        remaining = self.formatter._get_remaining_preliminary_results()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].text, "test")

    def test_remaining_preliminary_filters_by_chunk_overlap(self):
        """Results with ANY finalized chunk should be filtered out."""
        partial1 = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='preliminary', chunk_ids=[1, 2])
        partial2 = RecognitionResult(text="world", start_time=1.6, end_time=2.0, status='preliminary', chunk_ids=[3, 4])

        self.formatter.partial_update(partial1)
        self.formatter.partial_update(partial2)

        # Finalize only chunk 2 (partial1 has [1, 2], so it should be filtered)
        self.formatter.finalized_chunk_ids.add(2)

        remaining = self.formatter._get_remaining_preliminary_results()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].text, "world")

    def test_remaining_preliminary_empty_when_all_finalized(self):
        """When all chunks finalized, remaining should be empty."""
        partial1 = RecognitionResult(text="Hello", start_time=1.0, end_time=1.5, status='preliminary', chunk_ids=[1])
        partial2 = RecognitionResult(text="world", start_time=1.6, end_time=2.0, status='preliminary', chunk_ids=[2])

        self.formatter.partial_update(partial1)
        self.formatter.partial_update(partial2)

        # Finalize all chunks
        self.formatter.finalized_chunk_ids.update([1, 2])

        remaining = self.formatter._get_remaining_preliminary_results()
        self.assertEqual(len(remaining), 0)


if __name__ == '__main__':
    unittest.main()
