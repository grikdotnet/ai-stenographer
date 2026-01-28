# tests/test_text_matcher.py
import unittest
import queue
import sys
import os
from unittest.mock import MagicMock, Mock

# Add the parent directory to the path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.postprocessing.TextMatcher import TextMatcher
from src.postprocessing.TextNormalizer import TextNormalizer
from src.types import RecognitionResult

class TestTextMatcher(unittest.TestCase):
    """Test that TextMatcher correctly handles overlapping speech recognition results"""

    def setUp(self):
        """Set up test fixtures"""
        self.text_queue = queue.Queue()
        self.mock_publisher = Mock(spec=['publish_partial_update', 'publish_finalization', 'subscribe', 'unsubscribe'])

    def test_resolve_overlap_with_boundary_errors(self):
        """Test window overlap resolution handling STT boundary recognition errors"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        # Window overlap resolution test cases
        resolution_cases = [
            # (window1, window2, expected_finalized, expected_remaining, description)
            (
                ("one two three four five six", 1.0), ("or five six seven", 1.2),
                "one two three four five six", "seven",
                "Handle 'four' -> 'or' corruption"
            ),(
                ("the weather is nice too", 1.0), ("ether is nice today folks", 1.2),
                "the weather is nice", "today folks",
                "Skip corrupted 'ether'"
            ),(
                ("hello world inter", 1.0), ("world international", 1.1),
                "hello world", "international",
                "Skip corrupted 'old' from window2"
            ),
        ]

        for window1, window2, expected_final, expected_remaining, description in resolution_cases:
            with self.subTest(window1=window1, window2=window2, desc=description):
                finalized, remaining = text_matcher.resolve_overlap(window1, window2)
                self.assertEqual(finalized, expected_final,
                    f"{description}: Expected finalized '{expected_final}', got '{finalized}'")
                self.assertEqual(remaining[0], expected_remaining,
                    f"{description}: Expected remaining '{expected_remaining}', got '{remaining[0]}'")

    def test_resolve_overlap_with_punctuation(self):
        """Test resolve_overlap() handles punctuation variants correctly"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        # Test cases: punctuation should not prevent overlap detection
        cases = [
            # Real debug scenario from user's terminal output
            (
                ("Hello?", 1.0), ("Hello, how are you?", 2.0),
                "Hello?", "how are you?",
                "Question mark vs comma - real debug case"
            ),
            # Additional punctuation variants
            (
                ("world!", 1.0), ("world. today", 2.0),
                "world!", "today",
                "Exclamation vs period"
            ),
            (
                ("one, two.", 1.0), ("two? three!", 2.0),
                "one, two.", "three!",
                "Multiple punctuation marks"
            ),
            (
                ("hello world.", 1.0), ("world! how are", 2.0),
                "hello world.", "how are",
                "Period vs exclamation in middle"
            ),
            (
                ("Good morning?", 1.0), ("Good morning! How", 2.0),
                "Good morning?", "How",
                "Two-word phrase with punctuation change"
            ),
        ]

        for window1, window2, expected_final, expected_remaining, description in cases:
            with self.subTest(window1=window1, window2=window2, desc=description):
                finalized, remaining = text_matcher.resolve_overlap(window1, window2)
                self.assertEqual(finalized, expected_final,
                    f"{description}: Expected finalized '{expected_final}', got '{finalized}'")
                self.assertEqual(remaining[0], expected_remaining,
                    f"{description}: Expected remaining '{expected_remaining}', got '{remaining[0]}'")

    def test_resolve_overlap_with_case_variations(self):
        """Test resolve_overlap() handles case variations correctly"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        cases = [
            # Case should not prevent overlap detection, but original case preserved in output
            (
                ("HELLO world", 1.0), ("hello World HOW", 2.0),
                "HELLO world", "HOW",
                "Mixed case overlap - preserve original case"
            ),
            (
                ("The Quick", 1.0), ("the quick Brown", 2.0),
                "The Quick", "Brown",
                "Capitalization differences"
            ),
            (
                ("ALL CAPS HERE", 1.0), ("all caps here now", 2.0),
                "ALL CAPS HERE", "now",
                "All caps vs lowercase"
            ),
        ]

        for window1, window2, expected_final, expected_remaining, description in cases:
            with self.subTest(window1=window1, window2=window2, desc=description):
                finalized, remaining = text_matcher.resolve_overlap(window1, window2)
                self.assertEqual(finalized, expected_final,
                    f"{description}: Expected finalized '{expected_final}', got '{finalized}'")
                self.assertEqual(remaining[0], expected_remaining,
                    f"{description}: Expected remaining '{expected_remaining}', got '{remaining[0]}'")

    def test_resolve_overlap_with_unicode(self):
        """Test resolve_overlap() handles Unicode and accented characters"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        cases = [
            # Accented characters should normalize for comparison
            (
                ("café here", 1.0), ("cafe here today", 2.0),
                "café here", "today",
                "Accented vs non-accented"
            ),
            # German ß normalization
            (
                ("Straße gut", 1.0), ("strasse gut morgen", 2.0),
                "Straße gut", "morgen",
                "German ß vs ss"
            ),
            # Russian ё normalization
            (
                ("опять чёрный кот", 1.0), ("черный кот бежит", 2.0),
                "опять чёрный кот", "бежит",
                "Russian ё vs е"
            ),
        ]

        for window1, window2, expected_final, expected_remaining, description in cases:
            with self.subTest(window1=window1, window2=window2, desc=description):
                finalized, remaining = text_matcher.resolve_overlap(window1, window2)
                self.assertEqual(finalized, expected_final,
                    f"{description}: Expected finalized '{expected_final}', got '{finalized}'")
                self.assertEqual(remaining[0], expected_remaining,
                    f"{description}: Expected remaining '{expected_remaining}', got '{remaining[0]}'")

    def test_resolve_overlap_edge_cases(self):
        """Test resolve_overlap() edge cases"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        # No overlap - should raise exception
        with self.assertRaises(Exception) as context:
            text_matcher.resolve_overlap(
                ("completely different", 1.0),
                ("nothing matches here", 2.0)
            )
        self.assertIn("No overlap", str(context.exception))

        # Single word windows
        finalized, remaining = text_matcher.resolve_overlap(
            ("hello", 1.0),
            ("hello world", 2.0)
        )
        self.assertEqual(finalized, "hello")
        self.assertEqual(remaining[0], "world")

        # Complete overlap (window2 subset of window1)
        finalized, remaining = text_matcher.resolve_overlap(
            ("one two three", 1.0),
            ("two three", 2.0)
        )
        self.assertEqual(finalized, "one two three")
        self.assertEqual(remaining[0], "")

        # Overlap with extra whitespace
        # Note: split() normalizes whitespace, so "hello  world" becomes ["hello", "world"]
        # and rejoining with single spaces produces "hello world"
        finalized, remaining = text_matcher.resolve_overlap(
            ("hello  world", 1.0),
            ("world   today", 2.0)
        )
        # Whitespace is normalized by split() and rejoin
        self.assertEqual(finalized, "hello world")
        self.assertEqual(remaining[0], "today")

    def test_resolve_overlap_preserves_original_text(self):
        """Test that resolve_overlap() preserves original text formatting in output"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        # Original text should be preserved (case, punctuation)
        finalized, remaining = text_matcher.resolve_overlap(
            ("HELLO, World!", 1.0),
            ("hello world? Today.", 2.0)
        )

        # Should preserve original case and punctuation from input windows
        self.assertEqual(finalized, "HELLO, World!")
        self.assertEqual(remaining[0], "Today.")

    def test_process_first_finalized_text(self):
        """Test processing first finalized text"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        first_text = RecognitionResult(
            text='hello world',
            start_time=1.0,
            end_time=1.5,
            status='final',
            chunk_ids=[1, 2]
        )
        text_matcher.process_text(first_text)

        # First finalized text is stored but not displayed (awaiting overlap resolution)
        self.mock_publisher.publish_partial_update.assert_not_called()
        self.mock_publisher.publish_finalization.assert_not_called()
        self.assertIsNotNone(text_matcher.previous_result)
        self.assertEqual(text_matcher.previous_result.text, 'hello world')

    def test_process_finalized_texts_with_overlap(self):
        """Test processing sequential finalized texts with word overlap through process_text()"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        # Three overlapping windows to test sequential processing
        texts = [
            RecognitionResult('hello world ho', 0.0, 0.5, status='final', chunk_ids=[1, 2, 3]),
            RecognitionResult('world how are you do', 1.0, 1.5, status='final', chunk_ids=[2, 3, 4, 5, 6]),
            RecognitionResult('you doing great today', 2.0, 2.5, status='final', chunk_ids=[5, 6, 7, 8]),
        ]

        # Process each text through process_text()
        text_matcher.process_text(texts[0])  # First text -> stored, not displayed
        text_matcher.process_text(texts[1])  # Second text -> finalize first
        text_matcher.process_text(texts[2])  # Third text -> finalize second

        # Verify finalization was called twice (overlap resolution)
        self.assertEqual(self.mock_publisher.publish_finalization.call_count, 2)

        # Verify the finalized texts - now checking RecognitionResult objects
        finalize_calls = [call[0][0] for call in self.mock_publisher.publish_finalization.call_args_list]
        self.assertIsInstance(finalize_calls[0], RecognitionResult)
        self.assertEqual(finalize_calls[0].text, 'hello world')
        self.assertIsInstance(finalize_calls[1], RecognitionResult)
        self.assertEqual(finalize_calls[1].text, 'how are you')

        # Verify partial_update was never called (finalized flow only)
        self.mock_publisher.publish_partial_update.assert_not_called()

    def test_process_finalized_texts_no_overlap(self):
        """Test finalized texts with no overlap - finalize previous, store current"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        # Two finalized texts with no overlap
        text1 = RecognitionResult('hello world', 1.0, 1.5, status='final', chunk_ids=[1, 2])
        text2 = RecognitionResult('completely different', 2.0, 2.5, status='final', chunk_ids=[3, 4])

        # Process first text
        text_matcher.process_text(text1)
        self.mock_publisher.publish_finalization.assert_not_called()
        self.assertIsNotNone(text_matcher.previous_result)
        self.assertEqual(text_matcher.previous_result.text, 'hello world')

        # Process second text - no overlap, should finalize first and store second
        text_matcher.process_text(text2)
        self.assertEqual(self.mock_publisher.publish_finalization.call_count, 1)
        finalized = self.mock_publisher.publish_finalization.call_args[0][0]
        self.assertIsInstance(finalized, RecognitionResult)
        self.assertEqual(finalized.text, 'hello world')
        self.assertIsNotNone(text_matcher.previous_result)
        self.assertEqual(text_matcher.previous_result.text, 'completely different')

        # Verify partial_update never called
        self.mock_publisher.publish_partial_update.assert_not_called()

    def test_finalize_pending_finalized_text(self):
        """Test explicit finalize_pending() call finalizes stored text"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        # Process some finalized speech
        speech_text = RecognitionResult('hello world', 1.0, 1.5, status='final', chunk_ids=[1, 2])
        text_matcher.process_text(speech_text)

        # Verify it was stored but not displayed (awaiting overlap)
        self.mock_publisher.publish_partial_update.assert_not_called()
        self.mock_publisher.publish_finalization.assert_not_called()

        # Explicitly finalize
        text_matcher.finalize_pending()

        # Verify finalized
        self.assertEqual(self.mock_publisher.publish_finalization.call_count, 1)
        finalized = self.mock_publisher.publish_finalization.call_args[0][0]
        self.assertIsInstance(finalized, RecognitionResult)
        self.assertEqual(finalized.text, 'hello world')

        # State should be reset
        self.assertIsNone(text_matcher.previous_result)

    def test_finalize_pending_when_empty(self):
        """Test finalize_pending() with no pending text does nothing"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        # Call finalize with no pending text
        text_matcher.finalize_pending()

        # Should not crash, no GUI calls should be made
        self.mock_publisher.publish_partial_update.assert_not_called()
        self.mock_publisher.publish_finalization.assert_not_called()

    def test_process_preliminary(self):
        """Preliminary results should NOT use resolve_overlap()"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        # Two preliminary results with no overlap
        prelim1 = RecognitionResult('hello world', 1.0, 1.5, status='preliminary', chunk_ids=[1])
        prelim2 = RecognitionResult('how are you', 2.0, 2.5, status='preliminary', chunk_ids=[2])

        text_matcher.process_text(prelim1)

        self.assertEqual(self.mock_publisher.publish_partial_update.call_count, 1)
        partial = self.mock_publisher.publish_partial_update.call_args[0][0]
        self.assertIsInstance(partial, RecognitionResult)
        self.assertEqual(partial.text, 'hello world')
        self.mock_publisher.publish_finalization.assert_not_called()

        text_matcher.process_text(prelim2)

        # Both should call partial_update() directly (no overlap resolution)
        self.assertEqual(self.mock_publisher.publish_partial_update.call_count, 2)
        calls = [call[0][0] for call in self.mock_publisher.publish_partial_update.call_args_list]
        self.assertEqual(calls[0].text, 'hello world')
        self.assertEqual(calls[1].text, 'how are you')

    def test_process_mixed_preliminary_finalized(self):
        """System should handle mix of preliminary and finalized results"""
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        # Preliminary result (instant, word-level)
        prelim1 = RecognitionResult('hello', 1.0, 1.2, status='preliminary', chunk_ids=[1])

        # Finalized results (sliding windows with overlap)
        final1 = RecognitionResult('hello world how', 1.5, 2.0, status='final', chunk_ids=[1, 2, 3])
        final2 = RecognitionResult('world how are', 2.0, 2.5, status='final', chunk_ids=[2, 3, 4])

        text_matcher.process_text(prelim1)
        text_matcher.process_text(final1)
        text_matcher.process_text(final2)

        # Verify preliminary called partial_update()
        partial_calls = [call[0][0] for call in self.mock_publisher.publish_partial_update.call_args_list]
        self.assertTrue(any(p.text == 'hello' for p in partial_calls))

        # Verify finalized called finalization() after overlap resolution
        self.assertGreater(self.mock_publisher.publish_finalization.call_count, 0)

    def test_process_flush_finalizes_pending(self):
        """Flush result with empty text should trigger finalize_pending()"""
        self.mock_publisher = Mock(spec=['publish_partial_update', 'publish_finalization', 'subscribe', 'unsubscribe'])
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        # 1. Process final text (gets stored in previous_finalized_text)
        final_text = RecognitionResult('hello world', 1.0, 1.5, status='final', chunk_ids=[1, 2])
        text_matcher.process_text(final_text)

        # Verify stored but not finalized
        self.assertIsNotNone(text_matcher.previous_result)
        self.assertEqual(text_matcher.previous_result.text, 'hello world')
        self.mock_publisher.publish_finalization.assert_not_called()

        # 2. Process flush result with empty text
        flush_text = RecognitionResult('', 2.0, 2.0, status='flush', chunk_ids=[])
        text_matcher.process_text(flush_text)

        # Verify finalize_pending() was triggered
        self.assertEqual(self.mock_publisher.publish_finalization.call_count, 1)
        finalized = self.mock_publisher.publish_finalization.call_args[0][0]
        self.assertIsInstance(finalized, RecognitionResult)
        self.assertEqual(finalized.text, 'hello world')
        self.assertIsNone(text_matcher.previous_result)

    def test_process_flush_with_text(self):
        """Flush result with text should process as final, then finalize pending"""
        self.mock_publisher = Mock(spec=['publish_partial_update', 'publish_finalization', 'subscribe', 'unsubscribe'])
        text_matcher = TextMatcher(self.text_queue, self.mock_publisher, app_state=Mock())

        # Process flush result with text (from flushed audio segment)
        flush_text = RecognitionResult('hello world', 1.0, 2.0, status='flush', chunk_ids=[1, 2])
        text_matcher.process_text(flush_text)

        # Flush processing: text -> process_finalized (stores) -> finalize_pending (finalizes & clears)
        # Result: text should be finalized and state cleared
        self.assertEqual(self.mock_publisher.publish_finalization.call_count, 1)
        finalized = self.mock_publisher.publish_finalization.call_args[0][0]
        self.assertEqual(finalized.text, 'hello world')
        # State should be cleared after finalize
        self.assertIsNone(text_matcher.previous_result)


class TestTextMatcherObserverPattern(unittest.TestCase):
    """Tests for TextMatcher observer pattern with ApplicationState.

    Tests that TextMatcher subscribes to ApplicationState and reacts to
    shutdown events via observer callbacks.
    """

    def setUp(self):
        self.text_queue = queue.Queue()
        self.mock_publisher = Mock(spec=['publish_partial_update', 'publish_finalization', 'subscribe', 'unsubscribe'])
        self.mock_app_state = Mock()

    def test_text_matcher_shutdown_calls_finalize_pending(self):
        """Test that shutdown via observer calls finalize_pending() BEFORE stop().

        Logic: on_state_change(_, 'shutdown') should call finalize_pending()
               to flush any remaining partial text before stopping.
        """
        text_matcher = TextMatcher(
            text_queue=self.text_queue,
            publisher=self.mock_publisher,
            app_state=self.mock_app_state
        )

        result = RecognitionResult(
            text="pending text",
            start_time=1.0,
            end_time=2.0,
            status='final',
            chunk_ids=[1, 2],
            confidence=0.95,
            token_confidences=[],
            audio_rms=0.1,
            confidence_variance=0.01
        )
        text_matcher.previous_result = result

        text_matcher.on_state_change('running', 'shutdown')

        # Verify finalization was called with pending result
        self.mock_publisher.publish_finalization.assert_called_once()
        finalized = self.mock_publisher.publish_finalization.call_args[0][0]
        self.assertEqual(finalized.text, "pending text")

        self.assertIsNone(text_matcher.previous_result)

        # Verify is_running is False
        self.assertEqual(text_matcher.is_running, False)


if __name__ == '__main__':
    unittest.main()