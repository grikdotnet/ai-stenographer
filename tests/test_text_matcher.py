# tests/test_text_matcher.py
import unittest
import queue
import sys
import os
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.TextMatcher import TextMatcher
from src.TextNormalizer import TextNormalizer

class TestTextMatcher(unittest.TestCase):
    """Test that TextMatcher correctly handles overlapping speech recognition results"""

    def setUp(self):
        """Set up test fixtures"""
        self.text_queue = queue.Queue()
        self.final_queue = queue.Queue()
        self.partial_queue = queue.Queue()

    def test_resolve_overlap_with_boundary_errors(self):
        """Test window overlap resolution handling STT boundary recognition errors"""
        text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)

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
        text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)

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
        text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)

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
        text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)

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
        text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)

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
        text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)

        # Original text should be preserved (case, punctuation)
        finalized, remaining = text_matcher.resolve_overlap(
            ("HELLO, World!", 1.0),
            ("hello world? Today.", 2.0)
        )

        # Should preserve original case and punctuation from input windows
        self.assertEqual(finalized, "HELLO, World!")
        self.assertEqual(remaining[0], "Today.")

    def test_process_first_text(self):
        """Test processing the very first text (no previous text)"""
        text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)

        first_text = {
            'text': 'hello world',
            'timestamp': 1.0
        }

        # Process first text using the actual method
        text_matcher.process_text(first_text)

        # Verify results
        self.assertTrue(text_matcher.final_queue.empty(), "First text should not go to final queue")
        self.assertFalse(text_matcher.partial_queue.empty(), "First text should go to partial queue")

        partial_result = text_matcher.partial_queue.get()
        self.assertEqual(partial_result['text'], 'hello world')
        self.assertEqual(partial_result['timestamp'], 1.0)
        self.assertEqual(text_matcher.previous_text, 'hello world')

    def test_process_texts_with_overlap(self):
        """Test processing sequential texts with word overlap through process_text()"""
        text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)

        # Three overlapping windows to test sequential processing
        texts = [
            {'text': 'hello world ho', 'timestamp': 0.0, "is_silence": False, "window_duration": 1.0},
            {'text': 'world how are you do', 'timestamp': 1.0, "is_silence": False, "window_duration": 1.0},
            {'text': 'you doing great today', 'timestamp': 2.0, "is_silence": False, "window_duration": 1.0},
        ]

        # Process each text through process_text()
        text_matcher.process_text(texts[0])  # First text -> partial only
        text_matcher.process_text(texts[1])  # Second text -> finalize first, partial remaining
        text_matcher.process_text(texts[2])  # Third text -> finalize second, partial remaining

        # Verify finalized results
        final_results = []
        while not text_matcher.final_queue.empty():
            final_results.append(text_matcher.final_queue.get())

        self.assertEqual(len(final_results), 2)  # Two finalizations from overlaps
        self.assertEqual(final_results[0]['text'], 'hello world')       # From first overlap (start + common "world")
        self.assertEqual(final_results[1]['text'], 'how are you')       # From second overlap (start + common "you")

        # Verify partial results
        partial_results = []
        while not text_matcher.partial_queue.empty():
            partial_results.append(text_matcher.partial_queue.get())

        self.assertEqual(len(partial_results), 3)  # Three partial texts
        self.assertEqual(partial_results[0]['text'], 'hello world ho')    # First window as partial
        self.assertEqual(partial_results[1]['text'], 'how are you do')    # Remaining after first overlap
        self.assertEqual(partial_results[2]['text'], 'doing great today') # Remaining after second overlap

    def test_finalize_pending_method(self):
        """Test explicit finalize_pending() call finalizes partial text"""
        text_matcher = TextMatcher(
            self.text_queue,
            self.final_queue,
            self.partial_queue
        )

        # Process some speech
        speech_text = {'text': 'hello world', 'timestamp': 1.0}
        text_matcher.process_text(speech_text)

        # Verify it went to partial queue only
        self.assertTrue(self.final_queue.empty())
        self.assertFalse(self.partial_queue.empty())
        partial = self.partial_queue.get()
        self.assertEqual(partial['text'], 'hello world')

        # Explicitly finalize
        text_matcher.finalize_pending()

        # Verify finalized
        self.assertFalse(self.final_queue.empty())
        final = self.final_queue.get()
        self.assertEqual(final['text'], 'hello world')

        # State should be reset
        self.assertEqual(text_matcher.previous_text, "")

    def test_finalize_pending_when_empty(self):
        """Test finalize_pending() with no pending text does nothing"""
        text_matcher = TextMatcher(
            self.text_queue,
            self.final_queue,
            self.partial_queue
        )

        # Call finalize with no pending text
        text_matcher.finalize_pending()

        # Should not crash, queues remain empty
        self.assertTrue(self.final_queue.empty())
        self.assertTrue(self.partial_queue.empty())


if __name__ == '__main__':
    unittest.main()