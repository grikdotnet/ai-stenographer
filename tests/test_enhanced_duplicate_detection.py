# tests/test_enhanced_duplicate_detection.py
import unittest
import queue
from unittest.mock import MagicMock
from src.TextMatcher import TextMatcher
from src.TextNormalizer import TextNormalizer


class TestEnhancedDuplicateDetection(unittest.TestCase):
    """Test enhanced duplicate detection in process_text() - focuses only on duplicate detection phase."""

    def setUp(self):
        """Set up test fixtures."""
        self.text_queue = queue.Queue()
        self.final_queue = queue.Queue()
        self.partial_queue = queue.Queue()
        self.text_normalizer = TextNormalizer()
        self.text_matcher = TextMatcher(
            self.text_queue, self.final_queue, self.partial_queue,
            text_normalizer=self.text_normalizer, verbose=False
        )

    def test_normalized_variants_detected_as_duplicates(self):
        """Test that normalized variants (punctuation, case, Unicode) are detected as duplicates."""
        test_cases = [
            ('Hello.', 'Hello?'),    # Punctuation variants
            ('Hello', 'HELLO'),      # Case variants
            ('чёрный', 'черный'),    # Unicode variants
            ('Straße', 'strasse'),   # Writing variants
            ('Hello', 'Hello')       # Exact duplicates (regression)
        ]

        for original, variant in test_cases:
            with self.subTest(original=original, variant=variant):
                # Clear queues
                while not self.partial_queue.empty():
                    self.partial_queue.get()
                while not self.final_queue.empty():
                    self.final_queue.get()

                # Reset matcher state
                self.text_matcher.previous_text = ""
                self.text_matcher.previous_text_timestamp = 0.0

                # First text
                text_data1 = {'text': original, 'timestamp': 1.0}
                self.text_matcher.process_text(text_data1)

                # Second text - variant within threshold
                text_data2 = {'text': variant, 'timestamp': 1.3}  # 0.3s later
                self.text_matcher.process_text(text_data2)

                # Should only have partial from first text, second should be skipped as duplicate
                self.assertEqual(self.partial_queue.qsize(), 1)
                self.assertEqual(self.final_queue.qsize(), 0)

    def test_different_texts_not_skipped_as_duplicates(self):
        """Test that different texts proceed to overlap resolution."""
        # First text
        text_data1 = {'text': 'Hello', 'timestamp': 1.0}
        self.text_matcher.process_text(text_data1)

        # Second text - completely different content within threshold
        text_data2 = {'text': 'Goodbye', 'timestamp': 1.2}  # 0.2s later
        self.text_matcher.process_text(text_data2)

        # Should proceed to overlap resolution (no duplicate detected)
        # When no overlap found: previous → final_queue, current → partial_queue
        self.assertEqual(self.final_queue.qsize(), 1)
        self.assertEqual(self.partial_queue.qsize(), 2)  # Both texts in partial (first then second)

    def test_time_threshold_affects_duplicate_detection(self):
        """Test that time threshold controls duplicate detection."""
        # First text
        text_data1 = {'text': 'Hello.', 'timestamp': 1.0}
        self.text_matcher.process_text(text_data1)

        # Second text - normalized duplicate but outside default threshold (>0.6s)
        text_data2 = {'text': 'Hello?', 'timestamp': 1.7}  # 0.7s later
        self.text_matcher.process_text(text_data2)

        # Should proceed to overlap resolution since time threshold exceeded
        # "Hello." and "Hello?" have overlap, so finalized part goes to final, remaining to partial
        self.assertEqual(self.final_queue.qsize(), 1)
        self.assertEqual(self.partial_queue.qsize(), 1)


if __name__ == '__main__':
    unittest.main()