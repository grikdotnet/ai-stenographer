# tests/test_enhanced_duplicate_detection.py
import unittest
import queue
from unittest.mock import Mock
from src.TextMatcher import TextMatcher
from src.TextNormalizer import TextNormalizer


class TestEnhancedDuplicateDetection(unittest.TestCase):
    """Test enhanced duplicate detection in process_text() - focuses only on duplicate detection phase."""

    def setUp(self):
        """Set up test fixtures."""
        self.text_queue = queue.Queue()
        self.mock_gui = Mock(spec=['update_partial', 'finalize_text'])
        self.text_normalizer = TextNormalizer()
        self.text_matcher = TextMatcher(
            self.text_queue, self.mock_gui,
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
                # Reset mock and matcher state
                self.mock_gui.reset_mock()
                self.text_matcher.previous_finalized_text = ""
                self.text_matcher.previous_finalized_timestamp = 0.0

                # First text (finalized - duplicate detection only in finalized path)
                text_data1 = {'text': original, 'timestamp': 1.0, 'is_preliminary': False}
                self.text_matcher.process_text(text_data1)

                # Second text - variant within threshold
                text_data2 = {'text': variant, 'timestamp': 1.3, 'is_preliminary': False}  # 0.3s later
                self.text_matcher.process_text(text_data2)

                # Should only have one update_partial call from first text, second should be skipped as duplicate
                self.assertEqual(self.mock_gui.update_partial.call_count, 1)
                self.assertEqual(self.mock_gui.finalize_text.call_count, 0)

    def test_different_texts_not_skipped_as_duplicates(self):
        """Test that different texts proceed to overlap resolution."""
        # First text (finalized)
        text_data1 = {'text': 'Hello', 'timestamp': 1.0, 'is_preliminary': False}
        self.text_matcher.process_text(text_data1)

        # Second text - completely different content within threshold
        text_data2 = {'text': 'Goodbye', 'timestamp': 1.2, 'is_preliminary': False}  # 0.2s later
        self.text_matcher.process_text(text_data2)

        # Should proceed to overlap resolution (no duplicate detected)
        # When no overlap found: previous → finalize_text, current → update_partial
        self.assertEqual(self.mock_gui.finalize_text.call_count, 1)
        self.assertEqual(self.mock_gui.update_partial.call_count, 2)  # Both texts in partial (first then second)

    def test_time_threshold_affects_duplicate_detection(self):
        """Test that time threshold controls duplicate detection."""
        # First text (finalized)
        text_data1 = {'text': 'Hello.', 'timestamp': 1.0, 'is_preliminary': False}
        self.text_matcher.process_text(text_data1)

        # Second text - normalized duplicate but outside default threshold (>0.6s)
        text_data2 = {'text': 'Hello?', 'timestamp': 1.7, 'is_preliminary': False}  # 0.7s later
        self.text_matcher.process_text(text_data2)

        # Should proceed to overlap resolution since time threshold exceeded
        # "Hello." and "Hello?" have overlap, so finalized part goes to finalize_text
        self.assertEqual(self.mock_gui.finalize_text.call_count, 1)
        # First partial + remaining after overlap
        self.assertGreaterEqual(self.mock_gui.update_partial.call_count, 1)


if __name__ == '__main__':
    unittest.main()