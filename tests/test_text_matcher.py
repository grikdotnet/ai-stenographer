# tests/test_text_matcher.py
import unittest
import queue
import sys
import os
from unittest.mock import MagicMock

# Add the parent directory to the path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.TextMatcher import TextMatcher

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
                ("one two three four five six", 1.0), ("or five six seven", 2.0),
                "one two three four five six", "seven", 
                "Handle 'four' -> 'or' corruption"
            ),(
                ("the weather is nice today", 1.0), ("ether is nice today folks", 2.0),
                "the weather is nice today", "folks",
                "Skip corrupted 'ether'"
            ),(
                ("hello world how", 1.0), ("how are you", 2.0),
                "hello world how", "are you",
                "Exact overlap at boundary"
            ),
        ]

        for window1, window2, expected_final, expected_remaining, description in resolution_cases:
            with self.subTest(window1=window1, window2=window2, desc=description):
                finalized, remaining = text_matcher.resolve_overlap(window1, window2)
                self.assertEqual(finalized, expected_final,
                    f"{description}: Expected finalized '{expected_final}', got '{finalized}'")
                self.assertEqual(remaining[0], expected_remaining,
                    f"{description}: Expected remaining '{expected_remaining}', got '{remaining[0]}'")

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

    def test_silence_window_finalizes_partial_text(self):
        """Test that silence window finalizes remaining partial text"""
        text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)

        # Process some speech
        speech_text = {'text': 'hello world', 'timestamp': 1.0}
        text_matcher.process_text(speech_text)

        # Verify it went to partial queue but not final
        self.assertTrue(self.final_queue.empty())
        self.assertFalse(self.partial_queue.empty())
        partial_result = self.partial_queue.get()
        self.assertEqual(partial_result['text'], 'hello world')

        # Process silence signal - should finalize the partial text
        silence_signal = {
            'text': '',
            'timestamp': 2.0,
            'is_silence': True,
            'window_duration': 0
        }
        text_matcher.process_text(silence_signal)

        # Verify partial text was finalized
        self.assertFalse(self.final_queue.empty())
        final_result = self.final_queue.get()
        self.assertEqual(final_result['text'], 'hello world')

        # Silence signal should not go to partial queue (empty text)
        self.assertTrue(self.partial_queue.empty())

    def test_speech_after_silence_starts_new_phrase(self):
        """Test that speech after silence starts a new phrase properly"""
        text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)

        # Process speech -> silence -> new speech sequence
        texts = [
            {'text': 'first phrase', 'timestamp': 1.0},
            {'text': '', 'timestamp': 2.0, 'is_silence': True, 'window_duration': 0},
            {'text': 'second phrase', 'timestamp': 3.0}
        ]

        for text_data in texts:
            text_matcher.process_text(text_data)

        # Should have finalized the first phrase
        final_results = []
        while not text_matcher.final_queue.empty():
            final_results.append(text_matcher.final_queue.get())

        self.assertEqual(len(final_results), 1)
        self.assertEqual(final_results[0]['text'], 'first phrase')

        # Should have new phrase as partial
        partial_results = []
        while not text_matcher.partial_queue.empty():
            partial_results.append(text_matcher.partial_queue.get())

        self.assertEqual(len(partial_results), 2)  # first phrase + new phrase
        self.assertEqual(partial_results[0]['text'], 'first phrase')
        self.assertEqual(partial_results[1]['text'], 'second phrase')

    def test_multiple_speech_then_silence_pattern(self):
        """Test overlapping speech followed by silence finalization"""
        text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)

        # Process overlapping speech then silence
        texts = [
            {'text': 'hello world how', 'timestamp': 1.0},
            {'text': 'how are you', 'timestamp': 2.0},
            {'text': '', 'timestamp': 3.0, 'is_silence': True, 'window_duration': 0}
        ]

        for text_data in texts:
            text_matcher.process_text(text_data)

        # Should have finalized overlapping parts + silence finalization
        final_results = []
        while not text_matcher.final_queue.empty():
            final_results.append(text_matcher.final_queue.get())

        # Should have: overlap finalization + silence finalization
        self.assertEqual(len(final_results), 2)
        self.assertEqual(final_results[0]['text'], 'hello world how')  # From overlap resolution
        self.assertEqual(final_results[1]['text'], 'are you')          # From silence (remaining after overlap)

    def test_silence_only_no_crash(self):
        """Test that processing only silence doesn't cause issues"""
        text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)

        # Process silence as first input
        silence_signal = {
            'text': '',
            'timestamp': 1.0,
            'is_silence': True,
            'window_duration': 0
        }
        text_matcher.process_text(silence_signal)

        # Should not crash and queues should remain empty
        self.assertTrue(self.final_queue.empty())
        self.assertTrue(self.partial_queue.empty())


if __name__ == '__main__':
    unittest.main()