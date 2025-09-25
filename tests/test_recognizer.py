# tests/test_recognizer.py
import unittest
import queue
import sys
import os
import time
import numpy as np
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.Recognizer import Recognizer


class TestRecognizer(unittest.TestCase):
    """Test that windows are processed for speech recognition"""

    def setUp(self):
        """Set up test fixtures"""
        self.window_queue = queue.Queue()
        self.text_queue = queue.Queue()

    def test_recognize_window_with_different_inputs(self):
        """Test recognition with speech, silence, and data format validation"""
        mock_model = MagicMock()
        recognition_results = ["hello world", "", "test"]  # Speech, silence, speech
        mock_model.recognize.side_effect = recognition_results

        recognizer = Recognizer(self.window_queue, self.text_queue, mock_model)

        # Test different types of windows
        windows = [
            # Speech window with specific audio data
            {
                'data': np.sin(2 * np.pi * 440 * np.linspace(0, 0.2, 3200)).astype(np.float32),
                'timestamp': 1.0,
                'duration': 0.2
            },
            # Silence window
            {
                'data': np.zeros(3200, dtype=np.float32),
                'timestamp': 2.0,
                'duration': 0.2
            },
            # Another speech window
            {
                'data': np.random.rand(3200).astype(np.float32),
                'timestamp': 3.0,
                'duration': 0.2
            }
        ]

        # Process all windows
        for window_data in windows:
            recognizer.recognize_window(window_data)

        # Should have 2 text results (excluding empty string)
        text_results = []
        while not self.text_queue.empty():
            text_results.append(self.text_queue.get())

        self.assertEqual(len(text_results), 2)
        expected_texts = ["hello world", "test"]
        actual_texts = [result['text'] for result in text_results]
        self.assertEqual(actual_texts, expected_texts)

        # Verify result structure
        for i, result in enumerate(text_results):
            self.assertIn('text', result)
            self.assertIn('timestamp', result)
            self.assertIn('window_duration', result)
            self.assertEqual(result['window_duration'], 0.2)

    def test_process_method(self):
        """Test the actual process() method with controlled execution"""
        mock_model = MagicMock()
        mock_model.recognize.side_effect = ["hello", "world"]

        recognizer = Recognizer(self.window_queue, self.text_queue, mock_model)

        # Add windows to input queue
        windows = [
            {'data': np.random.rand(3200).astype(np.float32), 'timestamp': 1.0, 'duration': 0.2},
            {'data': np.random.rand(3200).astype(np.float32), 'timestamp': 2.0, 'duration': 0.2}
        ]
        for window in windows:
            self.window_queue.put(window)

        # Set up controlled execution
        recognizer.is_running = True
        original_get = recognizer.window_queue.get
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Process first two windows
                return original_get(*args, **kwargs)
            else:
                recognizer.is_running = False  # Stop after processing
                raise queue.Empty()

        # Replace the get method and call actual process()
        recognizer.window_queue.get = mock_get
        recognizer.process()  # Call the actual method being tested

        # Verify results
        text_results = []
        while not self.text_queue.empty():
            text_results.append(self.text_queue.get())

        self.assertEqual(len(text_results), 2)
        expected_texts = ["hello", "world"]
        actual_texts = [result['text'] for result in text_results]
        self.assertEqual(actual_texts, expected_texts)

        # Verify all windows were processed
        self.assertTrue(self.window_queue.empty())


if __name__ == '__main__':
    unittest.main()