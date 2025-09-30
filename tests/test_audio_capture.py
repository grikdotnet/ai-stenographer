# tests/test_audio_capture.py
import unittest
import queue
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import numpy for real (needed for tests)
import numpy as np

from src.AudioSource import AudioSource
from tests.audio_fixture import MockAudioStream


class TestAudioCapture(unittest.TestCase):
    """Test that voice is captured and added to chunk_queue"""

    def setUp(self):
        """Set up test fixtures"""
        self.chunk_queue = queue.Queue()
        self.audio_source = AudioSource(self.chunk_queue)

    def test_voice_capture_to_chunk_queue(self):
        """Test that audio is captured and chunks are added to queue"""
        # AudioSource now uses chunk_duration=0.5 by default
        # Create mock audio stream with test data matching the default
        mock_stream = MockAudioStream(duration=1.0, sample_rate=16000, chunk_duration=0.5)

        # Mock sounddevice.InputStream to prevent actual microphone access
        with patch('src.AudioSource.sd.InputStream') as mock_input_stream:
            # Create a mock stream instance
            mock_stream_instance = MagicMock()
            mock_input_stream.return_value = mock_stream_instance

            # Start the audio source
            self.audio_source.start()

            # Verify InputStream was called with correct parameters
            mock_input_stream.assert_called_once_with(
                samplerate=16000,
                channels=1,
                callback=self.audio_source.audio_callback,
                blocksize=8000  # 16000 * 0.5 (default chunk_duration)
            )

            # Verify stream.start() was called
            mock_stream_instance.start.assert_called_once()

            # Simulate audio capture by calling the callback directly with mock data
            mock_stream.run_with_callback(self.audio_source.audio_callback)

            # Stop the audio source
            self.audio_source.stop()
            mock_stream_instance.stop.assert_called_once()

        # Verify chunks were added to queue
        self.assertFalse(self.chunk_queue.empty(), "No chunks were added to the queue")

        # Count chunks in queue
        chunk_count = 0
        chunks = []
        while not self.chunk_queue.empty():
            chunk = self.chunk_queue.get()
            chunks.append(chunk)
            chunk_count += 1

        # Should have exactly 2 chunks (1 second of audio at 0.5s chunks = 2 chunks)
        self.assertEqual(chunk_count, 2, f"Expected exactly 2 chunks, got {chunk_count}")

        # Verify chunk structure
        for i, chunk in enumerate(chunks):
            self.assertIn('data', chunk, f"Chunk {i} missing 'data' field")
            self.assertIn('timestamp', chunk, f"Chunk {i} missing 'timestamp' field")

            # Verify data is numpy array with correct properties
            chunk_data = chunk['data']
            self.assertEqual(len(chunk_data), 8000, f"Chunk {i} has wrong size: {len(chunk_data)}")
            self.assertEqual(chunk_data.dtype.name, 'float32', f"Chunk {i} has wrong dtype: {chunk_data.dtype}")

            self.assertIsInstance(chunk['timestamp'], float, f"Chunk {i} timestamp not a float")

    def test_audio_callback_with_voice_data(self):
        """Test the audio callback directly with voice-like data (no microphone needed)"""
        # Create synthetic voice data (sine wave)
        sample_rate = 16000
        duration = 0.5  # 500ms chunk (matches default chunk_duration)
        t = np.linspace(0, duration, int(sample_rate * duration))
        voice_data = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz tone at half amplitude

        # Reshape to match sounddevice format (N, 1)
        indata = voice_data.reshape(-1, 1).astype(np.float32)

        # Call the callback directly (no start() needed)
        self.audio_source.audio_callback(indata, len(indata), None, None)

        # Verify chunk was added to queue
        self.assertFalse(self.chunk_queue.empty(), "No chunk added after callback")

        chunk = self.chunk_queue.get()

        self.assertIn('data', chunk)
        self.assertIn('timestamp', chunk)

        chunk_data = chunk['data']
        self.assertEqual(len(chunk_data), len(voice_data))
        self.assertEqual(chunk_data.dtype, np.float32)

        # Verify audio data was captured correctly (should be similar to input)
        np.testing.assert_array_almost_equal(chunk_data, voice_data, decimal=5)


if __name__ == '__main__':
    unittest.main()