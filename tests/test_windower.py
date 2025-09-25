# tests/test_windower.py
import unittest
import queue
import sys
import os
import time
import numpy as np

# Add the parent directory to the path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.Windower import Windower


class TestWindower(unittest.TestCase):
    """Test that windows are created correctly from chunks"""

    def setUp(self):
        """Set up test fixtures"""
        self.chunk_queue = queue.Queue()
        self.window_queue = queue.Queue()

    def test_windower_process_chunk(self):
        """Test that Windower can process individual chunks and create windows"""
        windower = Windower(
            self.chunk_queue,
            self.window_queue,
            window_duration=0.4,  # 400ms windows
            step_duration=0.2,    # 200ms steps (50% overlap)
            sample_rate=16000
        )

        # Create chunks with distinct patterns
        chunk_size = 1600  # 100ms at 16kHz
        chunks = []
        for i in range(6):
            chunk_data = np.full(chunk_size, i * 0.1, dtype=np.float32)
            chunks.append({
                'data': chunk_data,
                'timestamp': time.time() + i * 0.1
            })

        # Process each chunk individually (this is how Windower should work)
        for chunk in chunks:
            windower.process_chunk(chunk)

        # Verify windows were created in the output queue
        self.assertFalse(self.window_queue.empty(), "No windows were created")

        # Collect and verify windows
        windows = []
        while not self.window_queue.empty():
            window = self.window_queue.get()
            windows.append(window)

        # Should have exactly 2 windows (6 chunks = 600ms, with 400ms windows and 200ms steps)
        self.assertEqual(len(windows), 2, f"Expected exactly 2 windows, got {len(windows)}")

        # Verify window structure and properties
        for i, window in enumerate(windows):
            self.assertIn('data', window, f"Window {i} missing 'data' field")
            self.assertIn('timestamp', window, f"Window {i} missing 'timestamp' field")
            self.assertIn('duration', window, f"Window {i} missing 'duration' field")

            # Verify window data properties
            window_data = window['data']
            expected_size = 6400  # 0.4 seconds * 16000 samples/sec
            self.assertEqual(len(window_data), expected_size, f"Window {i} wrong size: {len(window_data)}")
            self.assertEqual(window_data.dtype, np.float32, f"Window {i} wrong dtype: {window_data.dtype}")

            # Verify duration
            self.assertAlmostEqual(window['duration'], 0.4, places=3, msg=f"Window {i} wrong duration")

    def test_windower_buffer_accumulation(self):
        """Test that Windower accumulates chunks in buffer before creating windows"""
        windower = Windower(
            self.chunk_queue,
            self.window_queue,
            window_duration=0.3,  # 300ms windows (need 3 chunks)
            step_duration=0.1,    # 100ms steps
            sample_rate=16000
        )

        chunk_size = 1600  # 100ms chunks

        # Add first chunk - should not create window yet
        chunk1 = {'data': np.full(chunk_size, 0.1, dtype=np.float32), 'timestamp': time.time()}
        windower.process_chunk(chunk1)
        self.assertTrue(self.window_queue.empty(), "Window created too early with insufficient data")

        # Add second chunk - still not enough
        chunk2 = {'data': np.full(chunk_size, 0.2, dtype=np.float32), 'timestamp': time.time() + 0.1}
        windower.process_chunk(chunk2)

        # Add third chunk - now should create first window
        chunk3 = {'data': np.full(chunk_size, 0.3, dtype=np.float32), 'timestamp': time.time() + 0.2}
        windower.process_chunk(chunk3)
        self.assertFalse(self.window_queue.empty(), "Window should be created after accumulating enough data")

        # Verify the window content matches concatenated chunks
        window = self.window_queue.get()
        self.assertEqual(window['data'].dtype, np.float32)

        expected_window_data = np.concatenate([
            chunk1['data'],  # 1600 samples of 0.1
            chunk2['data'],  # 1600 samples of 0.2
            chunk3['data']   # 1600 samples of 0.3
        ])

        np.testing.assert_array_equal(window['data'], expected_window_data,
                                    "Window data should be concatenation of chunks 1, 2, 3")

    def test_overlapping_windows(self):
        """Test that windows have correct overlap"""
        windower = Windower(
            self.chunk_queue,
            self.window_queue,
            window_duration=0.2,  # 200ms windows
            step_duration=0.1,    # 100ms steps (50% overlap)
            sample_rate=16000
        )

        # Create continuous audio data that increases linearly
        total_chunks = 4
        chunk_size = 1600  # 100ms
        chunks = []
        for i in range(total_chunks):
            # Create chunk with values [i, i+0.1, i+0.2, ...]
            chunk_data = np.arange(chunk_size) * 0.0001 + i
            chunk_data = chunk_data.astype(np.float32)
            chunks.append({
                'data': chunk_data,
                'timestamp': time.time() + i * 0.1
            })

        # Process each chunk using the actual Windower method
        for chunk in chunks:
            windower.process_chunk(chunk)

        # Collect windows
        windows = []
        while not self.window_queue.empty():
            windows.append(self.window_queue.get())

        window1_data = windows[0]['data']
        window2_data = windows[1]['data']

        # Windows should be 200ms (3200 samples), step is 100ms (1600 samples)
        # So second half of window1 should equal first half of window2
        expected_window_size = 3200
        expected_step_size = 1600

        self.assertEqual(len(window1_data), expected_window_size)
        self.assertEqual(len(window2_data), expected_window_size)

        # Check overlap: last 1600 samples of window1 should match first 1600 samples of window2
        overlap1 = window1_data[-expected_step_size:]
        overlap2 = window2_data[:expected_step_size]

        np.testing.assert_array_equal(
            overlap1, overlap2,
            err_msg="Windows should have exact overlap"
        )

    def test_window_buffer_management(self):
        """Test that buffer is managed correctly to prevent infinite growth"""
        windower = Windower(
            self.chunk_queue,
            self.window_queue,
            window_duration=0.2,
            step_duration=0.1,
            sample_rate=16000
        )

        # Add many chunks to test buffer management
        chunk_size = 1600
        for i in range(10):
            chunk_data = np.full(chunk_size, i, dtype=np.float32)
            self.chunk_queue.put({
                'data': chunk_data,
                'timestamp': time.time() + i * 0.1
            })

        # Process all chunks
        windower.is_running = True
        initial_buffer_len = 0

        while not self.chunk_queue.empty():
            chunk_data = self.chunk_queue.get()
            chunk = chunk_data['data']

            windower.buffer = np.concatenate([windower.buffer, chunk])

            while len(windower.buffer) >= windower.window_size:
                window = windower.buffer[:windower.window_size]
                windower.window_queue.put({
                    'data': window.copy(),
                    'timestamp': chunk_data['timestamp'],
                    'duration': windower.window_size / windower.sample_rate
                })
                windower.buffer = windower.buffer[windower.step_size:]

            # Apply buffer size limit (from original code)
            max_buffer_size = windower.window_size + windower.step_size
            if len(windower.buffer) > max_buffer_size:
                windower.buffer = windower.buffer[-max_buffer_size:]

        # Verify buffer didn't grow unbounded
        max_expected_buffer = windower.window_size + windower.step_size  # 3200 + 1600 = 4800
        self.assertLessEqual(len(windower.buffer), max_expected_buffer,
                           f"Buffer grew too large: {len(windower.buffer)}")

        # Verify windows were still created
        window_count = 0
        while not self.window_queue.empty():
            self.window_queue.get()
            window_count += 1

        self.assertGreater(window_count, 5, f"Expected multiple windows, got {window_count}")


if __name__ == '__main__':
    unittest.main()