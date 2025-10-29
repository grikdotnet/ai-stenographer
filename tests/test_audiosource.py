# tests/test_audiosource.py
import pytest
import queue
import numpy as np
import time
from unittest.mock import Mock, patch
from src.AudioSource import AudioSource


class TestAudioSource:
    """Tests for simplified AudioSource - minimal callback for audio capture.

    AudioSource now only captures audio and puts raw chunks to queue.
    No VAD, no buffering, no normalization - just fast audio capture.
    """

    @pytest.fixture
    def config(self):
        """Standard configuration for AudioSource."""
        return {
            'audio': {
                'sample_rate': 16000,
                'chunk_duration': 0.032  # 32ms chunks (512 samples)
            }
        }

    def test_audio_callback_puts_raw_chunk_to_queue(self, config):
        """AudioSource callback should put raw audio dict to queue.

        Logic: Callback receives audio → creates dict {audio, timestamp, chunk_id} → puts to queue.
        """
        chunk_queue = queue.Queue()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            config=config,
            verbose=False
        )

        # Simulate sounddevice callback
        # indata shape: (frames, channels)
        indata = np.random.randn(512, 1).astype(np.float32) * 0.1
        frames = 512
        time_info = None
        status = None

        # Mock time.time()
        with patch('time.time', return_value=123.456):
            audio_source.audio_callback(indata, frames, time_info, status)

        # Queue should have one item
        assert not chunk_queue.empty()
        chunk_data = chunk_queue.get()

        # Verify structure
        assert isinstance(chunk_data, dict)
        assert 'audio' in chunk_data
        assert 'timestamp' in chunk_data
        assert 'chunk_id' in chunk_data

        # Verify data
        assert isinstance(chunk_data['audio'], np.ndarray)
        assert chunk_data['audio'].dtype == np.float32
        assert len(chunk_data['audio']) == 512
        assert chunk_data['timestamp'] == 123.456
        assert chunk_data['chunk_id'] == 0


    def test_chunk_id_increments(self, config):
        """Chunk ID should increment with each callback.

        Logic: First call → chunk_id=0, second → chunk_id=1, third → chunk_id=2.
        """
        chunk_queue = queue.Queue()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            config=config,
            verbose=False
        )

        # Trigger callback 3 times
        indata = np.random.randn(512, 1).astype(np.float32) * 0.1

        for _ in range(3):
            audio_source.audio_callback(indata, 512, None, None)

        # Verify chunk IDs are [0, 1, 2]
        chunk_ids = []
        while not chunk_queue.empty():
            chunk_data = chunk_queue.get()
            chunk_ids.append(chunk_data['chunk_id'])

        assert chunk_ids == [0, 1, 2]


    def test_no_processing_in_callback(self, config):
        """Callback should complete very quickly (no blocking operations).

        Logic: Callback should only copy data and put to queue, taking < 1ms.
        """
        chunk_queue = queue.Queue()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            config=config,
            verbose=False
        )

        indata = np.random.randn(512, 1).astype(np.float32) * 0.1

        # Measure callback duration
        start = time.perf_counter()
        for _ in range(100):  # Average over 100 calls
            audio_source.audio_callback(indata, 512, None, None)
        end = time.perf_counter()

        avg_duration_ms = ((end - start) / 100) * 1000

        # Should be well under 1ms per call
        assert avg_duration_ms < 1.0, f"Callback took {avg_duration_ms:.3f}ms (should be <1ms)"


    def test_audio_source_start_stop(self, config):
        """AudioSource should start and stop stream correctly.

        Logic: start() → stream active → stop() → stream inactive.
        """
        chunk_queue = queue.Queue()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            config=config,
            verbose=False
        )

        # Initially not running
        assert not audio_source.is_running
        assert audio_source.stream is None

        # Start should create and start stream
        audio_source.start()
        assert audio_source.is_running
        assert audio_source.stream is not None
        assert audio_source.stream.active

        # Give it a moment to capture some audio
        time.sleep(0.1)

        # Stop should stop stream
        audio_source.stop()
        assert not audio_source.is_running
        assert not audio_source.stream.active
