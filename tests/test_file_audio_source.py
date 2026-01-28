# tests/test_file_audio_source.py
import pytest
import numpy as np
import queue
import time
from pathlib import Path
from src.sound.FileAudioSource import FileAudioSource


def test_file_audio_source_feeds_chunks_to_queue(config, download_test_audio):
    """Test that FileAudioSource feeds chunks to queue at real-time rate.

    Verifies:
    - start() begins feeding chunks
    - Chunks appear in queue with correct structure
    - Chunks contain 'audio' and 'timestamp' keys
    - Audio data matches expected chunk size
    """
    chunk_queue = queue.Queue()
    audio_file = download_test_audio / 'en.wav'

    source = FileAudioSource(
        chunk_queue=chunk_queue,
        config=config,
        file_path=str(audio_file),
        verbose=False
    )

    # Start feeding
    source.start()

    # Wait for first chunk
    start_time = time.time()
    chunk_data = chunk_queue.get(timeout=1.0)
    elapsed = time.time() - start_time

    # Verify chunk structure
    assert 'audio' in chunk_data
    assert 'timestamp' in chunk_data
    assert len(chunk_data['audio']) == 512
    assert chunk_data['audio'].dtype == np.float32

    # Should receive chunk almost immediately (within 100ms)
    assert elapsed < 0.1

    # Stop feeding
    source.stop()


def test_file_audio_source_timing_accuracy(config, download_test_audio):
    """Test that FileAudioSource feeds chunks at approximately 32ms intervals.

    Verifies:
    - Chunks are fed at real-time rate (~32ms per chunk)
    - Timing is consistent across multiple chunks
    """
    chunk_queue = queue.Queue()
    audio_file = download_test_audio / 'en.wav'

    source = FileAudioSource(
        chunk_queue=chunk_queue,
        config=config,
        file_path=str(audio_file),
        verbose=False
    )

    source.start()

    # Collect first 5 chunks with timestamps
    chunk_times = []
    for _ in range(5):
        start = time.time()
        chunk_queue.get(timeout=1.0)
        chunk_times.append(time.time() - start)

    source.stop()

    assert chunk_times[0] < 0.01

    # Subsequent chunks should be ~32ms apart (allowing some jitter)
    cumulative_time = sum(chunk_times)
    expected_time = 4 * 0.032  # 4 intervals between 5 chunks

    assert abs(cumulative_time - expected_time) < expected_time * 0.5


def test_file_audio_source_stop_halts_feeding(config, download_test_audio):
    """Test that stop() halts chunk feeding.

    Verifies:
    - stop() interrupts chunk feeding
    - No more chunks appear in queue after stop()
    - Thread terminates cleanly
    """
    chunk_queue = queue.Queue()
    audio_file = download_test_audio / 'en.wav'

    source = FileAudioSource(
        chunk_queue=chunk_queue,
        config=config,
        file_path=str(audio_file),
        verbose=False
    )

    source.start()

    # Get first chunk
    chunk_queue.get(timeout=1.0)

    # Stop immediately
    source.stop()

    # Clear any buffered chunks
    time.sleep(0.05)
    while not chunk_queue.empty():
        chunk_queue.get_nowait()

    # Wait a bit and verify no new chunks arrive
    time.sleep(0.1)
    assert chunk_queue.empty()

def test_file_audio_source_resamples_non_16khz_audio(config, tmp_path):
    """Test that FileAudioSource resamples audio to 16kHz if needed.

    Verifies:
    - Non-16kHz audio files are resampled correctly
    - Resampled audio has correct sample rate
    - Chunking works with resampled audio
    """
    import soundfile as sf

    # Create 8kHz audio (will need resampling to 16kHz)
    audio_8khz = np.random.randn(8000).astype(np.float32) * 0.1
    audio_file = tmp_path / 'audio_8khz.wav'
    sf.write(str(audio_file), audio_8khz, 8000)

    chunk_queue = queue.Queue()

    source = FileAudioSource(
        chunk_queue=chunk_queue,
        config=config,
        file_path=str(audio_file),
        verbose=False
    )

    # After resampling, should have ~16000 samples (from 8000)
    total_samples = sum(len(chunk) for chunk in source.audio)
    assert 15000 < total_samples < 17000  # Allow some tolerance

    # Chunks should still be 512 samples at 16kHz
    assert all(len(chunk) == 512 for chunk in source.audio)
