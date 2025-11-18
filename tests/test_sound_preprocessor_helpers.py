# tests/test_sound_preprocessor_helpers.py
"""Tests for SoundPreProcessor helper methods.

Tests extracted helper methods used by the state machine.
All tests follow TDD approach: written before implementation.
"""
import pytest
import numpy as np
import queue
from unittest.mock import Mock
from src.SoundPreProcessor import SoundPreProcessor
from src.types import AudioSegment


@pytest.fixture
def spp_config():
    """Standard configuration for SoundPreProcessor tests."""
    return {
        'audio': {
            'sample_rate': 16000,
            'silence_energy_threshold': 1.5,
            'rms_normalization': {
                'target_rms': 0.05,
                'silence_threshold': 0.01,
                'gain_smoothing': 0.9
            }
        },
        'vad': {
            'frame_duration_ms': 32,
            'threshold': 0.5
        },
        'windowing': {
            'max_speech_duration_ms': 3000,
            'silence_timeout': 2.0
        }
    }


@pytest.fixture
def spp(spp_config):
    """SoundPreProcessor instance for testing."""
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    mock_vad = Mock()
    mock_windower = Mock()
    return SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad,
        windower=mock_windower,
        config=spp_config,
        verbose=False
    )


def make_audio(num_samples=512):
    """Create audio array."""
    return np.random.randn(num_samples).astype(np.float32) * 0.1


# ============================================================================
# Test: Buffering Methods
# ============================================================================

def test_append_to_context_buffer(spp):
    """Test _append_to_context_buffer adds chunk dict to context_buffer."""
    audio = make_audio()
    timestamp = 1.0
    is_speech = True
    chunk_id = 42

    spp._append_to_context_buffer(audio, timestamp, is_speech, chunk_id)

    assert len(spp.context_buffer) == 1
    chunk = spp.context_buffer[0]
    assert np.array_equal(chunk['audio'], audio)
    assert chunk['timestamp'] == timestamp
    assert chunk['is_speech'] == is_speech
    assert chunk['chunk_id'] == chunk_id


def test_append_to_speech_buffer(spp):
    """Test _append_to_speech_buffer adds chunk dict to speech_buffer."""
    audio = make_audio()
    timestamp = 1.0
    is_speech = True
    chunk_id = 42

    spp._append_to_speech_buffer(audio, timestamp, is_speech, chunk_id)

    assert len(spp.speech_buffer) == 1
    chunk = spp.speech_buffer[0]
    assert np.array_equal(chunk['audio'], audio)
    assert chunk['timestamp'] == timestamp
    assert chunk['is_speech'] == is_speech
    assert chunk['chunk_id'] == chunk_id


# ============================================================================
# Test: Segment Initialization
# ============================================================================

def test_capture_left_context_from_buffer(spp):
    """Test _capture_left_context_from_buffer extracts left context correctly."""
    # Populate context_buffer with 10 chunks
    for i in range(10):
        spp.context_buffer.append({
            'audio': make_audio(),
            'timestamp': i * 0.032,
            'chunk_id': i,
            'is_speech': True
        })

    # Simulate 3 consecutive speech chunks (last 2 already in buffer)
    spp.consecutive_speech_count = 3

    # Call method
    spp._capture_left_context_from_buffer()

    # Verify left_context_snapshot contains first 8 chunks (10 - 2)
    assert spp.left_context_snapshot is not None
    assert len(spp.left_context_snapshot) == 8


def test_initialize_speech_buffer_from_context(spp):
    """Test _initialize_speech_buffer_from_context initializes speech buffer."""
    # Populate context_buffer with last 2 speech chunks
    for i in range(10):
        spp.context_buffer.append({
            'audio': make_audio(),
            'timestamp': i * 0.032,
            'chunk_id': None,
            'is_speech': False
        })

    # Add last 2 speech chunks
    spp.context_buffer.append({
        'audio': make_audio(),
        'timestamp': 10 * 0.032,
        'chunk_id': None,
        'is_speech': True
    })
    spp.context_buffer.append({
        'audio': make_audio(),
        'timestamp': 11 * 0.032,
        'chunk_id': None,
        'is_speech': True
    })

    spp.consecutive_speech_count = 3
    spp.chunk_id_counter = 100

    # Call method
    spp._initialize_speech_buffer_from_context()

    # Verify speech_buffer has 2 chunks
    assert len(spp.speech_buffer) == 2
    assert spp.speech_buffer[0]['chunk_id'] is not None
    assert spp.speech_buffer[1]['chunk_id'] is not None
    assert spp.speech_start_time == spp.speech_buffer[0]['timestamp']


# ============================================================================
# Test: Segment Finalization
# ============================================================================

def test_keep_remainder_after_breakpoint(spp):
    """Test _keep_remainder_after_breakpoint keeps remainder in buffer."""
    spp.speech_start_time = 1.0
    for i in range(60):
        spp.speech_buffer.append({
            'audio': make_audio(),
            'timestamp': 1.0 + i * 0.032,
            'chunk_id': i,
            'is_speech': True
        })

    spp._keep_remainder_after_breakpoint(40)

    # Verify remainder
    assert len(spp.speech_buffer) == 19  # 60 - 41
    assert spp.speech_start_time == 1.0 + 41 * 0.032
    assert spp.silence_energy == 0.0


def test_reset_segment_state(spp):
    """Test _reset_segment_state clears all segment state."""
    spp.speech_buffer = [{'audio': make_audio(), 'timestamp': 1.0, 'chunk_id': 0, 'is_speech': True}]
    spp.silence_energy = 1.0
    spp.left_context_snapshot = [make_audio()]

    spp._reset_segment_state()

    # Verify all cleared
    assert len(spp.speech_buffer) == 0
    assert spp.silence_energy == 0.0
    assert spp.left_context_snapshot is None
