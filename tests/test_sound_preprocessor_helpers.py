# tests/test_sound_preprocessor_helpers.py
"""Tests for SoundPreProcessor helper methods.

Tests extracted helper methods used by the state machine.
All tests follow TDD approach: written before implementation.
"""
import pytest
import numpy as np
import queue
from unittest.mock import Mock
from src.sound.SoundPreProcessor import SoundPreProcessor
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

def test_append_to_idle_buffer(spp):
    """Test _append_to_idle_buffer adds chunk dict to idle_buffer."""
    audio = make_audio()
    timestamp = 1.0
    is_speech = True

    spp._append_to_idle_buffer(audio, timestamp, is_speech)

    assert len(spp.audio_state.idle_buffer) == 1
    chunk = spp.audio_state.idle_buffer[0]
    assert np.array_equal(chunk['audio'], audio)
    assert chunk['timestamp'] == timestamp
    assert chunk['is_speech'] == is_speech
    assert chunk['chunk_id'] is None


def test_append_to_speech_buffer(spp):
    """Test _append_to_speech_buffer adds chunk dict to speech_buffer."""
    audio = make_audio()
    timestamp = 1.0
    is_speech = True
    chunk_id = 42

    spp._append_to_speech_buffer(audio, timestamp, is_speech, chunk_id)

    assert len(spp.audio_state.speech_buffer) == 1
    chunk = spp.audio_state.speech_buffer[0]
    assert np.array_equal(chunk['audio'], audio)
    assert chunk['timestamp'] == timestamp
    assert chunk['is_speech'] == is_speech
    assert chunk['chunk_id'] == chunk_id


# ============================================================================
# Test: Segment Finalization
# ============================================================================

def test_keep_remainder_after_breakpoint(spp):
    """Test _keep_remainder_after_breakpoint keeps remainder in buffer."""
    spp.audio_state.speech_start_time = 1.0
    for i in range(60):
        spp.audio_state.speech_buffer.append({
            'audio': make_audio(),
            'timestamp': 1.0 + i * 0.032,
            'chunk_id': i,
            'is_speech': True
        })

    spp._keep_remainder_after_breakpoint(40)

    # Verify remainder
    assert len(spp.audio_state.speech_buffer) == 19  # 60 - 41
    assert spp.audio_state.speech_start_time == 1.0 + 41 * 0.032
    assert spp.audio_state.silence_energy == 0.0


def test_reset_segment_state(spp):
    """Test _reset_segment_state clears all segment state."""
    spp.audio_state.speech_buffer = [{'audio': make_audio(), 'timestamp': 1.0, 'chunk_id': 0, 'is_speech': True}]
    spp.audio_state.silence_energy = 1.0
    spp.audio_state.left_context_snapshot = [make_audio()]

    spp._reset_segment_state()

    # Verify all cleared
    assert len(spp.audio_state.speech_buffer) == 0
    assert spp.audio_state.silence_energy == 0.0
    assert spp.audio_state.left_context_snapshot is None
