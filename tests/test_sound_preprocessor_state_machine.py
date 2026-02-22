# tests/test_sound_preprocessor_state_machine.py
"""Tests for SoundPreProcessor state machine transitions.

Tests the explicit state machine implementation using ProcessingStatesEnum enum.
All tests follow TDD approach: written before implementation.
"""
import pytest
import numpy as np
import queue
from unittest.mock import Mock, patch, MagicMock
from src.sound.SoundPreProcessor import SoundPreProcessor


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
            'max_speech_duration_ms': 3000
        }
    }


@pytest.fixture
def mock_vad():
    """Mock VoiceActivityDetector."""
    vad = Mock()
    return vad


@pytest.fixture
def mock_windower():
    """Mock GrowingWindowAssembler."""
    windower = Mock()
    return windower


@pytest.fixture
def spp(spp_config, mock_vad, mock_windower):
    """SoundPreProcessor instance for testing."""
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    return SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad,
        windower=mock_windower,
        config=spp_config,
        verbose=False
    )


def make_speech_chunk(duration_ms=32, timestamp=0.0):
    """Create a speech audio chunk (512 samples @ 16kHz)."""
    samples = int(16000 * duration_ms / 1000)
    audio = np.random.randn(samples).astype(np.float32) * 0.1
    return {'audio': audio, 'timestamp': timestamp}


def make_silence_chunk(duration_ms=32, timestamp=0.0):
    """Create a silence audio chunk."""
    samples = int(16000 * duration_ms / 1000)
    audio = np.zeros(samples, dtype=np.float32)
    return {'audio': audio, 'timestamp': timestamp}


# ============================================================================
# State Transitions - IDLE
# ============================================================================

def test_state_idle_to_waiting_on_speech(spp, mock_vad):
    """Test IDLE -> WAITING_CONFIRMATION on first speech chunk."""
    # Setup: Mock VAD to return speech
    mock_vad.process_frame.return_value = {
        'is_speech': True,
        'speech_probability': 0.8
    }

    from src.sound.SoundPreProcessor import ProcessingStatesEnum
    assert spp.state == ProcessingStatesEnum.IDLE
    assert spp.audio_state.consecutive_speech_count == 0

    chunk = make_speech_chunk(timestamp=0.0)
    spp._process_chunk(chunk)

    # Verify transition to WAITING_CONFIRMATION
    assert spp.state == ProcessingStatesEnum.WAITING_CONFIRMATION
    assert spp.audio_state.consecutive_speech_count == 1


def test_state_idle_stays_idle_on_silence(spp, mock_vad):
    """Test IDLE stays IDLE on silence chunks."""
    mock_vad.process_frame.return_value = {
        'is_speech': False,
        'speech_probability': 0.1
    }

    from src.sound.SoundPreProcessor import ProcessingStatesEnum
    assert spp.state == ProcessingStatesEnum.IDLE

    # Feed silence chunks
    for i in range(5):
        chunk = make_silence_chunk(timestamp=i * 0.032)
        spp._process_chunk(chunk)

    assert spp.state == ProcessingStatesEnum.IDLE
    assert spp.audio_state.consecutive_speech_count == 0


# ============================================================================
# State Transitions - WAITING_CONFIRMATION
# ============================================================================

def test_state_waiting_to_active_on_third_speech(spp, mock_vad):
    """Test WAITING_CONFIRMATION -> ACTIVE_SPEECH on 3rd consecutive speech chunk."""
    mock_vad.process_frame.return_value = {
        'is_speech': True,
        'speech_probability': 0.8
    }

    from src.sound.SoundPreProcessor import ProcessingStatesEnum

    for i in range(3):
        chunk = make_speech_chunk(timestamp=i * 0.032)
        spp._process_chunk(chunk)

    assert spp.state == ProcessingStatesEnum.ACTIVE_SPEECH
    assert len(spp.audio_state.speech_buffer) == 3
    assert spp.audio_state.left_context_snapshot is not None


def test_state_waiting_to_idle_on_silence(spp, mock_vad):
    """Test WAITING_CONFIRMATION -> IDLE on silence after partial speech."""
    from src.sound.SoundPreProcessor import ProcessingStatesEnum

    # Feed 2 speech chunks
    mock_vad.process_frame.return_value = {
        'is_speech': True,
        'speech_probability': 0.8
    }
    for i in range(2):
        chunk = make_speech_chunk(timestamp=i * 0.032)
        spp._process_chunk(chunk)

    assert spp.state == ProcessingStatesEnum.WAITING_CONFIRMATION
    assert spp.audio_state.consecutive_speech_count == 2

    # Feed silence chunk
    mock_vad.process_frame.return_value = {
        'is_speech': False,
        'speech_probability': 0.1
    }
    chunk = make_silence_chunk(timestamp=2 * 0.032)
    spp._process_chunk(chunk)

    # Verify transition back to IDLE
    assert spp.state == ProcessingStatesEnum.IDLE
    assert spp.audio_state.consecutive_speech_count == 0


def test_state_waiting_continues_on_second_speech(spp, mock_vad):
    """Test WAITING_CONFIRMATION continues on 2nd speech chunk."""
    mock_vad.process_frame.return_value = {
        'is_speech': True,
        'speech_probability': 0.8
    }

    from src.sound.SoundPreProcessor import ProcessingStatesEnum

    for i in range(2):
        chunk = make_speech_chunk(timestamp=i * 0.032)
        spp._process_chunk(chunk)

    # Verify still WAITING_CONFIRMATION
    assert spp.state == ProcessingStatesEnum.WAITING_CONFIRMATION
    assert spp.audio_state.consecutive_speech_count == 2


# ============================================================================
# State Transitions - ACTIVE_SPEECH
# ============================================================================

def test_state_active_stays_active_on_speech(spp, mock_vad):
    """Test ACTIVE_SPEECH stays active on continued speech."""
    mock_vad.process_frame.return_value = {
        'is_speech': True,
        'speech_probability': 0.8
    }

    from src.sound.SoundPreProcessor import ProcessingStatesEnum

    for i in range(10):
        chunk = make_speech_chunk(timestamp=i * 0.032)
        spp._process_chunk(chunk)

    # Verify still ACTIVE_SPEECH
    assert spp.state == ProcessingStatesEnum.ACTIVE_SPEECH
    assert len(spp.audio_state.speech_buffer) == 10


def test_state_active_to_accumulating_on_silence(spp, mock_vad):
    """Test ACTIVE_SPEECH -> ACCUMULATING_SILENCE on silence."""
    from src.sound.SoundPreProcessor import ProcessingStatesEnum

    # Feed 3 speech chunks to activate
    mock_vad.process_frame.return_value = {
        'is_speech': True,
        'speech_probability': 0.8
    }
    for i in range(3):
        chunk = make_speech_chunk(timestamp=i * 0.032)
        spp._process_chunk(chunk)

    assert spp.state == ProcessingStatesEnum.ACTIVE_SPEECH

    # Feed silence chunk
    mock_vad.process_frame.return_value = {
        'is_speech': False,
        'speech_probability': 0.2
    }
    chunk = make_silence_chunk(timestamp=3 * 0.032)
    spp._process_chunk(chunk)

    # Verify transition to ACCUMULATING_SILENCE
    assert spp.state == ProcessingStatesEnum.ACCUMULATING_SILENCE
    assert spp.audio_state.silence_energy > 0


def test_state_active_stays_active_on_max_duration(spp, mock_vad):
    """Test ACTIVE_SPEECH continues after max_duration with hard cut."""
    from src.sound.SoundPreProcessor import ProcessingStatesEnum

    # Setup: Mock VAD to return speech (no silence, so hard cut will happen)
    mock_vad.process_frame.return_value = {
        'is_speech': True,
        'speech_probability': 0.8
    }

    # Feed 95 speech chunks (> 3000ms / 32ms = 93.75 chunks)
    # This will trigger max_duration split with hard cut (no silence to find)
    for i in range(95):
        chunk = make_speech_chunk(timestamp=i * 0.032)
        spp._process_chunk(chunk)

    # Verify state stayed ACTIVE_SPEECH (continuation after hard cut)
    assert spp.state == ProcessingStatesEnum.ACTIVE_SPEECH
    # Verify hard cut happened and new segment started
    assert len(spp.audio_state.speech_buffer) == 1
    assert spp.audio_state.speech_buffer[0]['chunk_id'] == 94
    # Verify windower was called with the segment
    assert spp.windower.process_segment.called


# ============================================================================
# State Transitions - ACCUMULATING_SILENCE
# ============================================================================

def test_state_accumulating_to_active_on_speech_resume(spp, mock_vad):
    """Test ACCUMULATING_SILENCE -> ACTIVE_SPEECH on speech resumption."""
    from src.sound.SoundPreProcessor import ProcessingStatesEnum

    # Activate speech
    mock_vad.process_frame.return_value = {
        'is_speech': True,
        'speech_probability': 0.8
    }
    for i in range(3):
        chunk = make_speech_chunk(timestamp=i * 0.032)
        spp._process_chunk(chunk)

    # Feed silence to enter ACCUMULATING_SILENCE
    mock_vad.process_frame.return_value = {
        'is_speech': False,
        'speech_probability': 0.2
    }
    chunk = make_silence_chunk(timestamp=3 * 0.032)
    spp._process_chunk(chunk)

    assert spp.state == ProcessingStatesEnum.ACCUMULATING_SILENCE
    silence_energy_before = spp.audio_state.silence_energy

    # Resume speech
    mock_vad.process_frame.return_value = {
        'is_speech': True,
        'speech_probability': 0.8
    }
    chunk = make_speech_chunk(timestamp=4 * 0.032)
    spp._process_chunk(chunk)

    # Verify transition back to ACTIVE_SPEECH
    assert spp.state == ProcessingStatesEnum.ACTIVE_SPEECH
    assert spp.audio_state.silence_energy == 0.0


def test_state_accumulating_stays_accumulating(spp, mock_vad):
    """Test ACCUMULATING_SILENCE continues accumulating below threshold."""
    from src.sound.SoundPreProcessor import ProcessingStatesEnum

    # Activate speech
    mock_vad.process_frame.return_value = {
        'is_speech': True,
        'speech_probability': 0.8
    }
    for i in range(3):
        chunk = make_speech_chunk(timestamp=i * 0.032)
        spp._process_chunk(chunk)

    # Feed silence chunk
    mock_vad.process_frame.return_value = {
        'is_speech': False,
        'speech_probability': 0.5  # 1 - 0.5 = 0.5 energy per chunk
    }
    chunk = make_silence_chunk(timestamp=3 * 0.032)
    spp._process_chunk(chunk)

    assert spp.state == ProcessingStatesEnum.ACCUMULATING_SILENCE


def test_state_accumulating_to_idle_on_threshold(spp, mock_vad):
    """Test ACCUMULATING_SILENCE -> IDLE when threshold exceeded."""
    from src.sound.SoundPreProcessor import ProcessingStatesEnum

    # Activate speech
    mock_vad.process_frame.return_value = {
        'is_speech': True,
        'speech_probability': 0.8
    }
    for i in range(3):
        chunk = make_speech_chunk(timestamp=i * 0.032)
        spp._process_chunk(chunk)

    # Feed silence chunks to exceed threshold (1.5)
    # Each chunk with prob=0.1 adds 0.9 energy
    mock_vad.process_frame.return_value = {
        'is_speech': False,
        'speech_probability': 0.1
    }
    for i in range(3):  # 3 * 0.9 = 2.7 > 1.5
        chunk = make_silence_chunk(timestamp=(3 + i) * 0.032)
        spp._process_chunk(chunk)

    assert spp.state == ProcessingStatesEnum.IDLE
    assert spp.windower.flush.called


# ============================================================================
# Test 5: State Consistency
# ============================================================================

def test_state_property_is_speech_active(spp, mock_vad):
    """Test is_speech_active property reflects state correctly."""
    from src.sound.SoundPreProcessor import ProcessingStatesEnum

    # IDLE -> is_speech_active == False
    assert spp.state == ProcessingStatesEnum.IDLE
    assert spp.is_speech_active == False

    # WAITING_CONFIRMATION -> is_speech_active == False
    mock_vad.process_frame.return_value = {
        'is_speech': True,
        'speech_probability': 0.8
    }
    chunk = make_speech_chunk(timestamp=0.0)
    spp._process_chunk(chunk)
    assert spp.state == ProcessingStatesEnum.WAITING_CONFIRMATION
    assert spp.is_speech_active == False

    # ACTIVE_SPEECH -> is_speech_active == True
    for i in range(2):
        chunk = make_speech_chunk(timestamp=(i + 1) * 0.032)
        spp._process_chunk(chunk)
    assert spp.state == ProcessingStatesEnum.ACTIVE_SPEECH
    assert spp.is_speech_active == True

    # ACCUMULATING_SILENCE -> is_speech_active == True
    mock_vad.process_frame.return_value = {
        'is_speech': False,
        'speech_probability': 0.2
    }
    chunk = make_silence_chunk(timestamp=3 * 0.032)
    spp._process_chunk(chunk)
    assert spp.state == ProcessingStatesEnum.ACCUMULATING_SILENCE
    assert spp.is_speech_active == True
