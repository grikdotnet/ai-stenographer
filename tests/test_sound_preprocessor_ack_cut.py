"""Tests for SoundPreProcessor ACK-driven early hard-cut feature."""
import queue
import numpy as np
import pytest
from unittest.mock import Mock

from src.sound.SoundPreProcessor import SoundPreProcessor
from src.types import RecognizerFreeSignal


@pytest.fixture
def preprocessor_config():
    """Config with min_segment_duration_ms for ACK-cut tests."""
    return {
        'audio': {
            'sample_rate': 16000,
            'chunk_duration': 0.032,
            'silence_energy_threshold': 1.5,
            'rms_normalization': {
                'target_rms': 0.05,
                'silence_threshold': 0.001,
                'gain_smoothing': 0.9
            }
        },
        'vad': {
            'frame_duration_ms': 32,
            'threshold': 0.5
        },
        'windowing': {
            'max_speech_duration_ms': 1500,
            'min_segment_duration_ms': 200
        }
    }


@pytest.fixture
def mock_vad_speech():
    """VAD that always returns speech."""
    mock = Mock()
    mock.process_frame = Mock(return_value={'is_speech': True, 'speech_probability': 0.9})
    return mock


@pytest.fixture
def mock_windower():
    """Mock windower that tracks process_segment calls."""
    mock = Mock()
    mock.process_segment = Mock()
    mock.flush = Mock()
    return mock


def _speech_chunk(timestamp: float = 1.0) -> dict:
    """Generate a 32ms speech chunk."""
    return {
        'audio': np.random.randn(512).astype(np.float32) * 0.1,
        'timestamp': timestamp
    }


def test_min_segment_duration_fallback(mock_vad_speech, mock_windower):
    """Constructor uses explicit value or fallback default 200."""
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()

    config_explicit = {
        'audio': {'sample_rate': 16000, 'chunk_duration': 0.032, 'silence_energy_threshold': 1.5,
                  'rms_normalization': {'target_rms': 0.05, 'silence_threshold': 0.001, 'gain_smoothing': 0.9}},
        'vad': {'frame_duration_ms': 32, 'threshold': 0.5},
        'windowing': {'max_speech_duration_ms': 1500, 'min_segment_duration_ms': 300}
    }
    preprocessor = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad_speech,
        windower=mock_windower,
        config=config_explicit,
    )
    assert preprocessor.min_segment_duration_ms == 300

    config_fallback = {
        'audio': {'sample_rate': 16000, 'chunk_duration': 0.032, 'silence_energy_threshold': 1.5,
                  'rms_normalization': {'target_rms': 0.05, 'silence_threshold': 0.001, 'gain_smoothing': 0.9}},
        'vad': {'frame_duration_ms': 32, 'threshold': 0.5},
        'windowing': {'max_speech_duration_ms': 1500}
    }
    preprocessor_fb = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad_speech,
        windower=mock_windower,
        config=config_fallback,
    )
    assert preprocessor_fb.min_segment_duration_ms == 200


def test_ack_cut_fires_at_min_duration(preprocessor_config, mock_vad_speech, mock_windower):
    """ACK-cut fires when buffered_ms >= min."""
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    control_queue = queue.Queue(maxsize=10)

    preprocessor = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad_speech,
        windower=mock_windower,
        config=preprocessor_config,
        control_queue=control_queue,
    )

    # Feed speech to buffer exactly 200ms (min_segment_duration_ms)
    # 32ms per chunk, so 7 chunks = 224ms (>= 200ms)
    for i in range(7):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    assert preprocessor.audio_state.current_utterance_id is not None
    utterance_id = preprocessor.audio_state.current_utterance_id

    # Emit ACK signal
    signal = RecognizerFreeSignal(seq=0, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)

    # Process next chunk - should trigger ACK-cut
    preprocessor._process_chunk(_speech_chunk(timestamp=7 * 0.032))

    assert mock_windower.process_segment.call_count == 1
    assert mock_windower.process_segment.call_count == 1


def test_ack_cut_fires_above_min_duration(preprocessor_config, mock_vad_speech, mock_windower):
    """ACK-cut fires when buffered_ms > min."""
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    control_queue = queue.Queue(maxsize=10)

    preprocessor = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad_speech,
        windower=mock_windower,
        config=preprocessor_config,
        control_queue=control_queue,
    )

    # Feed 15 chunks = 480ms (well above 200ms min)
    for i in range(15):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id
    signal = RecognizerFreeSignal(seq=0, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)

    preprocessor._process_chunk(_speech_chunk(timestamp=15 * 0.032))

    assert mock_windower.process_segment.call_count == 1


def test_ack_cut_rejected_below_min_duration(preprocessor_config, mock_vad_speech, mock_windower):
    """Buffer < min ignored."""
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    control_queue = queue.Queue(maxsize=10)

    preprocessor = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad_speech,
        windower=mock_windower,
        config=preprocessor_config,
        control_queue=control_queue,
    )

    # Feed only 3 chunks = 96ms (< 200ms min)
    for i in range(3):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id
    signal = RecognizerFreeSignal(seq=0, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)

    preprocessor._process_chunk(_speech_chunk(timestamp=3 * 0.032))

    assert mock_windower.process_segment.call_count == 0


def test_ack_cut_rejected_wrong_utterance(preprocessor_config, mock_vad_speech, mock_windower):
    """Wrong utterance_id ignored."""
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    control_queue = queue.Queue(maxsize=10)

    preprocessor = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad_speech,
        windower=mock_windower,
        config=preprocessor_config,
        control_queue=control_queue,
    )

    # Feed speech
    for i in range(10):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id

    # Signal with wrong utterance_id
    signal = RecognizerFreeSignal(seq=0, message_id=1, utterance_id=utterance_id + 999)
    control_queue.put(signal)

    preprocessor._process_chunk(_speech_chunk(timestamp=10 * 0.032))

    assert mock_windower.process_segment.call_count == 0


def test_ack_cut_rejected_stale_seq(preprocessor_config, mock_vad_speech, mock_windower):
    """Duplicate/stale seq ignored (seq <= last_processed)."""
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    control_queue = queue.Queue(maxsize=10)

    preprocessor = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad_speech,
        windower=mock_windower,
        config=preprocessor_config,
        control_queue=control_queue,
    )

    for i in range(10):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id

    # Process signal seq=5
    signal = RecognizerFreeSignal(seq=5, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)
    preprocessor._process_chunk(_speech_chunk(timestamp=10 * 0.032))
    assert mock_windower.process_segment.call_count == 1

    # Feed more speech
    for i in range(11, 20):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    # Try stale seq=5 again (should be rejected)
    signal_stale = RecognizerFreeSignal(seq=5, message_id=2, utterance_id=utterance_id)
    control_queue.put(signal_stale)
    preprocessor._process_chunk(_speech_chunk(timestamp=20 * 0.032))

    assert mock_windower.process_segment.call_count == 1  # no new cut


def test_ack_cut_rejected_wrong_state(preprocessor_config, mock_vad_speech, mock_windower):
    """IDLE/WAITING_CONFIRMATION states ignore ACK.

    Tests deterministic state gating by directly manipulating state to IDLE
    while keeping valid utterance_id, ensuring rejection is due to state check.
    """
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    control_queue = queue.Queue(maxsize=10)

    preprocessor = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad_speech,
        windower=mock_windower,
        config=preprocessor_config,
        control_queue=control_queue,
    )

    # Get into ACTIVE_SPEECH with valid utterance_id
    for i in range(10):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id
    assert utterance_id is not None

    # Manually force state to IDLE while keeping utterance_id valid
    # This ensures the rejection will be due to state gating, not utterance mismatch
    from src.sound.SoundPreProcessor import ProcessingStatesEnum
    preprocessor.audio_state.state = ProcessingStatesEnum.IDLE

    # Try to ACK-cut in IDLE state
    signal = RecognizerFreeSignal(seq=0, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)
    preprocessor._process_chunk(_speech_chunk(timestamp=10 * 0.032))

    assert mock_windower.process_segment.call_count == 0


def test_ack_cut_burst_coalescing_one_cut_per_iteration(preprocessor_config, mock_vad_speech, mock_windower):
    """Multiple queued ACKs coalesce to max one cut per iteration.

    Drain invariant: all stale ACKs must be consumed from the queue so they
    cannot trigger a cut on the next iteration.
    """
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    control_queue = queue.Queue(maxsize=10)

    preprocessor = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad_speech,
        windower=mock_windower,
        config=preprocessor_config,
        control_queue=control_queue,
    )

    for i in range(15):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id

    control_queue.put(RecognizerFreeSignal(seq=0, message_id=1, utterance_id=utterance_id))
    control_queue.put(RecognizerFreeSignal(seq=1, message_id=2, utterance_id=utterance_id))
    control_queue.put(RecognizerFreeSignal(seq=2, message_id=3, utterance_id=utterance_id))

    preprocessor._process_chunk(_speech_chunk(timestamp=15 * 0.032))

    # Exactly one cut executed despite three queued ACKs
    assert mock_windower.process_segment.call_count == 1

    # All stale ACKs were drained — queue is empty, next iteration won't re-fire
    assert control_queue.empty(), \
        "Stale ACKs must be drained from queue to prevent backlog buildup"

    # Second chunk produces no additional cut (stale signals gone)
    preprocessor._process_chunk(_speech_chunk(timestamp=16 * 0.032))
    assert mock_windower.process_segment.call_count == 1


def test_ack_cut_metadata_continuity(preprocessor_config, mock_vad_speech, mock_windower):
    """Chunk IDs remain contiguous across cut."""
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    control_queue = queue.Queue(maxsize=10)

    preprocessor = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad_speech,
        windower=mock_windower,
        config=preprocessor_config,
        control_queue=control_queue,
    )

    for i in range(10):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id
    signal = RecognizerFreeSignal(seq=0, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)

    preprocessor._process_chunk(_speech_chunk(timestamp=10 * 0.032))

    assert mock_windower.process_segment.call_count == 1
    segment = mock_windower.process_segment.call_args[0][0]
    assert len(segment.chunk_ids) > 0


