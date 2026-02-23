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
        control_queue=queue.Queue(),
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
        control_queue=queue.Queue(),
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


def test_ack_cut_deferred_when_buffer_too_small(preprocessor_config, mock_vad_speech, mock_windower):
    """ACK signal deferred (not discarded) when buffer < min_segment_duration_ms."""
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
    assert preprocessor._pending_ack_signal is not None


def test_deferred_ack_cut_fires_when_buffer_reaches_minimum(preprocessor_config, mock_vad_speech, mock_windower):
    """Deferred ACK-cut fires exactly when speech_buffer crosses min_segment_duration_ms."""
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

    for i in range(3):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id
    signal = RecognizerFreeSignal(seq=1, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)

    # Chunk 4: pending stored (buffer=3 frames=96ms < 200ms)
    preprocessor._process_chunk(_speech_chunk(timestamp=3 * 0.032))
    assert mock_windower.process_segment.call_count == 0
    assert preprocessor._pending_ack_signal is not None

    # Chunks 5-6: still below threshold
    preprocessor._process_chunk(_speech_chunk(timestamp=4 * 0.032))
    preprocessor._process_chunk(_speech_chunk(timestamp=5 * 0.032))
    assert mock_windower.process_segment.call_count == 0

    # Chunk 7: buffer reaches 224ms >= 200ms → deferred cut fires
    preprocessor._process_chunk(_speech_chunk(timestamp=6 * 0.032))
    assert mock_windower.process_segment.call_count == 1
    assert preprocessor._pending_ack_signal is None
    assert preprocessor._last_ack_seq_processed == signal.seq


def test_deferred_ack_cut_cleared_on_max_duration_split(preprocessor_config, mock_vad_speech, mock_windower):
    """Max-duration split clears pending ACK signal; no additional ACK-cut fires."""
    config = {
        **preprocessor_config,
        'windowing': {'max_speech_duration_ms': 96, 'min_segment_duration_ms': 200}
    }

    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    control_queue = queue.Queue(maxsize=10)

    preprocessor = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad_speech,
        windower=mock_windower,
        config=config,
        control_queue=control_queue,
    )

    for i in range(3):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id
    signal = RecognizerFreeSignal(seq=1, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)

    # Chunk 4: pending stored (96ms < 200ms min); buffer now 4 frames=128ms > max=96ms → split fires
    preprocessor._process_chunk(_speech_chunk(timestamp=3 * 0.032))

    assert mock_windower.process_segment.call_count == 1
    assert preprocessor._pending_ack_signal is None


def test_deferred_ack_cut_cleared_on_utterance_end(preprocessor_config, mock_vad_speech, mock_windower):
    """flush() clears pending ACK signal; windower.flush called, not process_segment."""
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

    for i in range(3):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id
    signal = RecognizerFreeSignal(seq=1, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)

    preprocessor._process_chunk(_speech_chunk(timestamp=3 * 0.032))
    assert preprocessor._pending_ack_signal is not None

    preprocessor.flush()

    assert preprocessor._pending_ack_signal is None
    assert mock_windower.flush.call_count == 1
    assert mock_windower.process_segment.call_count == 0


def test_deferred_ack_cut_replaced_by_newer_signal(preprocessor_config, mock_vad_speech, mock_windower):
    """Higher-seq ACK signal replaces lower-seq pending; only the newer seq is recorded."""
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

    for i in range(3):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id

    control_queue.put(RecognizerFreeSignal(seq=1, message_id=1, utterance_id=utterance_id))
    preprocessor._process_chunk(_speech_chunk(timestamp=3 * 0.032))
    assert preprocessor._pending_ack_signal.seq == 1

    control_queue.put(RecognizerFreeSignal(seq=2, message_id=2, utterance_id=utterance_id))
    preprocessor._process_chunk(_speech_chunk(timestamp=4 * 0.032))
    assert preprocessor._pending_ack_signal.seq == 2

    preprocessor._process_chunk(_speech_chunk(timestamp=5 * 0.032))
    preprocessor._process_chunk(_speech_chunk(timestamp=6 * 0.032))

    assert mock_windower.process_segment.call_count == 1
    assert preprocessor._last_ack_seq_processed == 2


def test_deferred_ack_cut_cleared_on_ineligible_state(preprocessor_config, mock_vad_speech, mock_windower):
    """Pending ACK cleared when state becomes ineligible; no spurious cut fires."""
    from src.sound.SoundPreProcessor import ProcessingStatesEnum

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

    for i in range(3):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id
    signal = RecognizerFreeSignal(seq=1, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)

    preprocessor._process_chunk(_speech_chunk(timestamp=3 * 0.032))
    assert preprocessor._pending_ack_signal is not None

    preprocessor.audio_state.state = ProcessingStatesEnum.IDLE

    preprocessor._process_chunk(_speech_chunk(timestamp=4 * 0.032))

    assert preprocessor._pending_ack_signal is None
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


def _make_preprocessor(config, vad, windower, control_queue=None):
    """Helper to construct SoundPreProcessor with optional control_queue."""
    return SoundPreProcessor(
        chunk_queue=queue.Queue(),
        speech_queue=queue.Queue(),
        vad=vad,
        windower=windower,
        config=config,
        control_queue=control_queue or queue.Queue(maxsize=10),
    )


def _drive_to_accumulating_silence(preprocessor, speech_chunks: int, silence_prob: float = 0.1) -> int:
    """Feed speech_chunks of speech then one silence chunk; return utterance_id.

    Args:
        preprocessor: SoundPreProcessor instance
        speech_chunks: number of speech chunks to feed (must be >= 3 to confirm speech)
        silence_prob: VAD probability used for the silence chunk

    Returns:
        current_utterance_id after speech confirmation
    """
    from src.sound.SoundPreProcessor import ProcessingStatesEnum

    speech_vad = {'is_speech': True, 'speech_probability': 0.9}
    silence_vad = {'is_speech': False, 'speech_probability': silence_prob}

    preprocessor.vad.process_frame = Mock(return_value=speech_vad)
    for i in range(speech_chunks):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    utterance_id = preprocessor.audio_state.current_utterance_id
    assert utterance_id is not None, "utterance must be open after speech confirmation"

    preprocessor.vad.process_frame = Mock(return_value=silence_vad)
    preprocessor._process_chunk(_speech_chunk(timestamp=float(speech_chunks) * 0.032))

    assert preprocessor.audio_state.state == ProcessingStatesEnum.ACCUMULATING_SILENCE
    return utterance_id


def test_ack_cut_during_accumulating_silence_preserves_silence_energy(
    preprocessor_config, mock_vad_speech, mock_windower
):
    """ACK-cut while in ACCUMULATING_SILENCE must not reset silence_energy to zero.

    Algorithm:
    1. Drive to ACCUMULATING_SILENCE with 10 speech chunks + 1 silence chunk.
       silence_energy = 0.9 (one chunk, prob=0.1 → 1.0-0.1=0.9).
    2. Send RecognizerFreeSignal; feed a second silence chunk.
       _check_ack_signals runs first (buffer=10 chunks=320ms >= 200ms min → ACK-cut fires),
       then state machine processes the silence chunk.
    3. Bug: _execute_ack_cut resets silence_energy to 0 and sets state=ACTIVE_SPEECH.
       State machine then: ACTIVE_SPEECH + silence → ACCUMULATING_SILENCE, adds 0.9.
       After chunk: energy = 0.9.
    4. Fix: _execute_ack_cut preserves silence_energy and stays ACCUMULATING_SILENCE.
       State machine then: ACCUMULATING_SILENCE + silence → adds 0.9 → energy = 1.8 >= 1.5
       → IDLE triggered → reset_segment() sets energy=0.0.

    We detect the bug by checking the final state: with the fix the processor reaches
    IDLE (energy crossed threshold); with the bug it stays ACCUMULATING_SILENCE (energy=0.9).
    """
    from src.sound.SoundPreProcessor import ProcessingStatesEnum
    from unittest.mock import Mock

    control_queue = queue.Queue(maxsize=10)
    preprocessor = _make_preprocessor(preprocessor_config, mock_vad_speech, mock_windower, control_queue)

    # 10 speech chunks (320ms > 200ms min) then one silence chunk → ACCUMULATING_SILENCE
    # silence_prob=0.1 → energy contribution per silence chunk = 1.0 - 0.1 = 0.9
    utterance_id = _drive_to_accumulating_silence(preprocessor, speech_chunks=10, silence_prob=0.1)
    assert preprocessor.audio_state.silence_energy == pytest.approx(0.9)

    signal = RecognizerFreeSignal(seq=0, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)

    # Feed a second silence chunk: ACK-cut fires first (buffer >= min),
    # then state machine processes the silence chunk.
    preprocessor.vad.process_frame = Mock(return_value={'is_speech': False, 'speech_probability': 0.1})
    preprocessor._process_chunk(_speech_chunk(timestamp=11 * 0.032))

    # With fix: energy accumulated (0.9+0.9=1.8 >= 1.5 threshold) → IDLE reached
    # With bug: energy was reset → only 0.9 accumulated → still ACCUMULATING_SILENCE
    assert preprocessor.audio_state.state == ProcessingStatesEnum.IDLE, \
        "processor must reach IDLE when silence_energy accumulates across ACK-cut"


def test_ack_cut_during_accumulating_silence_eventually_reaches_idle(
    preprocessor_config, mock_vad_speech, mock_windower
):
    """After an ACK-cut during ACCUMULATING_SILENCE, more silence chunks must reach IDLE.

    Verifies that silence_energy continues accumulating (not reset) so the
    ACCUMULATING_SILENCE → IDLE transition fires correctly.

    silence_energy_threshold = 1.5; silence_prob=0.1 → each chunk adds 0.9.
    With fix: 2 silence chunks total → energy = 1.8 → IDLE triggered.
    With bug: energy keeps resetting; processor never reaches IDLE.
    """
    from src.sound.SoundPreProcessor import ProcessingStatesEnum
    from unittest.mock import Mock

    control_queue = queue.Queue(maxsize=10)
    preprocessor = _make_preprocessor(preprocessor_config, mock_vad_speech, mock_windower, control_queue)

    utterance_id = _drive_to_accumulating_silence(preprocessor, speech_chunks=10, silence_prob=0.1)

    signal = RecognizerFreeSignal(seq=0, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)

    silence_vad = {'is_speech': False, 'speech_probability': 0.1}
    preprocessor.vad.process_frame = Mock(return_value=silence_vad)

    # Feed silence chunks; with the fix IDLE is reached in ≤3 chunks (energy 0.9+0.9+0.9=2.7)
    # With the bug, the processor loops indefinitely and never reaches IDLE
    for i in range(11, 20):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))
        if preprocessor.audio_state.state == ProcessingStatesEnum.IDLE:
            break

    assert preprocessor.audio_state.state == ProcessingStatesEnum.IDLE, \
        "processor must reach IDLE after silence energy exceeds threshold"


def test_ack_cut_during_active_speech_resets_to_active_speech(
    preprocessor_config, mock_vad_speech, mock_windower
):
    """ACK-cut during ACTIVE_SPEECH still transitions to ACTIVE_SPEECH (existing behaviour).

    This test documents the correct behaviour for the ACTIVE_SPEECH case so it
    doesn't regress when fixing the ACCUMULATING_SILENCE case.
    """
    from src.sound.SoundPreProcessor import ProcessingStatesEnum

    control_queue = queue.Queue(maxsize=10)
    preprocessor = _make_preprocessor(preprocessor_config, mock_vad_speech, mock_windower, control_queue)

    # 10 speech chunks, VAD always speech → stays ACTIVE_SPEECH
    for i in range(10):
        preprocessor._process_chunk(_speech_chunk(timestamp=float(i) * 0.032))

    assert preprocessor.audio_state.state == ProcessingStatesEnum.ACTIVE_SPEECH

    utterance_id = preprocessor.audio_state.current_utterance_id
    signal = RecognizerFreeSignal(seq=0, message_id=1, utterance_id=utterance_id)
    control_queue.put(signal)

    preprocessor._process_chunk(_speech_chunk(timestamp=10 * 0.032))

    assert preprocessor.audio_state.state == ProcessingStatesEnum.ACTIVE_SPEECH, \
        "ACK-cut during ACTIVE_SPEECH must keep state as ACTIVE_SPEECH"
    assert mock_windower.process_segment.call_count == 1


