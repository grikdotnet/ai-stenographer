# tests/test_sound_preprocessor.py
import pytest
import queue
import numpy as np
import time
from unittest.mock import Mock
from src.SoundPreProcessor import SoundPreProcessor


class TestSoundPreProcessor:
    """Tests for SoundPreProcessor - VAD, buffering, and windowing orchestration.

    SoundPreProcessor runs in a dedicated thread, processing raw audio chunks
    from chunk_queue through VAD and buffering logic, emitting AudioSegments
    to speech_queue.
    """

    @pytest.fixture
    def sample_audio_chunk(self):
        """32ms chunk of audio at 16kHz (512 samples)"""
        return {
            'audio': np.random.randn(512).astype(np.float32) * 0.1,
            'timestamp': 1.0
        }

    @pytest.fixture
    def preprocessor_config(self):
        """Minimal config for SoundPreProcessor"""
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
                'max_speech_duration_ms': 3000,
                'silence_timeout': 0.5
            }
        }

    @pytest.fixture
    def mock_vad(self):
        mock = Mock()
        mock.process_frame = Mock(return_value={'is_speech': True, 'speech_probability': 0.9})
        return mock

    @pytest.fixture
    def mock_windower(self):
        mock = Mock()
        mock.process_segment = Mock()
        mock.flush = Mock()
        return mock

    def test_rms_normalization_applied_to_audio(self, sample_audio_chunk, preprocessor_config, mock_vad, mock_windower):
        """RMS normalization should boost quiet audio toward target RMS.

        Logic: If audio has RMS=0.02 and target is 0.05, gain should boost it closer to target.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Create chunk with low RMS (~0.02)
        audio = np.random.randn(512).astype(np.float32) * 0.02
        chunk = {'audio': audio, 'timestamp': 1.0}

        original_rms = np.sqrt(np.mean(audio**2))

        preprocessor._process_chunk(chunk)

        assert mock_vad.process_frame.called
        normalized_audio = mock_vad.process_frame.call_args[0][0]

        normalized_rms = np.sqrt(np.mean(normalized_audio**2))

        # Normalized RMS should be closer to target (0.05) than original RMS
        target_rms = 0.05
        assert abs(normalized_rms - target_rms) < abs(original_rms - target_rms)


    def test_rms_normalization_skips_silence(self, preprocessor_config, mock_vad, mock_windower):
        """RMS normalization should skip silent chunks to avoid boosting noise.

        Logic: If RMS < silence_threshold (0.001), gain should remain unchanged.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Create silent chunk (RMS < 0.001)
        audio = np.random.randn(512).astype(np.float32) * 0.0001
        chunk = {'audio': audio, 'timestamp': 1.0}

        original_gain = preprocessor.current_gain

        preprocessor._process_chunk(chunk)

        # Gain should remain unchanged for silence
        assert preprocessor.current_gain == original_gain


    def test_vad_called_with_normalized_audio(self, preprocessor_config, mock_windower):
        """VAD should receive normalized audio, not raw audio.

        Logic: Dual-path processing - normalized for VAD, raw for STT.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Create mock VAD that captures its input
        mock_vad = Mock()
        mock_vad.process_frame = Mock(return_value={'is_speech': True, 'speech_probability': 0.9})

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Create chunk with distinct RMS
        raw_audio = np.random.randn(512).astype(np.float32) * 0.02
        chunk = {'audio': raw_audio.copy(), 'timestamp': 1.0}

        preprocessor._process_chunk(chunk)

        # VAD should have been called
        assert mock_vad.process_frame.called

        # Get the audio passed to VAD
        vad_audio = mock_vad.process_frame.call_args[0][0]

        # VAD audio should be normalized (different from raw)
        raw_rms = np.sqrt(np.mean(raw_audio**2))
        vad_rms = np.sqrt(np.mean(vad_audio**2))

        assert vad_rms > raw_rms


    def test_silence_timeout_triggers_windower_flush(self, preprocessor_config, mock_windower):
        """Silence duration >= timeout should trigger windower.flush().

        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # VAD: 3 speech to trigger, then silence
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return {'is_speech': True, 'speech_probability': 0.9}
            else:
                return {'is_speech': False, 'speech_probability': 0.1}

        mock_vad = Mock()
        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        for i in range(3):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Feed silence chunks for > 0.5s (silence_timeout)
        # 0.5s / 0.032s = ~16 chunks
        for i in range(20):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.01,
                'timestamp': 1.0 + (3 + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        assert mock_windower.flush.called


    def test_flush_emits_pending_segment(self, preprocessor_config, mock_vad, mock_windower):
        """flush() should emit pending segment and call windower.flush().

        Logic: Active buffer with 3 chunks → flush() → segment emitted + windower flushed.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        mock_vad.process_frame = Mock(return_value={'is_speech': True, 'speech_probability': 0.9})

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        for i in range(3):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Manually flush
        preprocessor.flush()

        # Should have emitted preliminary segment
        assert not speech_queue.empty()
        segment = speech_queue.get()
        assert segment.type == 'preliminary'

        assert mock_windower.flush.called

        assert len(preprocessor.speech_buffer) == 0


    def test_timestamp_accuracy(self, preprocessor_config, mock_windower):
        """AudioSegment timestamps should accurately reflect chunk timestamps.

        Logic: Chunks at [1.0, 1.032, 1.064] → start_time=1.0, end_time includes trailing silence.

        With prepended and trailing silence chunks, end_time must include
        total audio duration (prepended + speech + trailing).
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # VAD: silence (1 chunk), speech (3 chunks), then silence
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {'is_speech': False, 'speech_probability': 0.2}
            elif call_count <= 4:
                return {'is_speech': True, 'speech_probability': 0.9}
            else:
                return {'is_speech': False, 'speech_probability': 0.1}

        mock_vad = Mock()
        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Feed 1 silence chunk to be prepended
        chunk = {
            'audio': np.random.randn(512).astype(np.float32) * 0.01,
            'timestamp': 1.0,
        }
        preprocessor._process_chunk(chunk)

        # Feed 3 speech chunks with precise timestamps
        timestamps = [1.032, 1.064, 1.096]
        for i, ts in enumerate(timestamps):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': ts,
            }
            preprocessor._process_chunk(chunk)

        # Feed silence chunks until threshold
        # Each silence chunk with prob=0.1 adds (1-0.1) = 0.9
        # Need 2 chunks: 0.9 + 0.9 = 1.8 >= 1.5
        for i in range(2):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.01,
                'timestamp': 1.128 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Check timestamps
        assert not speech_queue.empty()
        segment = speech_queue.get()

        # start_time should be the first speech chunk (not the prepended silence)
        assert segment.start_time == 1.032

        # end_time = start_time + (total_audio_length / sample_rate)
        # Total audio = 1 prepended + 3 speech + 2 trailing = 6 chunks = 192ms
        # Expected: start_time=1.032, end_time=1.032 + 0.192 = 1.224
        expected_end = 1.032 + (len(segment.data) / 16000)
        assert abs(segment.end_time - expected_end) < 0.001


    def test_idle_buffer_fills_before_speech(self, preprocessor_config, mock_windower):
        """Verify idle_buffer accumulates silence before speech starts.

        Logic: Feed 10 silence chunks → idle_buffer should contain 10 chunks.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD to return silence
        mock_vad = Mock()
        mock_vad.process_frame = Mock(return_value={'is_speech': False, 'speech_probability': 0.2})

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Feed 10 silence chunks
        for i in range(10):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Idle buffer should have 10 chunks
        assert len(preprocessor.idle_buffer) == 10


    def test_left_context_extraction(self, preprocessor_config, mock_windower):
        """Verify left context is extracted from idle_buffer when speech finalizes.

        Logic: Fill idle_buffer with silence, then trigger speech confirmation, then finalize.
        left_context should contain idle_buffer minus chunks extracted for speech_buffer.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Use constants dynamically to avoid fragility
        buffer_size = SoundPreProcessor.CONTEXT_BUFFER_SIZE
        consecutive_chunks = 3  # CONSECUTIVE_SPEECH_CHUNKS

        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= buffer_size:
                return {'is_speech': False, 'speech_probability': 0.2}
            elif call_count <= buffer_size + consecutive_chunks:
                return {'is_speech': True, 'speech_probability': 0.9}
            else:
                return {'is_speech': False, 'speech_probability': 0.1}

        mock_vad = Mock()
        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Feed buffer_size silence chunks to fill idle_buffer
        for i in range(buffer_size):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Feed consecutive_chunks speech chunks to trigger ACTIVE_SPEECH
        for i in range(consecutive_chunks):
            audio = np.random.randn(512).astype(np.float32) * 0.1
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + (buffer_size + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Feed silence to finalize
        for i in range(2):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + (buffer_size + consecutive_chunks + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        assert not speech_queue.empty()
        segment = speech_queue.get()

        # Verify left_context contains idle_buffer minus chunks extracted to speech_buffer
        # When speech is confirmed, consecutive_chunks are extracted from idle_buffer
        # Remaining in idle_buffer = buffer_size - consecutive_chunks
        expected_left_context_chunks = buffer_size - consecutive_chunks
        expected_samples = expected_left_context_chunks * 512
        assert len(segment.left_context) == expected_samples


    def test_right_context_extraction(self, preprocessor_config, mock_windower):
        """Verify right context is last 6 chunks when segment finalized.

        Logic: 10 speech chunks, then 3 silence chunks that finalize segment.
        speech_buffer has 12 chunks total after confirmation and finalization trigger.

        Expected:
        - data = chunks [0-5] (6 chunks, since 12 total - 6 = 6)
        - right_context = chunks [6-11] (last 6 chunks: includes trailing speech + silence)
        - chunk_ids = [0-5] (6 IDs)
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: 10 speech chunks, then 3 silence to finalize
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= 10:
                return {'is_speech': True, 'speech_probability': 0.9}
            else:
                return {'is_speech': False, 'speech_probability': 0.1}

        mock_vad = Mock()
        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Feed 10 speech chunks
        for i in range(10):
            audio = np.random.randn(512).astype(np.float32) * 0.1
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Feed 3 silence chunks to finalize
        for i in range(3):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + (10 + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Segment should be emitted
        assert not speech_queue.empty()
        segment = speech_queue.get()

        # Verify data: 12 - 6 = 6 chunks
        assert len(segment.data) == 6 * 512, \
            f"Expected data length {6 * 512}, got {len(segment.data)}"

        # Verify right_context: last 6 chunks (includes trailing speech + silence)
        assert len(segment.right_context) == 6 * 512, \
            f"Expected right_context length {6 * 512}, got {len(segment.right_context)}"

        # Verify chunk_ids: [0-5] (6 IDs only)
        assert len(segment.chunk_ids) == 6, \
            f"Expected 6 chunk_ids, got {len(segment.chunk_ids)}"
        assert segment.chunk_ids == list(range(6)), \
            f"Expected chunk_ids [0-5], got {segment.chunk_ids}"


    def test_no_left_context_at_startup(self, preprocessor_config, mock_windower):
        """Verify empty left context when speech starts immediately.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return {'is_speech': True, 'speech_probability': 0.9}
            else:
                return {'is_speech': False, 'speech_probability': 0.1}

        mock_vad = Mock()
        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Verify idle_buffer is empty at startup
        assert len(preprocessor.idle_buffer) == 0

        for i in range(3):
            audio = np.random.randn(512).astype(np.float32) * 0.1
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        for i in range(2):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + (3 + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        assert not speech_queue.empty()
        segment = speech_queue.get()

        assert len(segment.left_context) == 0


    def test_idle_buffer_initializes_speech_buffer(self, preprocessor_config, mock_windower):
        """Verify chunks from idle_buffer are used to initialize speech_buffer 
        on transition to ACTIVE_SPEECH.

        Logic: 10 silence chunks (IDLE), 3 speech chunks (triggers ACTIVE_SPEECH) →
        speech_buffer should have 3 chunks with sequential chunk_ids starting at 0.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: 10 silence, then speech
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= 10:
                return {'is_speech': False, 'speech_probability': 0.2}
            else:
                return {'is_speech': True, 'speech_probability': 0.9}

        mock_vad = Mock()
        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Feed 10 silence chunks
        for i in range(10):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Feed 3 speech chunks (triggers ACTIVE_SPEECH)
        for i in range(3):
            audio = np.random.randn(512).astype(np.float32) * 0.1
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + (10 + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Verify speech_buffer has 3 chunks (all from idle_buffer)
        from src.SoundPreProcessor import ProcessingState
        assert preprocessor.state == ProcessingState.ACTIVE_SPEECH
        assert len(preprocessor.speech_buffer) == 3

        assert preprocessor.speech_buffer[0]['chunk_id'] == 0
        assert preprocessor.speech_buffer[1]['chunk_id'] == 1
        assert preprocessor.speech_buffer[2]['chunk_id'] == 2


    def test_waiting_confirmation_timeout_preserves_idle_buffer(self, preprocessor_config, mock_windower):
        """
        Logic: 8 silence chunks (IDLE), 2 speech chunks (WAITING_CONFIRMATION), 1 silence chunk (back to IDLE) →
        idle_buffer should have 11 chunks (8 + 2 + 1).
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: silence, speech, silence
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= 8:
                return {'is_speech': False, 'speech_probability': 0.2}
            elif call_count <= 10:
                return {'is_speech': True, 'speech_probability': 0.9}
            else:
                return {'is_speech': False, 'speech_probability': 0.1}

        mock_vad = Mock()
        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Feed 8 silence chunks
        for i in range(8):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Feed 2 speech chunks
        for i in range(2):
            audio = np.random.randn(512).astype(np.float32) * 0.1
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + (8 + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Feed 1 silence chunk (back to IDLE)
        audio = np.random.randn(512).astype(np.float32) * 0.01
        chunk = {
            'audio': audio,
            'timestamp': 1.0 + 10 * 0.032,
        }
        preprocessor._process_chunk(chunk)

        # Verify state is IDLE and idle_buffer has all 11 chunks
        from src.SoundPreProcessor import ProcessingState
        assert preprocessor.state == ProcessingState.IDLE
        assert len(preprocessor.idle_buffer) == 11


    # Consecutive Speech Chunk Tests (False Positive Prevention)

    def test_max_duration_skips_last_three_chunks_in_search(self, preprocessor_config, mock_windower):
        """Backward search should skip last 3 chunks to ensure right_context.

        Pattern: 88 speech → 1 silence → 5 speech (total 94 chunks)
        Expected: Search skips chunks 91-93, finds silence at chunk 88.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: 88 speech, 1 silence (at idx 88), 5 speech
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            is_silence = (call_count == 88)
            call_count += 1
            if is_silence:
                return {'is_speech': False, 'speech_probability': 0.1}
            else:
                return {'is_speech': True, 'speech_probability': 0.9}

        mock_vad = Mock()
        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        for i in range(94):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        assert not speech_queue.empty()
        segment = speech_queue.get()

        # Segment data: chunks 0-88 (89 chunks: 88 speech + 1 silence)
        assert len(segment.chunk_ids) == 89

        # Right context: chunks 89-93
        assert len(segment.right_context) == 512 * 5


    def test_max_duration_silence_close_to_end(self, preprocessor_config, mock_windower):
        """Silence within searchable range should be used as breakpoint.

        Pattern: 87 speech → 1 silence → 6 speech (total 94 chunks)
        Edge case: Silence at position 87, search range 0-90 (skipping 91-93).
        Expected: Silence found and used.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: 87 speech, 1 silence (at idx 87), 6 speech
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            is_silence = (call_count == 87)
            call_count += 1
            if is_silence:
                return {'is_speech': False, 'speech_probability': 0.1}
            else:
                return {'is_speech': True, 'speech_probability': 0.9}

        mock_vad = Mock()
        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Feed 94 chunks
        for i in range(94):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Should have emitted segment
        assert not speech_queue.empty()
        segment = speech_queue.get()

        # Segment data: chunks 0-87 (88 chunks: 87 speech + 1 silence)
        assert len(segment.chunk_ids) == 88


    def test_max_duration_multiple_silence_uses_last_one(self, preprocessor_config, mock_windower):
        """Backward search should find the last silence chunk.

        Pattern: 30 speech → 1 silence → 30 speech → 1 silence → 32 speech (total 94 chunks)
        Silences at: positions 30 and 61.
        Expected: Backward search finds silence at position 61.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            is_silence = (call_count == 30 or call_count == 61)
            call_count += 1
            if is_silence:
                return {'is_speech': False, 'speech_probability': 0.1}
            else:
                return {'is_speech': True, 'speech_probability': 0.9}

        mock_vad = Mock()
        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        for i in range(94):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        assert not speech_queue.empty()
        segment = speech_queue.get()

        # Segment data: chunks 0-61
        assert len(segment.chunk_ids) == 62
        assert segment.chunk_ids == list(range(62))


    def test_hard_cut_no_breakpoint_found(self, preprocessor_config, mock_windower):
        """When NO silence breakpoint found, use last 6 chunks as right_context.

        Pattern: 94 chunks of continuous speech (no silence)
        Search range: chunks [0-90] (skips last 3: [91-93])
        Breakpoint found: None (no silence in searchable range)

        Expected:
        - data = chunks [0-87] (88 chunks, since 94 - 6 = 88)
        - right_context = chunks [88-93] (last 6 chunks, all speech)
        - chunk_ids = [0-87] (88 IDs, NO duplication)
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: ALL speech, no silence
        mock_vad = Mock()
        mock_vad.process_frame = Mock(return_value={
            'is_speech': True,
            'speech_probability': 0.9
        })

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Feed 94 chunks (triggers max_speech_duration_ms = 3000ms = 93.75 chunks)
        # Collect audio to verify segment contains correct data in correct order
        all_chunks_audio = []
        for i in range(94):
            audio = np.random.randn(512).astype(np.float32) * 0.1
            all_chunks_audio.append(audio)
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        assert not speech_queue.empty()
        segment = speech_queue.get()

        expected_data = np.concatenate(all_chunks_audio[:88])
        assert np.array_equal(segment.data, expected_data), \
            "segment.data should contain exactly chunks 0-87 in order"

        # Verify right_context: last 6 chunks with audio content
        expected_right_context = np.concatenate(all_chunks_audio[88:94])
        assert np.array_equal(segment.right_context, expected_right_context), \
            "right_context should contain exactly chunks 88-93 in order"

        assert len(segment.chunk_ids) == 88, \
            f"Expected 88 chunk_ids, got {len(segment.chunk_ids)}"
        assert segment.chunk_ids == list(range(88)), \
            f"Expected chunk_ids [0-87], got {segment.chunk_ids[:5]}...{segment.chunk_ids[-5:]}"

        for chunk_id in range(88, 94):
            assert chunk_id not in segment.chunk_ids, \
                f"chunk_id {chunk_id} should NOT be in segment.chunk_ids (belongs to right_context)"


    def test_normal_finalization_with_mixed_chunks(self, preprocessor_config, mock_windower):
        """Normal finalization: data includes both speech AND silence chunks.

        Pattern: 5 speech → 1 silence → 7 speech → 3 silence (total 16 chunks)
        Last 3 silence chunks trigger finalization (energy=2.7 > 1.5).

        Note: First 3 speech chunks go to idle_buffer. After 3rd speech chunk is appended,
        speech_buffer is initialized from idle_buffer with all 3 chunks.
        Then remaining chunks are added, totaling 15 chunks in speech_buffer
        (the final silence chunk triggers finalization but isn't added).

        Expected:
        - data = chunks [0-8] (9 chunks: 5 speech + 1 silence + 3 speech)
        - right_context = chunks [9-14] (last 6 chunks: 4 speech + 2 silence)
        - chunk_ids = [0-8] (9 IDs)
        - data includes silence chunk [5] proving silence is preserved in data
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:          # chunks 0-4: speech
                return {'is_speech': True, 'speech_probability': 0.9}
            elif call_count == 6:        # chunk 5: silence
                return {'is_speech': False, 'speech_probability': 0.1}
            elif call_count <= 13:       # chunks 6-12: speech
                return {'is_speech': True, 'speech_probability': 0.9}
            else:                        # chunks 13-15: silence (triggers finalization)
                return {'is_speech': False, 'speech_probability': 0.1}

        mock_vad = Mock()
        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=False
        )

        # Feed 16 chunks total and verify segment contains correct data in correct order
        all_chunks_audio = []
        for i in range(16):
            audio = np.random.randn(512).astype(np.float32) * 0.1
            all_chunks_audio.append(audio)
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        assert not speech_queue.empty()
        segment = speech_queue.get()

        # Verify data: 9 chunks (5 speech + 1 silence + 3 speech)
        expected_data = np.concatenate(all_chunks_audio[:9])
        assert np.array_equal(segment.data, expected_data), \
            "segment.data should contain exactly chunks 0-8 in order (including silence at chunk 5)"

        # Verify right_context: last 6 chunks (4 speech + 2 silence)
        expected_right_context = np.concatenate(all_chunks_audio[9:15])
        assert np.array_equal(segment.right_context, expected_right_context), \
            "right_context should contain exactly chunks 9-14 in order"

        # Verify chunk_ids: [0-8] (9 IDs)
        assert len(segment.chunk_ids) == 9, \
            f"Expected 9 chunk_ids, got {len(segment.chunk_ids)}"
        assert segment.chunk_ids == list(range(9)), \
            f"Expected chunk_ids [0-8], got {segment.chunk_ids}"

        for chunk_id in range(9, 15):
            assert chunk_id not in segment.chunk_ids, \
                f"chunk_id {chunk_id} should NOT be in segment.chunk_ids (belongs to right_context)"

