# tests/test_sound_preprocessor.py
import pytest
import queue
import numpy as np
import time
from unittest.mock import Mock
from src.sound.SoundPreProcessor import SoundPreProcessor, CONTEXT_BUFFER_SIZE


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

        # flush() passes segment to windower.flush()
        assert mock_windower.flush.called
        segment = mock_windower.flush.call_args[0][0]
        assert segment.type == 'incremental'

        assert len(preprocessor.audio_state.speech_buffer) == 0


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

        # Segment goes to windower.process_segment()
        assert mock_windower.process_segment.called
        segment = mock_windower.process_segment.call_args[0][0]

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
        assert len(preprocessor.audio_state.idle_buffer) == 10


    def test_left_context_extraction(self, preprocessor_config, mock_windower):
        """Verify left context is extracted from idle_buffer when speech finalizes.

        Logic: Fill idle_buffer with silence, then trigger speech confirmation, then finalize.
        left_context should contain idle_buffer minus chunks extracted for speech_buffer.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Use constants dynamically to avoid fragility
        consecutive_chunks = 3

        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= CONTEXT_BUFFER_SIZE:
                return {'is_speech': False, 'speech_probability': 0.2}
            elif call_count <= CONTEXT_BUFFER_SIZE + consecutive_chunks:
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
        for i in range(CONTEXT_BUFFER_SIZE):
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
                'timestamp': 1.0 + (CONTEXT_BUFFER_SIZE + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Feed silence to finalize
        for i in range(2):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + (CONTEXT_BUFFER_SIZE + consecutive_chunks + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Segment goes to windower.process_segment()
        assert mock_windower.process_segment.called
        segment = mock_windower.process_segment.call_args[0][0]

        # Verify left_context contains idle_buffer minus chunks extracted to speech_buffer
        # When speech is confirmed, consecutive_chunks are extracted from idle_buffer
        # Remaining in idle_buffer = buffer_size - consecutive_chunks
        expected_left_context_chunks = CONTEXT_BUFFER_SIZE - consecutive_chunks
        expected_samples = expected_left_context_chunks * 512
        assert len(segment.left_context) == expected_samples


    def test_right_context_extraction(self, preprocessor_config, mock_windower):
        """Verify right context contains trailing silence chunks when segment finalized.

        Logic: 10 speech chunks, then 3 silence chunks that finalize segment.
        First 3 speech chunks trigger confirmation, so speech_buffer gets:
        - 3 chunks from confirmation (chunk_ids 0-2)
        - 7 more speech chunks (chunk_ids 3-9)
        - 2 silence chunks before finalization (chunk_ids 10-11)
        Note: 3rd silence chunk not processed because finalization triggers after 2nd.

        With silence_start_idx tracking (silence_start_idx=10):
        - data = chunks [0-10] (11 chunks: 10 speech + first silence at breakpoint)
        - right_context = chunk [11] (1 silence chunk after breakpoint)
        - chunk_ids = [0-10] (11 IDs)
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

        # Segment goes to windower.process_segment()
        assert mock_windower.process_segment.called
        segment = mock_windower.process_segment.call_args[0][0]

        # Verify data: 10 speech + 1 silence at breakpoint
        assert len(segment.data) == 11 * 512, \
            f"Expected data length {11 * 512}, got {len(segment.data)}"

        # Verify right_context: 1 silence chunk after breakpoint
        assert len(segment.right_context) == 1 * 512, \
            f"Expected right_context length {1 * 512}, got {len(segment.right_context)}"

        # Verify chunk_ids: [0-10] (11 IDs)
        assert len(segment.chunk_ids) == 11, \
            f"Expected 11 chunk_ids, got {len(segment.chunk_ids)}"
        assert segment.chunk_ids == list(range(11)), \
            f"Expected chunk_ids [0-10], got {segment.chunk_ids}"


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
        assert len(preprocessor.audio_state.idle_buffer) == 0

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

        # Segment goes to windower.process_segment()
        assert mock_windower.process_segment.called
        segment = mock_windower.process_segment.call_args[0][0]

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
        from src.sound.SoundPreProcessor import ProcessingStatesEnum
        assert preprocessor.audio_state.state == ProcessingStatesEnum.ACTIVE_SPEECH
        assert len(preprocessor.audio_state.speech_buffer) == 3

        assert preprocessor.audio_state.speech_buffer[0]['chunk_id'] == 0
        assert preprocessor.audio_state.speech_buffer[1]['chunk_id'] == 1
        assert preprocessor.audio_state.speech_buffer[2]['chunk_id'] == 2


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
        from src.sound.SoundPreProcessor import ProcessingStatesEnum
        assert preprocessor.audio_state.state == ProcessingStatesEnum.IDLE
        assert len(preprocessor.audio_state.idle_buffer) == 11


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

        # Segment goes to windower.process_segment()
        assert mock_windower.process_segment.called
        segment = mock_windower.process_segment.call_args[0][0]

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

        # Segment goes to windower.process_segment()
        assert mock_windower.process_segment.called
        segment = mock_windower.process_segment.call_args[0][0]

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

        # Segment goes to windower.process_segment()
        assert mock_windower.process_segment.called
        segment = mock_windower.process_segment.call_args[0][0]

        # Segment data: chunks 0-61
        assert len(segment.chunk_ids) == 62
        assert segment.chunk_ids == list(range(62))


    def test_hard_cut_no_breakpoint_found(self, preprocessor_config, mock_windower):
        """When NO silence breakpoint found, hard cut with empty right_context.

        Pattern: 94 chunks of continuous speech (no silence)
        Search range: chunks [0-90] (skips last 3: [91-93])
        Breakpoint found: None (no silence in searchable range)

        Expected (hard cut = no natural boundary):
        - data = all chunks [0-93] (94 chunks)
        - right_context = empty (no silence boundary exists)
        - chunk_ids = [0-93] (94 IDs)
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

        # Segment goes to windower.process_segment()
        assert mock_windower.process_segment.called
        segment = mock_windower.process_segment.call_args[0][0]

        # Hard cut: all data, no right_context
        expected_data = np.concatenate(all_chunks_audio[:94])
        assert np.array_equal(segment.data, expected_data), \
            "segment.data should contain all chunks 0-93 in order (hard cut)"

        # Verify right_context: empty (no silence boundary)
        assert len(segment.right_context) == 0, \
            f"right_context should be empty for hard cut, got {len(segment.right_context)} samples"

        assert len(segment.chunk_ids) == 94, \
            f"Expected 94 chunk_ids, got {len(segment.chunk_ids)}"
        assert segment.chunk_ids == list(range(94)), \
            f"Expected chunk_ids [0-93], got {segment.chunk_ids[:5]}...{segment.chunk_ids[-5:]}"


    def test_normal_finalization_with_mixed_chunks(self, preprocessor_config, mock_windower):
        """Normal finalization: data includes both speech AND silence chunks.

        Pattern: 5 speech → 1 silence → 7 speech → 3 silence (total 16 chunks)
        Finalization triggers after 2nd final silence (energy=1.8 > 1.5).

        Flow:
        - chunks 0-2: confirm speech → speech_buffer initialized with chunk_ids 0-2
        - chunks 3-4: speech → chunk_ids 3-4
        - chunk 5: silence → ACCUMULATING_SILENCE, silence_start_idx=5, chunk_id 5
        - chunks 6-12: speech → ACTIVE_SPEECH, silence_start_idx reset, chunk_ids 6-12
        - chunk 13: silence → ACCUMULATING_SILENCE, silence_start_idx=13, chunk_id 13
        - chunk 14: silence → energy=1.8 > 1.5 → FINALIZE (chunk 15 never processed)

        With silence_start_idx=13:
        - data = chunks [0-13] (14 chunks: all up to and including breakpoint)
        - right_context = chunk [14] (1 silence chunk after breakpoint)
        - chunk_ids = [0-13] (14 IDs)
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

        # Segment goes to windower.process_segment()
        assert mock_windower.process_segment.called
        segment = mock_windower.process_segment.call_args[0][0]

        # Verify data: 14 chunks (all up to and including breakpoint at chunk 13)
        expected_data = np.concatenate(all_chunks_audio[:14])
        assert np.array_equal(segment.data, expected_data), \
            "segment.data should contain exactly chunks 0-13 in order"

        # Verify right_context: 1 silence chunk after breakpoint (chunk 14)
        expected_right_context = all_chunks_audio[14]
        assert np.array_equal(segment.right_context, expected_right_context), \
            "right_context should contain exactly chunk 14"

        # Verify chunk_ids: [0-13] (14 IDs)
        assert len(segment.chunk_ids) == 14, \
            f"Expected 14 chunk_ids, got {len(segment.chunk_ids)}"
        assert segment.chunk_ids == list(range(14)), \
            f"Expected chunk_ids [0-13], got {segment.chunk_ids}"

        assert 14 not in segment.chunk_ids, \
            "chunk_id 14 should NOT be in segment.chunk_ids (belongs to right_context)"


def test_pause_state_flushes_segments():
    """Test that pausing triggers flush() to finalize pending segments.

    Logic: state changes to 'paused' → flush() called → pending segments emitted.
    """
    from unittest.mock import Mock
    import queue

    # Setup
    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    mock_vad = Mock()
    mock_vad.process_frame.return_value = {'speech_probability': 0.9, 'is_speech': True}
    mock_windower = Mock()
    mock_app_state = Mock()

    config = {
        'audio': {
            'sample_rate': 16000,
            'chunk_duration': 0.032,
            'silence_energy_threshold': 1.5,
            'rms_normalization': {
                'target_rms': 0.05,
                'silence_threshold': 0.001,
                'gain_smoothing': 0.8
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

    preprocessor = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad,
        windower=mock_windower,
        config=config,
        app_state=mock_app_state,
        verbose=False
    )

    # Add some speech chunks to create a pending segment
    for i in range(5):
        chunk = {
            'audio': np.random.randn(512).astype(np.float32) * 0.1,
            'timestamp': float(i * 0.032)
        }
        preprocessor._process_chunk(chunk)

    # Simulate state change to paused
    preprocessor.on_state_change('running', 'paused')

    # flush() passes segment to windower.flush()
    assert mock_windower.flush.called, "flush() should call windower.flush()"


def test_sound_preprocessor_shutdown_calls_flush():
    """Test that shutdown via observer calls flush() to emit pending segments.

    Logic: on_state_change(_, 'shutdown') should call stop(), which calls flush()
           to ensure pending segments are emitted before shutdown.
    """
    from unittest.mock import Mock
    import queue

    chunk_queue = queue.Queue()
    speech_queue = queue.Queue()
    mock_vad = Mock()
    mock_vad.process_frame.return_value = {'speech_probability': 0.9, 'is_speech': True}
    mock_windower = Mock()
    mock_app_state = Mock()

    config = {
        'audio': {
            'sample_rate': 16000,
            'chunk_duration': 0.032,
            'silence_energy_threshold': 1.5,
            'rms_normalization': {
                'target_rms': 0.05,
                'silence_threshold': 0.001,
                'gain_smoothing': 0.8
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

    preprocessor = SoundPreProcessor(
        chunk_queue=chunk_queue,
        speech_queue=speech_queue,
        vad=mock_vad,
        windower=mock_windower,
        config=config,
        app_state=mock_app_state,
        verbose=False
    )
    preprocessor.is_running = True

    # Add some speech chunks to create a pending segment
    for i in range(5):
        chunk = {
            'audio': np.random.randn(512).astype(np.float32) * 0.1,
            'timestamp': float(i * 0.032)
        }
        preprocessor._process_chunk(chunk)

    preprocessor.on_state_change('running', 'shutdown')

    # flush() passes segment to windower.flush()
    assert mock_windower.flush.called, "stop() should call flush() which calls windower.flush()"
    assert preprocessor.is_running == False

