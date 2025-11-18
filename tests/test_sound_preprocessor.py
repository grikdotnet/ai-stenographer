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


    def test_context_buffer_fills_before_speech(self, preprocessor_config, mock_windower):
        """Verify context_buffer accumulates silence before speech starts.

        Logic: Feed 10 silence chunks → context_buffer should contain 10 chunks.
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

        # Context buffer should have 10 chunks
        assert len(preprocessor.context_buffer) == 10


    def test_left_context_extraction(self, preprocessor_config, mock_windower):
        """Verify left context is extracted from buffer when speech finalizes.

        Logic: 20 silence chunks, then 3 speech chunks, then finalize.
        left_context should contain approximately 20 chunks of silence.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= 20:
                return {'is_speech': False, 'speech_probability': 0.2}
            elif call_count <= 23:
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

        # Feed 20 silence chunks
        for i in range(20):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Feed 3 speech chunks
        for i in range(3):
            audio = np.random.randn(512).astype(np.float32) * 0.1
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + (20 + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Feed silence to finalize
        for i in range(2):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + (23 + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        assert not speech_queue.empty()
        segment = speech_queue.get()

        # Verify left_context contains silence (20 chunks * 512 samples)
        # May be less due to circular buffer overflow
        assert len(segment.left_context) > 0
        expected_samples = 20 * 512
        assert len(segment.left_context) <= expected_samples


    def test_right_context_extraction(self, preprocessor_config, mock_windower):
        """Verify right context is extracted from trailing silence.

        Logic: 3 speech chunks, then 2 silence chunks that finalize segment.
        right_context should contain the 2 silence chunks.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: 3  speech chunks, then silence to finalize
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

        # Feed 3 speech chunks
        for i in range(3):
            audio = np.random.randn(512).astype(np.float32) * 0.1
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Feed 2 silence chunks to finalize
        for i in range(2):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + (3 + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # Segment should be emitted
        assert not speech_queue.empty()
        segment = speech_queue.get()

        # Verify right_context contains the 2 silence chunks (2 * 512 samples)
        assert len(segment.right_context) == 2 * 512


    def test_no_left_context_at_startup(self, preprocessor_config, mock_windower):
        """Verify empty left context when speech starts immediately.

        Logic: Speech starts immediately (no preceding silence).
        left_context should be empty.
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

