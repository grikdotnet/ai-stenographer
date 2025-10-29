# tests/test_sound_preprocessor.py
import pytest
import queue
import numpy as np
import time
from unittest.mock import Mock
from src.SoundPreProcessor import SoundPreProcessor
from src.types import AudioSegment


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
            'timestamp': 1.0,
            'chunk_id': 1
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
        """Mock VAD that returns configurable speech probabilities"""
        mock = Mock()
        mock.process_frame = Mock(return_value={'is_speech': True, 'speech_probability': 0.9})
        return mock

    @pytest.fixture
    def mock_windower(self):
        """Mock AdaptiveWindower to verify synchronous calls"""
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
        chunk = {'audio': audio, 'timestamp': 1.0, 'chunk_id': 0}

        original_rms = np.sqrt(np.mean(audio**2))

        # Process chunk (synchronous call for testing)
        preprocessor._process_chunk(chunk)

        # Verify VAD was called with normalized audio (not original)
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
        chunk = {'audio': audio, 'timestamp': 1.0, 'chunk_id': 0}

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
        chunk = {'audio': raw_audio.copy(), 'timestamp': 1.0, 'chunk_id': 0}

        preprocessor._process_chunk(chunk)

        # VAD should have been called
        assert mock_vad.process_frame.called

        # Get the audio passed to VAD
        vad_audio = mock_vad.process_frame.call_args[0][0]

        # VAD audio should be normalized (different from raw)
        raw_rms = np.sqrt(np.mean(raw_audio**2))
        vad_rms = np.sqrt(np.mean(vad_audio**2))

        assert vad_rms > raw_rms  # Normalized should be boosted


    def test_speech_frame_buffering(self, preprocessor_config, mock_vad, mock_windower):
        """Speech frames should accumulate in buffer.

        Logic: 3 speech frames → buffer should contain all 3.
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

        # Feed 3 speech chunks
        for i in range(3):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
                'chunk_id': i
            }
            preprocessor._process_chunk(chunk)

        # Buffer should contain 3 chunks
        assert len(preprocessor.speech_buffer) == 3


    def test_preliminary_segment_emission_on_silence(self, preprocessor_config, mock_windower):
        """Silence after speech should emit preliminary AudioSegment.

        Logic: 3 speech chunks → silence chunk → preliminary segment in queue.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: speech for first 3, silence for 4th
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
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
                'chunk_id': i
            }
            preprocessor._process_chunk(chunk)

        # Feed silence chunks until threshold reached
        # silence_energy_threshold = 1.5
        # Each silence chunk with prob=0.1 adds (1-0.1) = 0.9
        # Need 2 chunks: 0.9 + 0.9 = 1.8 >= 1.5
        for i in range(2):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.01,
                'timestamp': 1.0 + (3 + i) * 0.032,
                'chunk_id': 3 + i
            }
            preprocessor._process_chunk(chunk)

        # Should have emitted preliminary segment
        assert not speech_queue.empty()
        segment = speech_queue.get()

        assert isinstance(segment, AudioSegment)
        assert segment.type == 'preliminary'
        assert len(segment.chunk_ids) == 3  # IDs [0, 1, 2]


    def test_silence_energy_accumulation(self, preprocessor_config, mock_windower):
        """Silence energy should accumulate as (1.0 - speech_prob).

        Logic: Probabilities [0.4, 0.3] → energy = 0.6 + 0.7 = 1.3 (below threshold, not finalized yet)
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # VAD sequence: speech, then silence with decreasing probabilities
        probabilities = [0.9, 0.4, 0.3]
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            prob = probabilities[call_count]
            call_count += 1
            return {'is_speech': prob > 0.5, 'speech_probability': prob}

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

        # Feed 3 chunks
        for i in range(3):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
                'chunk_id': i
            }
            preprocessor._process_chunk(chunk)

        # Expected energy: (1-0.4) + (1-0.3) = 0.6 + 0.7 = 1.3
        # threshold is 1.5, so NOT finalized yet, energy should still be accumulating
        expected_energy = 1.3
        assert abs(preprocessor.silence_energy - expected_energy) < 0.01


    def test_segment_finalization_at_energy_threshold(self, preprocessor_config, mock_windower):
        """Segment should finalize when silence energy >= threshold.

        Logic: threshold=1.5, silence probs accumulate until >= 1.5.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # VAD: speech, then silence chunks
        probabilities = [0.9, 0.9, 0.3, 0.3]  # (1-0.3) + (1-0.3) = 1.4, need one more
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            prob = probabilities[min(call_count, len(probabilities) - 1)]
            call_count += 1
            return {'is_speech': prob > 0.5, 'speech_probability': prob}

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

        # Feed chunks until threshold reached
        for i in range(5):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
                'chunk_id': i
            }
            preprocessor._process_chunk(chunk)

        # Should have finalized segment (energy >= 1.5)
        assert not speech_queue.empty()


    def test_max_speech_duration_forces_split(self, preprocessor_config, mock_vad, mock_windower):
        """Continuous speech exceeding max duration should force segment split.

        Logic: max_speech_duration_ms=3000, chunk=32ms → 3000/32 = 93.75 chunks.
        After 94 chunks, should force split.
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

        # Feed 95 speech chunks (exceeds 3000ms)
        num_chunks = 95
        for i in range(num_chunks):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
                'chunk_id': i
            }
            preprocessor._process_chunk(chunk)

        # Should have emitted at least one segment due to max duration
        assert not speech_queue.empty()


    def test_adaptive_windower_called_synchronously(self, preprocessor_config, mock_windower):
        """AdaptiveWindower.process_segment() should be called with preliminary segment.

        Logic: Verify windower receives the preliminary AudioSegment.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # VAD: speech, then silence
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

        # Feed speech + silence
        for i in range(5):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
                'chunk_id': i
            }
            preprocessor._process_chunk(chunk)

        # Windower should have been called
        assert mock_windower.process_segment.called
        segment = mock_windower.process_segment.call_args[0][0]
        assert isinstance(segment, AudioSegment)
        assert segment.type == 'preliminary'


    def test_silence_timeout_triggers_windower_flush(self, preprocessor_config, mock_windower):
        """Silence duration >= timeout should trigger windower.flush().

        Logic: speech → silence > 0.5s → flush windower.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # VAD: speech, then silence
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
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

        # Feed 2 speech chunks
        for i in range(2):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
                'chunk_id': i
            }
            preprocessor._process_chunk(chunk)

        # Feed silence chunks for > 0.5s (silence_timeout)
        # 0.5s / 0.032s = ~16 chunks
        for i in range(20):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.01,
                'timestamp': 1.0 + (2 + i) * 0.032,
                'chunk_id': 2 + i
            }
            preprocessor._process_chunk(chunk)

        # Windower.flush() should have been called
        assert mock_windower.flush.called


    def test_flush_emits_pending_segment(self, preprocessor_config, mock_vad, mock_windower):
        """flush() should emit pending segment and call windower.flush().

        Logic: Active buffer with 2 chunks → flush() → segment emitted + windower flushed.
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

        # Feed 2 speech chunks
        for i in range(2):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
                'chunk_id': i
            }
            preprocessor._process_chunk(chunk)

        # Manually flush
        preprocessor.flush()

        # Should have emitted preliminary segment
        assert not speech_queue.empty()
        segment = speech_queue.get()
        assert segment.type == 'preliminary'

        # Windower should have been flushed
        assert mock_windower.flush.called

        # Buffer should be cleared
        assert len(preprocessor.speech_buffer) == 0


    def test_chunk_id_tracking_in_segments(self, preprocessor_config, mock_windower):
        """AudioSegment.chunk_ids should track chunk IDs correctly.

        Logic: Chunks with IDs [10, 11, 12] → segment.chunk_ids == [10, 11, 12].
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # VAD: speech, then silence
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

        # Feed chunks with IDs [10, 11, 12]
        for i, chunk_id in enumerate([10, 11, 12]):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
                'chunk_id': chunk_id
            }
            preprocessor._process_chunk(chunk)

        # Feed silence to trigger finalization
        for i in range(2):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.01,
                'timestamp': 1.0 + (3 + i) * 0.032,
                'chunk_id': 13 + i
            }
            preprocessor._process_chunk(chunk)

        # Check emitted segment
        assert not speech_queue.empty()
        segment = speech_queue.get()
        assert segment.chunk_ids == [10, 11, 12]


    def test_timestamp_accuracy(self, preprocessor_config, mock_windower):
        """AudioSegment timestamps should accurately reflect chunk timestamps.

        Logic: Chunks at [1.0, 1.032, 1.064] → start_time=1.0, end_time≈1.096.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # VAD: speech, then silence
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

        # Feed 3 chunks with precise timestamps
        timestamps = [1.0, 1.032, 1.064]
        for i, ts in enumerate(timestamps):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': ts,
                'chunk_id': i
            }
            preprocessor._process_chunk(chunk)

        # Feed silence
        for i in range(2):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.01,
                'timestamp': 1.096 + i * 0.032,
                'chunk_id': 3 + i
            }
            preprocessor._process_chunk(chunk)

        # Check timestamps
        assert not speech_queue.empty()
        segment = speech_queue.get()

        assert segment.start_time == 1.0
        # end_time = last_timestamp + chunk_duration
        # 3 chunks of 512 samples @ 16kHz = 3 * 0.032 = 0.096s
        expected_end = 1.0 + 0.096
        assert abs(segment.end_time - expected_end) < 0.001


    def test_threading_starts_and_stops_cleanly(self, preprocessor_config, mock_vad, mock_windower):
        """SoundPreProcessor thread should start and stop without errors.

        Logic: start() → thread runs → stop() → thread terminates cleanly.
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

        # Start thread
        preprocessor.start()
        assert preprocessor.is_running
        assert preprocessor.thread is not None
        assert preprocessor.thread.is_alive()

        # Give thread time to start
        time.sleep(0.1)

        # Stop thread
        preprocessor.stop()
        assert not preprocessor.is_running

        # Thread should terminate
        time.sleep(0.2)
        assert not preprocessor.thread.is_alive()
