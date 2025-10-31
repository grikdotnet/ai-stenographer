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
            }
            preprocessor._process_chunk(chunk)

        assert len(preprocessor.speech_buffer) == 3


    def test_preliminary_segment_emission_on_silence(self, preprocessor_config, mock_windower):
        """Silence after speech should emit preliminary AudioSegment.

        Logic: 3 speech chunks → silence chunk → preliminary segment in queue.

        Segment audio will include trailing silence chunks.
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
            }
            preprocessor._process_chunk(chunk)

        # Should have emitted preliminary segment
        assert not speech_queue.empty()
        segment = speech_queue.get()

        assert isinstance(segment, AudioSegment)
        assert segment.type == 'preliminary'
        # Audio data should include trailing chunks
        # 3 speech + 2 trailing silence = 5 chunks = 2560 samples
        assert len(segment.data) == 512 * 5


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
            }
            preprocessor._process_chunk(chunk)

        # Expected energy: (1-0.4) + (1-0.3) = 0.6 + 0.7 = 1.3
        expected_energy = 1.3
        assert abs(preprocessor.silence_energy - expected_energy) < 0.01


    def test_segment_finalization_at_energy_threshold(self, preprocessor_config, mock_windower):
        """Segment should finalize when silence energy >= threshold.

        Logic: threshold=1.5, silence probs accumulate until >= 1.5.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # VAD: speech, then silence chunks
        # Need to accumulate to >= 1.5
        # Using prob=0.25: (1-0.25) + (1-0.25) = 0.75 + 0.75 = 1.5
        probabilities = [0.9, 0.9, 0.25, 0.25]
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
        for i in range(4):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

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
            }
            preprocessor._process_chunk(chunk)

        # Feed silence chunks for > 0.5s (silence_timeout)
        # 0.5s / 0.032s = ~16 chunks
        for i in range(20):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.01,
                'timestamp': 1.0 + (2 + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

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


    def test_last_silence_chunk_stored(self, preprocessor_config, mock_windower):
        """Verify that silence chunks are stored in last_silence_chunk variable.

        Logic: Feed 2 silence chunks → last_silence_chunk should contain the second chunk.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD to return silence for first 2 chunks
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

        # Feed 2 silence chunks
        chunks = []
        for i in range(2):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            chunks.append(chunk)
            preprocessor._process_chunk(chunk)

        # Verify last_silence_chunk is stored
        assert preprocessor.last_silence_chunk is not None
        assert 'audio' in preprocessor.last_silence_chunk
        assert 'timestamp' in preprocessor.last_silence_chunk
        # Should contain the second chunk's timestamp
        assert preprocessor.last_silence_chunk['timestamp'] == chunks[1]['timestamp']


    def test_last_silence_chunk_prepended_on_speech_start(self, preprocessor_config, mock_windower):
        """Verify that the last silence chunk is prepended when speech starts.

        Logic: silence, silence, speech, speech → speech_buffer has 3 items (1 prepended + 2 speech).
        First item in buffer matches last_silence_chunk.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: silence, silence, speech, speech
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
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

        # Feed 4 chunks (2 silence + 2 speech)
        silence_chunks = []
        for i in range(2):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            silence_chunks.append(chunk)
            preprocessor._process_chunk(chunk)

        for i in range(2):
            audio = np.random.randn(512).astype(np.float32) * 0.1
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + (2 + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        assert len(preprocessor.speech_buffer) == 3

        # First item should match last silence chunk
        assert preprocessor.speech_buffer[0]['timestamp'] == silence_chunks[1]['timestamp']
        assert np.array_equal(preprocessor.speech_buffer[0]['audio'], silence_chunks[1]['audio'])

        # 2 speech chunks should have chunk_ids (0, 1)
        chunk_ids = [item['chunk_id'] for item in preprocessor.speech_buffer if 'chunk_id' in item]
        assert chunk_ids == [0, 1]


    def test_silence_chunks_buffered_during_energy_accumulation(self, preprocessor_config, mock_windower):
        """Verify that silence chunks are buffered while counting energy.

        Logic: speech (3 chunks), then silence with prob=0.25
        silence_energy_threshold = 1.5
        Each silence adds (1.0 - 0.25) = 0.75
        Need 2 silence chunks: 0.75 + 0.75 = 1.5 >= 1.5
        speech_buffer should contain 3 speech chunks + 2 trailing silence chunks.
        AudioSegment's audio data includes trailing silence.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: speech (3 chunks), then silence with prob=0.25
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return {'is_speech': True, 'speech_probability': 0.9}
            else:
                return {'is_speech': False, 'speech_probability': 0.25}

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

        # Feed silence chunks until energy reaches threshold
        # (1.0 - 0.25) = 0.75 per chunk, need 2: 0.75 + 0.75 = 1.5
        for i in range(2):
            audio = np.random.randn(512).astype(np.float32) * 0.01
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + (3 + i) * 0.032,
            }
            preprocessor._process_chunk(chunk)

        assert not speech_queue.empty()
        segment = speech_queue.get()

        assert isinstance(segment, AudioSegment)
        assert segment.type == 'preliminary'

        assert len(segment.chunk_ids) == 3
        assert segment.chunk_ids == [0, 1, 2]

        # Audio data should include trailing silence chunks
        # 3 speech chunks (512 * 3) + 2 silence chunks (512 * 2) = 2560 samples
        expected_length = 512 * 5
        assert len(segment.data) == expected_length


    def test_no_prepend_if_no_previous_silence(self, preprocessor_config, mock_windower):
        """Verify no prepending if speech starts immediately (no prior silence chunk).

        Logic: speech, speech, speech (from start) → speech_buffer has exactly 3 items.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

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

        # Feed 3 speech chunks from start
        for i in range(3):
            audio = np.random.randn(512).astype(np.float32) * 0.1
            chunk = {
                'audio': audio,
                'timestamp': 1.0 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        assert len(preprocessor.speech_buffer) == 3


    def test_last_silence_chunk_cleared_after_prepend(self, preprocessor_config, mock_windower):
        """Verify last_silence_chunk is cleared after prepending to avoid reuse.

        Logic: silence, speech, silence (finalize), silence, speech (two speech segments)
        After first speech start: last_silence_chunk is None.
        After finalization and new silence: last_silence_chunk stores new chunk.
        Second speech segment prepends the NEW silence chunk.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: silence, speech, silence (prob=0.1 to trigger finalization), silence, speech
        sequence = [
            ('silence', 0.2),
            ('speech', 0.9),
            ('silence', 0.1),  # (1-0.1) = 0.9, not enough yet
            ('silence', 0.1),  # (1-0.1) = 0.9, total 1.8 >= 1.5, triggers finalization
            ('silence', 0.2),  # New silence after finalization
            ('speech', 0.9)    # Second speech segment
        ]
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            is_speech, prob = sequence[call_count]
            is_speech = (is_speech == 'speech')
            call_count += 1
            return {'is_speech': is_speech, 'speech_probability': prob}

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

        chunk1 = {
            'audio': np.random.randn(512).astype(np.float32) * 0.01,
            'timestamp': 1.0,
        }
        preprocessor._process_chunk(chunk1)
        assert preprocessor.last_silence_chunk is not None
        first_silence_timestamp = preprocessor.last_silence_chunk['timestamp']

        chunk2 = {
            'audio': np.random.randn(512).astype(np.float32) * 0.1,
            'timestamp': 1.032,
        }
        preprocessor._process_chunk(chunk2)

        assert preprocessor.last_silence_chunk is None

        # Feed silence chunks to finalize first segment
        for i in range(2):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.01,
                'timestamp': 1.064 + i * 0.032,
            }
            preprocessor._process_chunk(chunk)

        # First segment should be finalized
        assert not speech_queue.empty()

        # Feed new silence (after finalization)
        chunk5 = {
            'audio': np.random.randn(512).astype(np.float32) * 0.01,
            'timestamp': 1.128,
        }
        preprocessor._process_chunk(chunk5)
        assert preprocessor.last_silence_chunk is not None
        second_silence_timestamp = preprocessor.last_silence_chunk['timestamp']
        assert second_silence_timestamp != first_silence_timestamp

        # Feed second speech
        chunk6 = {
            'audio': np.random.randn(512).astype(np.float32) * 0.1,
            'timestamp': 1.160,
        }
        preprocessor._process_chunk(chunk6)

        # Second speech segment should prepend the NEW silence chunk
        assert len(preprocessor.speech_buffer) == 2
        assert preprocessor.speech_buffer[0]['timestamp'] == second_silence_timestamp
