# tests/test_integration_threaded_pipeline.py
import pytest
import queue
import numpy as np
from unittest.mock import Mock
from src.sound.AudioSource import AudioSource
from src.sound.SoundPreProcessor import SoundPreProcessor
from src.asr.VoiceActivityDetector import VoiceActivityDetector
from src.sound.GrowingWindowAssembler import GrowingWindowAssembler
from src.SpeechEndRouter import SpeechEndRouter
from src.types import AudioSegment, RecognizerAck, RecognizerFreeSignal, SpeechEndSignal


class TestIntegrationThreadedPipeline:
    """End-to-end integration tests for threaded audio pipeline.

    Tests the complete flow: AudioSource → chunk_queue → SoundPreProcessor → speech_queue.
    """

    @pytest.fixture
    def pipeline_config(self):
        """Configuration for full pipeline."""
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
                'model_path': './models/silero_vad/silero_vad.onnx',
                'frame_duration_ms': 32,
                'threshold': 0.5
            },
            'windowing': {
                'max_window_duration': 7.0,
                'max_speech_duration_ms': 3000
            }
        }

    def test_raw_audio_flows_to_windower(self, pipeline_config):
        """Raw audio chunks should flow through SoundPreProcessor to windower.

        Logic: AudioSource puts raw chunks → SoundPreProcessor processes → windower receives AudioSegments
        with the original (non-normalized) audio data.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: first 3 chunks are speech, then silence
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

        mock_windower = Mock()
        mock_windower.process_segment = Mock()
        mock_windower.flush = Mock()

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=pipeline_config,
            control_queue=queue.Queue(),
            verbose=False
        )

        sent_audio_chunks = []
        for i in range(5):
            audio_chunk = np.full(512, 0.01 * (i + 1), dtype=np.float32)
            sent_audio_chunks.append(audio_chunk.copy())
            preprocessor._process_chunk({
                'audio': audio_chunk,
                'timestamp': 1.0 + i * 0.032,
                'chunk_id': i
            })

        assert mock_windower.process_segment.called or mock_windower.flush.called
        if mock_windower.process_segment.called:
            segment = mock_windower.process_segment.call_args[0][0]
            assert isinstance(segment, AudioSegment)
            assert segment.type == 'incremental'
            expected_audio = np.concatenate(sent_audio_chunks)
            np.testing.assert_array_equal(segment.data, expected_audio)


    def test_incremental_segments_in_speech_queue(self, pipeline_config):
        """speech_queue should contain incremental segments from GrowingWindowAssembler.

        Logic: Process speech → silence → incremental/flush segments appear in speech_queue.
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

        windower = GrowingWindowAssembler(
            speech_queue=speech_queue,
            config=pipeline_config,
            verbose=False
        )

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=windower,
            config=pipeline_config,
            control_queue=queue.Queue(),
            verbose=False
        )

        for i in range(6):
            preprocessor._process_chunk({
                'audio': np.random.randn(512).astype(np.float32) * 0.1,
                'timestamp': 1.0 + i * 0.032,
                'chunk_id': i
            })

        segments = []
        while not speech_queue.empty():
            segments.append(speech_queue.get_nowait())

        assert len(segments) == 2
        assert isinstance(segments[0], AudioSegment)
        assert segments[0].type == 'incremental'
        assert isinstance(segments[1], SpeechEndSignal)


    def test_preprocessor_delegates_to_windower(self, pipeline_config):
        """SoundPreProcessor should delegate segments to windower (not directly to speech_queue).

        Logic: Verify SoundPreProcessor calls windower.process_segment() with AudioSegments.
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

        mock_windower = Mock()
        mock_windower.process_segment = Mock()
        mock_windower.flush = Mock()

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=pipeline_config,
            control_queue=queue.Queue(),
            verbose=False
        )

        sent_audio_chunks = []
        for i in range(5):
            audio_chunk = np.full(512, 0.01 * (i + 1), dtype=np.float32)
            sent_audio_chunks.append(audio_chunk.copy())
            preprocessor._process_chunk({
                'audio': audio_chunk,
                'timestamp': 1.0 + i * 0.032,
                'chunk_id': i
            })

        assert mock_windower.flush.call_count == 1
        segment = mock_windower.flush.call_args[0][0]
        assert isinstance(segment, AudioSegment)
        expected_audio = np.concatenate(sent_audio_chunks)
        np.testing.assert_array_equal(segment.data, expected_audio)

