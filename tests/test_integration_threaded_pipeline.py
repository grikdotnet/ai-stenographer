# tests/test_integration_threaded_pipeline.py
import pytest
import queue
import numpy as np
import time
from unittest.mock import Mock
from src.sound.AudioSource import AudioSource
from src.sound.SoundPreProcessor import SoundPreProcessor
from src.asr.VoiceActivityDetector import VoiceActivityDetector
from src.sound.GrowingWindowAssembler import GrowingWindowAssembler
from src.types import AudioSegment


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
                'window_duration': 3.0,
                'max_window_duration': 7.0,
                'max_speech_duration_ms': 3000,
                'silence_timeout': 0.5
            }
        }

    def test_raw_audio_flows_to_windower(self, pipeline_config):
        """Raw audio chunks should flow through SoundPreProcessor to windower.

        Logic: AudioSource puts raw chunks → SoundPreProcessor processes → windower receives AudioSegments.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD
        mock_vad = Mock()
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            # First 3 are speech, then silence
            if call_count <= 3:
                return {'is_speech': True, 'speech_probability': 0.9}
            else:
                return {'is_speech': False, 'speech_probability': 0.1}

        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        # Mock windower
        mock_windower = Mock()
        mock_windower.process_segment = Mock()
        mock_windower.flush = Mock()

        # Create SoundPreProcessor
        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=pipeline_config,
            verbose=False
        )

        # Start preprocessor thread
        preprocessor.start()

        try:
            # Simulate AudioSource putting raw chunks
            for i in range(5):
                chunk_data = {
                    'audio': np.random.randn(512).astype(np.float32) * 0.1,
                    'timestamp': 1.0 + i * 0.032,
                    'chunk_id': i
                }
                chunk_queue.put(chunk_data)

            # Give preprocessor time to process
            time.sleep(0.5)

            # Windower should have received segments
            assert mock_windower.process_segment.called or mock_windower.flush.called
            if mock_windower.process_segment.called:
                segment = mock_windower.process_segment.call_args[0][0]
                assert isinstance(segment, AudioSegment)
                assert segment.type == 'incremental'
                assert len(segment.data) > 0

        finally:
            preprocessor.stop()


    def test_incremental_segments_in_speech_queue(self, pipeline_config):
        """speech_queue should contain incremental segments from GrowingWindowAssembler.

        Logic: Process speech → silence → incremental/flush segments appear in speech_queue.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD
        mock_vad = Mock()
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return {'is_speech': True, 'speech_probability': 0.9}
            else:
                return {'is_speech': False, 'speech_probability': 0.1}

        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        # Real GrowingWindowAssembler
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
            verbose=False
        )

        preprocessor.start()

        try:
            # Feed chunks
            for i in range(6):
                chunk_data = {
                    'audio': np.random.randn(512).astype(np.float32) * 0.1,
                    'timestamp': 1.0 + i * 0.032,
                    'chunk_id': i
                }
                chunk_queue.put(chunk_data)

            # Wait for processing
            time.sleep(0.5)

            segments = []
            while not speech_queue.empty():
                try:
                    segments.append(speech_queue.get(timeout=0.1))
                except queue.Empty:
                    break

            assert len(segments) > 0

            for s in segments:
                assert s.type in ('incremental', 'flush'), f"Unexpected segment type: {s.type}"

        finally:
            preprocessor.stop()


    def test_preprocessor_delegates_to_windower(self, pipeline_config):
        """SoundPreProcessor should delegate segments to windower (not directly to speech_queue).

        Logic: Verify SoundPreProcessor calls windower.process_segment() with AudioSegments.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()

        # Mock VAD: speech then silence
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

        # Mock windower
        mock_windower = Mock()
        mock_windower.process_segment = Mock()
        mock_windower.flush = Mock()

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=pipeline_config,
            verbose=False
        )

        preprocessor.start()

        try:
            # Feed chunks (3 speech + 2 silence to trigger finalization)
            for i in range(5):
                chunk_data = {
                    'audio': np.random.randn(512).astype(np.float32) * 0.1,
                    'timestamp': 1.0 + i * 0.032,
                    'chunk_id': i
                }
                chunk_queue.put(chunk_data)

            # Wait for processing
            time.sleep(0.5)

            # Windower should have received segments
            assert mock_windower.process_segment.called or mock_windower.flush.called
            for call in mock_windower.process_segment.call_args_list:
                segment = call[0][0]
                assert isinstance(segment, AudioSegment)

        finally:
            preprocessor.stop()
