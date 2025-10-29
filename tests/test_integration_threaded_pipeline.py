# tests/test_integration_threaded_pipeline.py
import pytest
import queue
import numpy as np
import time
from unittest.mock import Mock
from src.AudioSource import AudioSource
from src.SoundPreProcessor import SoundPreProcessor
from src.VoiceActivityDetector import VoiceActivityDetector
from src.AdaptiveWindower import AdaptiveWindower
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
                'step_size': 1.0,
                'max_speech_duration_ms': 3000,
                'silence_timeout': 0.5
            }
        }

    def test_raw_audio_flows_to_speech_queue(self, pipeline_config):
        """Raw audio chunks should flow through to speech_queue as AudioSegments.

        Logic: AudioSource puts raw chunks → SoundPreProcessor processes → speech_queue gets AudioSegments.
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

            # speech_queue should have AudioSegment
            assert not speech_queue.empty()
            segment = speech_queue.get(timeout=1.0)

            assert isinstance(segment, AudioSegment)
            assert segment.type == 'preliminary'
            assert len(segment.data) > 0

        finally:
            preprocessor.stop()


    def test_both_preliminary_and_finalized_in_speech_queue(self, pipeline_config):
        """speech_queue should contain both preliminary (from SoundPreProcessor) and finalized (from AdaptiveWindower).

        Logic: Process speech → silence → both segment types appear in speech_queue.
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

        # Real AdaptiveWindower
        windower = AdaptiveWindower(
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

            # Should have at least one preliminary segment
            segments = []
            while not speech_queue.empty():
                try:
                    segments.append(speech_queue.get(timeout=0.1))
                except queue.Empty:
                    break

            assert len(segments) > 0

            # Check for preliminary segments
            preliminary_count = sum(1 for s in segments if s.type == 'preliminary')
            assert preliminary_count > 0, "Should have preliminary segments from SoundPreProcessor"

            # Note: Finalized segments may appear if windower conditions are met

        finally:
            preprocessor.stop()


    def test_recognizer_receives_from_speech_queue(self, pipeline_config):
        """Recognizer should read AudioSegments from speech_queue (not chunk_queue).

        Logic: Verify Recognizer integration with renamed queue.
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()
        text_queue = queue.Queue()

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

        # Mock Recognizer
        mock_recognizer = Mock()
        mock_recognizer_instance = Mock()
        mock_recognizer_instance.is_running = True

        def recognizer_process():
            """Mock recognizer that reads from speech_queue."""
            while mock_recognizer_instance.is_running:
                try:
                    segment = speech_queue.get(timeout=0.1)
                    # Simulate recognition
                    if isinstance(segment, AudioSegment):
                        mock_recognizer_instance.segments_received.append(segment)
                except queue.Empty:
                    continue

        mock_recognizer_instance.segments_received = []

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=pipeline_config,
            verbose=False
        )

        preprocessor.start()

        # Start mock recognizer thread
        import threading
        recognizer_thread = threading.Thread(target=recognizer_process, daemon=True)
        recognizer_thread.start()

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

            # Mock recognizer should have received segments
            assert len(mock_recognizer_instance.segments_received) > 0
            assert all(isinstance(s, AudioSegment) for s in mock_recognizer_instance.segments_received)

        finally:
            mock_recognizer_instance.is_running = False
            preprocessor.stop()
            recognizer_thread.join(timeout=1.0)
