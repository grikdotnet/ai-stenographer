# tests/test_integration_preliminary_finalized.py
import pytest
import queue
import numpy as np
import time
from unittest.mock import Mock
from src.sound.SoundPreProcessor import SoundPreProcessor
from src.sound.GrowingWindowAssembler import GrowingWindowAssembler
from src.asr.Recognizer import Recognizer
from src.types import AudioSegment, RecognitionResult, SpeechEndSignal
from onnx_asr.asr import TimestampedResult


class TestIncrementalFlushIntegration:
    """Integration tests for incremental and flush recognition flow.

    Tests the complete pipeline: SoundPreProcessor → GrowingWindowAssembler → Recognizer,
    verifying that incremental (growing window) and flush (end-of-speech) results
    are produced correctly.
    """

    @pytest.fixture
    def config(self):
        """Standard configuration for integration tests."""
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
                'threshold': 0.5,
                'frame_duration_ms': 32
            },
            'windowing': {
                'max_window_duration': 7.0,
                'max_speech_duration_ms': 3000
            }
        }

    def test_preprocessor_windower_integration(self, config):
        """SoundPreProcessor should call windower with incremental segments."""
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()
        mock_windower = Mock()

        # Mock VAD that returns speech for first 30 chunks, then silence
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            if call_count <= 30:
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
            config=config
        )

        # Feed raw audio chunks (simulating AudioSource)
        chunk_size = 512
        for i in range(50):  # Feed 50 chunks (30 speech + 20 silence)
            chunk_data = {
                'audio': np.random.randn(chunk_size).astype(np.float32) * 0.1,
                'timestamp': i * 0.032,
                'chunk_id': i
            }
            preprocessor._process_chunk(chunk_data)

        preprocessor.flush()

        # Verify windower.process_segment was called with incremental segments
        assert mock_windower.process_segment.called or mock_windower.flush.called
        for call in mock_windower.process_segment.call_args_list:
            segment = call[0][0]
            assert isinstance(segment, AudioSegment)
            assert segment.type == 'incremental'


    def test_windower_produces_incremental_windows(self, config):
        """GrowingWindowAssembler should emit incremental windows on flush."""
        speech_queue = queue.Queue()
        windower = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        # Create incremental word segments
        for i in range(15):  # 15 words across 3 seconds
            start = i * 0.2
            end = start + 0.15
            segment = AudioSegment(
                type='incremental',
                data=np.random.randn(int(0.15 * 16000)).astype(np.float32) * 0.1,
                left_context=np.array([], dtype=np.float32),
                right_context=np.array([], dtype=np.float32),
                start_time=start,
                end_time=end,
                chunk_ids=[i]
            )
            windower.process_segment(segment)

        windower.flush()

        # Drain all windows from queue
        windows = []
        while not speech_queue.empty():
            windows.append(speech_queue.get())

        # Should have incremental windows
        assert len(windows) > 1

        for w in windows[:-1]:
            assert w.type == 'incremental'
        assert windows[-1].type == 'incremental'
        assert len(windows[-1].chunk_ids) > 1


    def test_recognizer_handles_incremental_segments(self):
        """Recognizer should produce RecognitionResult from incremental segments."""
        mock_model = Mock()
        mock_model.recognize.side_effect = [
            TimestampedResult(text="instant result", tokens=None, timestamps=None),
            TimestampedResult(text="quality result", tokens=None, timestamps=None)
        ]

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model, sample_rate=16000, app_state=Mock())

        seg1 = AudioSegment(
            type='incremental',
            data=np.random.randn(3200).astype(np.float32) * 0.1,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=0.2,
            chunk_ids=[0]
        )
        seg2 = AudioSegment(
            type='incremental',
            data=np.random.randn(48000).astype(np.float32) * 0.1,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=3.0,
            chunk_ids=[0, 1, 2, 3, 4]
        )

        result1 = recognizer.recognize_window(seg1)
        result2 = recognizer.recognize_window(seg2)

        assert isinstance(result1, RecognitionResult)
        assert result1.text == "instant result"

        assert isinstance(result2, RecognitionResult)
        assert result2.text == "quality result"


    def test_full_pipeline_flow(self, config):
        """Test complete flow: SoundPreProcessor → GrowingWindowAssembler → Recognizer."""
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue()
        text_queue = queue.Queue()

        # Mock VAD that returns speech for most chunks
        call_count = 0
        def vad_side_effect(audio):
            nonlocal call_count
            call_count += 1
            # Speech for chunks, then some silence
            if call_count <= 100:
                return {'is_speech': True, 'speech_probability': 0.9}
            else:
                return {'is_speech': False, 'speech_probability': 0.1}

        mock_vad = Mock()
        mock_vad.process_frame = Mock(side_effect=vad_side_effect)

        # Set up windower
        windower = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        # Set up preprocessor with windower
        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=windower,
            config=config
        )

        # Set up recognizer with mock model
        mock_model = Mock()
        mock_model.recognize.return_value = TimestampedResult(
            text="recognized text",
            tokens=None,
            timestamps=None
        )
        recognizer = Recognizer(input_queue=speech_queue, output_queue=text_queue, model=mock_model, sample_rate=16000, app_state=Mock())

        # Feed raw audio chunks - use 4 seconds to ensure finalized window
        chunk_size = 512
        num_chunks = int((4.0 / 0.032))  # 4 seconds @ 32ms per chunk
        for i in range(num_chunks):
            chunk_data = {
                'audio': np.random.randn(chunk_size).astype(np.float32) * 0.1,
                'timestamp': i * 0.032,
                'chunk_id': i
            }
            preprocessor._process_chunk(chunk_data)

        # Flush to complete pipeline
        preprocessor.flush()

        # Process recognition queue
        segments_to_process = []
        while not speech_queue.empty():
            segments_to_process.append(speech_queue.get())

        # Recognize each audio segment (skip SpeechEndSignal boundaries)
        results = []
        for segment in segments_to_process:
            if isinstance(segment, SpeechEndSignal):
                continue
            result = recognizer.recognize_window(segment)
            if result is not None:
                results.append(result)

        # Verify we got recognition results
        assert len(results) > 0

        for result in results:
            assert isinstance(result, RecognitionResult)


    def test_timing_preservation_through_pipeline(self, config):
        """Timing information should be preserved through entire pipeline."""
        mock_model = Mock()
        mock_model.recognize.return_value = TimestampedResult(
            text="test",
            tokens=None,
            timestamps=None
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model, sample_rate=16000, app_state=Mock())

        # Create segment with specific timing
        segment = AudioSegment(
            type='incremental',
            data=np.random.randn(48000).astype(np.float32) * 0.1,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.234,
            end_time=4.567,
            chunk_ids=[0, 1, 2]
        )

        # Process segment
        result = recognizer.recognize_window(segment)

        # Check timing preserved
        assert result.start_time == 1.234
        assert result.end_time == 4.567
