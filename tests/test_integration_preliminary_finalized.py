# tests/test_integration_preliminary_finalized.py
import pytest
import queue
import numpy as np
import time
from unittest.mock import Mock
from src.SoundPreProcessor import SoundPreProcessor
from src.AdaptiveWindower import AdaptiveWindower
from src.Recognizer import Recognizer
from src.types import AudioSegment, RecognitionResult
from onnx_asr.asr import TimestampedResult


class TestPreliminaryFinalizedIntegration:
    """Integration tests for preliminary and finalized recognition flow.

    Tests the complete pipeline: SoundPreProcessor → AdaptiveWindower → Recognizer,
    verifying that both preliminary (instant) and finalized (high-quality) results
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
                'window_duration': 3.0,
                'max_speech_duration_ms': 3000,
                'silence_timeout': 0.5
            }
        }

    def test_preprocessor_windower_integration(self, config):
        """SoundPreProcessor should emit preliminary segments and call windower for finalized windows."""
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

        # Verify preliminary segments in speech_queue
        assert not speech_queue.empty()
        preliminary_segment = speech_queue.get()
        assert isinstance(preliminary_segment, AudioSegment)
        assert preliminary_segment.type == 'preliminary'

        # Verify windower.process_segment was called with preliminary segments
        assert mock_windower.process_segment.called
        for call in mock_windower.process_segment.call_args_list:
            segment = call[0][0]
            assert isinstance(segment, AudioSegment)
            assert segment.type == 'preliminary'


    def test_windower_produces_finalized_windows(self, config):
        """AdaptiveWindower should aggregate preliminary segments into finalized windows."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        # Create preliminary word segments
        for i in range(15):  # 15 words across 3 seconds
            start = i * 0.2
            end = start + 0.15
            segment = AudioSegment(
                type='preliminary',
                data=np.random.randn(int(0.15 * 16000)).astype(np.float32) * 0.1,
                left_context=np.array([], dtype=np.float32),
                right_context=np.array([], dtype=np.float32),
                start_time=start,
                end_time=end,
                chunk_ids=[i]
            )
            windower.process_segment(segment)

        windower.flush()

        # Should produce flush windows (since we called flush())
        assert not speech_queue.empty()
        window = speech_queue.get()
        assert isinstance(window, AudioSegment)
        assert window.type == 'flush'
        assert len(window.chunk_ids) > 1  # Aggregated multiple chunks


    def test_recognizer_handles_both_types(self):
        """Recognizer should handle both preliminary and finalized segments correctly."""
        mock_model = Mock()
        mock_model.recognize.side_effect = [
            TimestampedResult(text="instant result", tokens=None, timestamps=None),
            TimestampedResult(text="quality result", tokens=None, timestamps=None)
        ]

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model, sample_rate=16000)

        # Create preliminary and finalized segments
        preliminary = AudioSegment(
            type='preliminary',
            data=np.random.randn(3200).astype(np.float32) * 0.1,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=0.2,
            chunk_ids=[0]
        )
        finalized = AudioSegment(
            type='finalized',
            data=np.random.randn(48000).astype(np.float32) * 0.1,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=3.0,
            chunk_ids=[0, 1, 2, 3, 4]
        )

        # Process both segments
        result1 = recognizer.recognize_window(preliminary)
        result2 = recognizer.recognize_window(finalized)

        # Verify both results with correct status
        assert isinstance(result1, RecognitionResult)
        assert result1.text == "instant result"
        assert result1.status == 'preliminary'

        assert isinstance(result2, RecognitionResult)
        assert result2.text == "quality result"
        assert result2.status == 'final'


    def test_full_pipeline_flow(self, config):
        """Test complete flow: SoundPreProcessor → AdaptiveWindower → Recognizer."""
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
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

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
        recognizer = Recognizer(speech_queue, text_queue, mock_model, sample_rate=16000)

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

        # Recognize each segment
        results = []
        for segment in segments_to_process:
            result = recognizer.recognize_window(segment)
            if result is not None:
                results.append(result)

        # Verify we got recognition results
        assert len(results) > 0

        # Check that we have both preliminary and finalized/flush results
        preliminary_count = 0
        finalized_count = 0
        flush_count = 0

        for result in results:
            assert isinstance(result, RecognitionResult)
            if result.status == 'preliminary':
                preliminary_count += 1
            elif result.status == 'final':
                finalized_count += 1
            elif result.status == 'flush':
                flush_count += 1

        # Should have at least one of each type (flush is like finalized)
        assert preliminary_count >= 1, "Should have preliminary results"
        assert (finalized_count + flush_count) >= 1, "Should have finalized or flush results"


    def test_timing_preservation_through_pipeline(self, config):
        """Timing information should be preserved through entire pipeline."""
        mock_model = Mock()
        mock_model.recognize.return_value = TimestampedResult(
            text="test",
            tokens=None,
            timestamps=None
        )

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model, sample_rate=16000)

        # Create segment with specific timing
        segment = AudioSegment(
            type='finalized',
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
