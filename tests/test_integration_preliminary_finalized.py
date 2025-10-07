# tests/test_integration_preliminary_finalized.py
import pytest
import queue
import numpy as np
from unittest.mock import Mock
from src.AudioSource import AudioSource
from src.AdaptiveWindower import AdaptiveWindower
from src.Recognizer import Recognizer
from src.types import AudioSegment, RecognitionResult


class TestPreliminaryFinalizedIntegration:
    """Integration tests for preliminary and finalized recognition flow.

    Tests the complete pipeline: AudioSource → AdaptiveWindower → Recognizer,
    verifying that both preliminary (instant) and finalized (high-quality) results
    are produced correctly.
    """

    @pytest.fixture
    def config(self):
        """Standard configuration for integration tests."""
        return {
            'audio': {
                'sample_rate': 16000,
                'chunk_duration': 0.032
            },
            'vad': {
                'model_path': './models/silero_vad/silero_vad.onnx',
                'threshold': 0.5,
                'frame_duration_ms': 32,
                'min_speech_duration_ms': 64,
                'silence_timeout_ms': 32,
                'max_speech_duration_ms': 3000
            },
            'windowing': {
                'window_duration': 3.0,
                'step_size': 1.0
            }
        }

    def test_audiosource_windower_integration(self, config, vad, speech_audio):
        """AudioSource should emit preliminary segments and call windower for finalized windows."""
        chunk_queue = queue.Queue()
        mock_windower = Mock()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=vad,
            windower=mock_windower,
            config=config
        )

        # Feed audio chunks
        chunk_size = int(config['audio']['sample_rate'] * config['audio']['chunk_duration'])
        for i in range(0, len(speech_audio[:16000]), chunk_size):
            chunk = speech_audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                break
            audio_source.process_chunk_with_vad(chunk, i * config['audio']['chunk_duration'])

        audio_source.flush_vad()

        # Verify preliminary segments in queue
        assert not chunk_queue.empty()
        preliminary_segment = chunk_queue.get()
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
        chunk_queue = queue.Queue()
        windower = AdaptiveWindower(chunk_queue=chunk_queue, config=config)

        # Create preliminary word segments
        for i in range(15):  # 15 words across 3 seconds
            start = i * 0.2
            end = start + 0.15
            segment = AudioSegment(
                type='preliminary',
                data=np.random.randn(int(0.15 * 16000)).astype(np.float32) * 0.1,
                start_time=start,
                end_time=end,
                chunk_ids=[i]
            )
            windower.process_segment(segment)

        windower.flush()

        # Should produce finalized windows
        assert not chunk_queue.empty()
        window = chunk_queue.get()
        assert isinstance(window, AudioSegment)
        assert window.type == 'finalized'
        assert len(window.chunk_ids) > 1  # Aggregated multiple chunks


    def test_recognizer_handles_both_types(self):
        """Recognizer should handle both preliminary and finalized segments correctly."""
        mock_model = Mock()
        mock_model.recognize.side_effect = ["instant result", "quality result"]

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model)

        # Create preliminary and finalized segments
        preliminary = AudioSegment(
            type='preliminary',
            data=np.random.randn(3200).astype(np.float32) * 0.1,
            start_time=0.0,
            end_time=0.2,
            chunk_ids=[0]
        )
        finalized = AudioSegment(
            type='finalized',
            data=np.random.randn(48000).astype(np.float32) * 0.1,
            start_time=0.0,
            end_time=3.0,
            chunk_ids=[0, 1, 2, 3, 4]
        )

        # Process both segments
        result1 = recognizer.recognize_window(preliminary, is_preliminary=True)
        result2 = recognizer.recognize_window(finalized, is_preliminary=False)

        # Verify both results with correct flags
        assert isinstance(result1, RecognitionResult)
        assert result1.text == "instant result"
        assert result1.is_preliminary is True

        assert isinstance(result2, RecognitionResult)
        assert result2.text == "quality result"
        assert result2.is_preliminary is False


    def test_full_pipeline_flow(self, config, vad, speech_audio):
        """Test complete flow: AudioSource → AdaptiveWindower → Recognizer."""
        chunk_queue = queue.Queue()
        text_queue = queue.Queue()

        # Set up windower
        windower = AdaptiveWindower(chunk_queue=chunk_queue, config=config)

        # Set up audio source with windower
        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=vad,
            windower=windower,
            config=config
        )

        # Set up recognizer with mock model
        mock_model = Mock()
        mock_model.recognize.return_value = "recognized text"
        recognizer = Recognizer(chunk_queue, text_queue, mock_model)

        # Feed audio through source - use 4 seconds to ensure finalized window
        chunk_size = int(config['audio']['sample_rate'] * config['audio']['chunk_duration'])
        for i in range(0, len(speech_audio[:64000]), chunk_size):  # 4 seconds @ 16kHz
            chunk = speech_audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                break
            audio_source.process_chunk_with_vad(chunk, i * config['audio']['chunk_duration'])

        # Flush to complete pipeline
        audio_source.stop()

        # Process recognition queue
        segments_to_process = []
        while not chunk_queue.empty():
            segments_to_process.append(chunk_queue.get())

        # Recognize each segment
        results = []
        for segment in segments_to_process:
            is_preliminary = (segment.type == 'preliminary')
            result = recognizer.recognize_window(segment, is_preliminary=is_preliminary)
            if result is not None:
                results.append(result)

        # Verify we got recognition results
        assert len(results) > 0

        # Check that we have both preliminary and finalized results
        preliminary_count = 0
        finalized_count = 0

        for result in results:
            assert isinstance(result, RecognitionResult)
            if result.is_preliminary:
                preliminary_count += 1
            else:
                finalized_count += 1

        # Should have at least one of each type
        assert preliminary_count >= 1, "Should have preliminary results"
        assert finalized_count >= 1, "Should have finalized results"


    def test_timing_preservation_through_pipeline(self, config):
        """Timing information should be preserved through entire pipeline."""
        mock_model = Mock()
        mock_model.recognize.return_value = "test"

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model)

        # Create segment with specific timing
        segment = AudioSegment(
            type='finalized',
            data=np.random.randn(48000).astype(np.float32) * 0.1,
            start_time=1.234,
            end_time=4.567,
            chunk_ids=[0, 1, 2]
        )

        # Process segment
        result = recognizer.recognize_window(segment, is_preliminary=False)

        # Check timing preserved
        assert result.start_time == 1.234
        assert result.end_time == 4.567
