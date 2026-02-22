# tests/test_sound_preprocessor_queue_full.py
import pytest
import queue
import numpy as np
from unittest.mock import Mock
from src.sound.SoundPreProcessor import SoundPreProcessor
from src.types import AudioSegment


class TestSoundPreProcessorQueueFull:
    """Tests for SoundPreProcessor queue full handling.

    Verifies that SoundPreProcessor uses non-blocking queue operations
    and properly handles full speech_queue by dropping segments with logging.
    """

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
                'max_speech_duration_ms': 3000
            }
        }

    @pytest.fixture
    def mock_vad(self):
        """Mock VAD that returns speech detection"""
        mock = Mock()
        mock.process_frame = Mock(return_value={'is_speech': True, 'speech_probability': 0.9})
        return mock

    @pytest.fixture
    def mock_windower(self):
        """Mock windower"""
        mock = Mock()
        mock.process_segment = Mock()
        mock.flush = Mock()
        return mock

    def test_segment_delegated_to_windower(self, preprocessor_config, mock_vad, mock_windower):
        """SoundPreProcessor delegates segment emission to windower.

        Logic: Speech → silence threshold → windower.process_segment() called with segment.
        Queue full handling is now windower's responsibility (tested in test_growing_window_assembler.py).
        """
        chunk_queue = queue.Queue()
        speech_queue = queue.Queue(maxsize=1)

        preprocessor = SoundPreProcessor(
            chunk_queue=chunk_queue,
            speech_queue=speech_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=preprocessor_config,
            verbose=True
        )

        # Transition to ACTIVE_SPEECH (requires 3 consecutive speech chunks)
        for i in range(3):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.5,
                'timestamp': float(i * 0.032)
            }
            preprocessor._process_chunk(chunk)

        # Add silence chunks to trigger finalization
        mock_vad.process_frame = Mock(return_value={'is_speech': False, 'speech_probability': 0.1})

        for i in range(2):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.01,
                'timestamp': float((3 + i) * 0.032)
            }
            preprocessor._process_chunk(chunk)

        # Segment should be delegated to windower, not put on speech_queue directly
        assert mock_windower.process_segment.called
        segment = mock_windower.process_segment.call_args[0][0]
        assert isinstance(segment, AudioSegment)
        assert segment.type == 'incremental'

