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
                'max_speech_duration_ms': 3000,
                'silence_timeout': 0.5
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

    def test_speech_queue_full_drops_segment_and_logs(self, preprocessor_config, mock_vad, mock_windower, caplog):
        """Full speech_queue causes non-blocking drop with warning.

        Logic: When speech_queue.put_nowait() raises queue.Full,
        segment is dropped, counter incremented, warning logged.
        """
        # Create small queue (maxsize=1) to trigger full condition
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

        # Fill the speech_queue
        dummy_segment = AudioSegment(
            type='preliminary',
            data=np.zeros(512, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=0.032,
            chunk_ids=[999]
        )
        speech_queue.put(dummy_segment)

        # Create segment that will trigger put attempt
        # Need to force segment emission: IDLE → WAITING_CONFIRMATION → ACTIVE_SPEECH → silence threshold

        # Transition to ACTIVE_SPEECH (requires 3 consecutive speech chunks)
        for i in range(3):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.5,  # Strong speech
                'timestamp': float(i * 0.032)
            }
            preprocessor._process_chunk(chunk)

        # Now add silence chunks to trigger finalization (silence_energy_threshold = 1.5)
        # Each silence chunk with prob=0.1 adds energy of 0.9
        # Need 2 silence chunks to exceed threshold
        mock_vad.process_frame = Mock(return_value={'is_speech': False, 'speech_probability': 0.1})

        for i in range(2):
            chunk = {
                'audio': np.random.randn(512).astype(np.float32) * 0.01,  # Silence
                'timestamp': float((3 + i) * 0.032)
            }
            preprocessor._process_chunk(chunk)

        # Verify segment was dropped (counter incremented)
        assert preprocessor.dropped_segments == 1

        # Verify warning logged
        assert "speech_queue full" in caplog.text
        assert "total drops: 1" in caplog.text
