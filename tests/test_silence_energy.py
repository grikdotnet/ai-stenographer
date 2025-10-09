# tests/test_silence_energy.py
import pytest
import queue
import numpy as np
from src.AudioSource import AudioSource
from src.types import AudioSegment


class TestSilenceEnergyComputation:
    """Tests for cumulative probability-based silence energy computation.

    Tests the new silence energy mechanism that uses speech probability
    scores to adaptively determine when to finalize speech segments.
    """

    @pytest.fixture
    def config(self):
        """Configuration with silence energy parameters."""
        return {
            'audio': {
                'sample_rate': 16000,
                'chunk_duration': 0.032,  # 32ms chunks
                'silence_energy_threshold': 1.5
            },
            'vad': {
                'model_path': './models/silero_vad/silero_vad.onnx',
                'threshold': 0.35,
                'frame_duration_ms': 32
            },
            'windowing': {
                'window_duration': 3.0,
                'step_size': 1.0,
                'max_speech_duration_ms': 3000
            }
        }

    @pytest.fixture
    def audio_source(self, config, mock_windower):
        """Create AudioSource with mocked VAD for testing."""
        from unittest.mock import Mock

        mock_vad = Mock()
        chunk_queue = queue.Queue()
        return AudioSource(
            chunk_queue=chunk_queue,
            vad=mock_vad,
            windower=mock_windower,
            config=config
        )

    def test_speech_resets_silence_energy(self, config, audio_source):
        """Speech detection should reset silence_energy to 0.0."""
        # Create high-probability speech frame
        frame = np.random.randn(512).astype(np.float32) * 0.1

        # Set initial silence energy
        audio_source.silence_energy = 1.0

        # Process frame - mock VAD to return speech
        from unittest.mock import patch
        with patch.object(audio_source.vad, 'process_frame') as mock_vad:
            mock_vad.return_value = {
                'is_speech': True,
                'speech_probability': 0.8
            }
            audio_source.process_chunk_with_vad(frame, 0.0)

        # Silence energy should be reset to 0.0
        assert audio_source.silence_energy == 0.0

    def test_silence_accumulates_energy(self, config, audio_source):
        """Silence (prob < 0.35) should accumulate silence_energy."""
        frame = np.random.randn(512).astype(np.float32) * 0.1

        # Start with zero silence energy
        audio_source.silence_energy = 0.0
        audio_source.is_speech_active = True
        audio_source.speech_buffer = [frame]

        # Process frame with low probability
        from unittest.mock import patch
        with patch.object(audio_source.vad, 'process_frame') as mock_vad:
            mock_vad.return_value = {
                'is_speech': False,
                'speech_probability': 0.15  # Silence
            }
            audio_source.process_chunk_with_vad(frame, 0.0)

        # Silence energy should accumulate: 0.0 + (1.0 - 0.15) = 0.85
        assert audio_source.silence_energy == pytest.approx(0.85, abs=0.01)

    def test_silence_energy_only_accumulates_on_non_speech(self, config, audio_source):
        """Silence energy should only accumulate when is_speech=False."""
        frame = np.random.randn(512).astype(np.float32) * 0.1

        # Start with zero silence energy
        audio_source.silence_energy = 0.0

        # Process frame with low probability BUT is_speech=True (above threshold)
        from unittest.mock import patch
        with patch.object(audio_source.vad, 'process_frame') as mock_vad:
            mock_vad.return_value = {
                'is_speech': True,  # Above threshold
                'speech_probability': 0.4  # Low but above threshold
            }
            audio_source.process_chunk_with_vad(frame, 0.0)

        # Silence energy should be reset (not accumulated)
        assert audio_source.silence_energy == 0.0

    def test_energy_threshold_triggers_finalization(self, config, audio_source):
        """When silence_energy >= silence_energy_threshold, segment should finalize."""
        chunk_queue = audio_source.chunk_queue
        frame = np.random.randn(512).astype(np.float32) * 0.1

        # Start speech segment
        audio_source.is_speech_active = True
        audio_source.speech_start_time = 0.0
        audio_source.speech_buffer = [frame, frame]  # 64ms
        audio_source.silence_energy = 1.4  # Just below threshold (1.5)

        # Process silence frame that pushes energy over threshold
        from unittest.mock import patch
        with patch.object(audio_source.vad, 'process_frame') as mock_vad:
            mock_vad.return_value = {
                'is_speech': False,
                'speech_probability': 0.2  # Adds 0.8 energy
            }
            audio_source.process_chunk_with_vad(frame, 0.064)

        # Segment should be finalized and emitted
        assert not chunk_queue.empty()
        segment = chunk_queue.get()
        assert isinstance(segment, AudioSegment)
        assert segment.type == 'preliminary'
        assert len(segment.chunk_ids) == 1

    def test_silence_energy_resets_after_finalization(self, config, audio_source):
        """Silence energy should reset to 0.0 after segment finalization."""
        chunk_queue = audio_source.chunk_queue
        frame = np.random.randn(512).astype(np.float32) * 0.1

        # Start speech segment
        audio_source.is_speech_active = True
        audio_source.speech_start_time = 0.0
        audio_source.speech_buffer = [frame, frame]
        audio_source.silence_energy = 2.0  # Above threshold

        # Process frame to trigger finalization
        from unittest.mock import patch
        with patch.object(audio_source.vad, 'process_frame') as mock_vad:
            mock_vad.return_value = {
                'is_speech': False,
                'speech_probability': 0.2
            }
            audio_source.process_chunk_with_vad(frame, 0.064)

        # Segment should be finalized
        assert not chunk_queue.empty()

        # Silence energy should be reset
        assert audio_source.silence_energy == 0.0
