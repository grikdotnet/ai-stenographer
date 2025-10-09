# tests/test_audiosource_rms_normalization.py
import pytest
import queue
import time
import numpy as np
from src.AudioSource import AudioSource


class TestAudioSourceRMSNormalization:
    """Tests for RMS normalization (AGC) in AudioSource.

    RMS (Root Mean Square) normalization provides automatic gain control,
    boosting quiet audio and reducing loud audio to maintain consistent
    volume levels. This mimics the behavior of video conferencing apps
    like Google Meet and Teams.
    """

    @pytest.fixture
    def config(self):
        """Configuration with RMS normalization enabled."""
        return {
            'audio': {
                'sample_rate': 16000,
                'chunk_duration': 0.032,
                'silence_energy_threshold': 1.2,
                'rms_normalization': {
                    'enabled': True,
                    'target_rms': 0.05,
                    'silence_threshold': 0.001
                }
            },
            'vad': {
                'model_path': './models/silero_vad/silero_vad.onnx',
                'threshold': 0.5,
                'frame_duration_ms': 32
            },
            'windowing': {
                'window_duration': 3.0,
                'step_size': 1.0,
                'max_speech_duration_ms': 3000,
                'silence_timeout': 0.5
            }
        }

    def test_normalizes_quiet_audio_to_target_rms(self, config, vad, mock_windower):
        """Quiet audio should be boosted to target RMS level."""
        chunk_queue = queue.Queue()
        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=vad,
            windower=mock_windower,
            config=config
        )

        # Create quiet audio chunk (RMS ~0.01, should be boosted 5x to 0.05)
        quiet_audio = np.random.randn(512).astype(np.float32) * 0.01
        original_rms = np.sqrt(np.mean(quiet_audio**2))

        # Mock VAD to return speech (so we can capture the normalized audio)
        vad.process_frame = lambda x: {'is_speech': True, 'speech_probability': 0.9}

        # Process chunk
        audio_source.process_chunk_with_vad(quiet_audio, time.time())

        # Get the buffered audio (it's in speech_buffer)
        buffered_audio = audio_source.speech_buffer[0]
        normalized_rms = np.sqrt(np.mean(buffered_audio**2))

        # RMS should be close to target_rms (0.05)
        target_rms = config['audio']['rms_normalization']['target_rms']
        assert abs(normalized_rms - target_rms) < 0.01, \
            f"Expected RMS ~{target_rms}, got {normalized_rms:.4f}"

        # Verify it was boosted (not reduced)
        assert normalized_rms > original_rms, "Quiet audio should be boosted"


    def test_normalizes_loud_audio_to_target_rms(self, config, vad, mock_windower):
        """Loud audio should be reduced to target RMS level."""
        chunk_queue = queue.Queue()
        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=vad,
            windower=mock_windower,
            config=config
        )

        # Create loud audio chunk (RMS ~0.2, should be reduced to 0.05)
        loud_audio = np.random.randn(512).astype(np.float32) * 0.2
        original_rms = np.sqrt(np.mean(loud_audio**2))

        # Mock VAD to return speech
        vad.process_frame = lambda x: {'is_speech': True, 'speech_probability': 0.9}

        # Process chunk
        audio_source.process_chunk_with_vad(loud_audio, time.time())

        # Get the buffered audio
        buffered_audio = audio_source.speech_buffer[0]
        normalized_rms = np.sqrt(np.mean(buffered_audio**2))

        # RMS should be close to target_rms (0.05)
        target_rms = config['audio']['rms_normalization']['target_rms']
        assert abs(normalized_rms - target_rms) < 0.01, \
            f"Expected RMS ~{target_rms}, got {normalized_rms:.4f}"

        # Verify it was reduced (not boosted)
        assert normalized_rms < original_rms, "Loud audio should be reduced"


    def test_does_not_boost_silence(self, config, vad, mock_windower):
        """Silence (RMS below threshold) should not be boosted."""
        chunk_queue = queue.Queue()
        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=vad,
            windower=mock_windower,
            config=config
        )

        # Create near-silence (RMS ~0.0005, below silence_threshold of 0.001)
        silence = np.random.randn(512).astype(np.float32) * 0.0005
        original_rms = np.sqrt(np.mean(silence**2))

        # Mock VAD to return speech (to capture the audio)
        vad.process_frame = lambda x: {'is_speech': True, 'speech_probability': 0.9}

        # Process chunk
        audio_source.process_chunk_with_vad(silence, time.time())

        # Get the buffered audio
        buffered_audio = audio_source.speech_buffer[0]

        # Audio should be unchanged (not boosted)
        assert np.allclose(buffered_audio, silence), \
            "Silence should not be normalized (to avoid boosting background noise)"


    def test_normalization_disabled_when_config_false(self, config, vad, mock_windower):
        """When normalization is disabled, audio should pass through unchanged."""
        # Disable normalization
        config['audio']['rms_normalization']['enabled'] = False

        chunk_queue = queue.Queue()
        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=vad,
            windower=mock_windower,
            config=config
        )

        # Create quiet audio that would normally be boosted
        quiet_audio = np.random.randn(512).astype(np.float32) * 0.01
        original_rms = np.sqrt(np.mean(quiet_audio**2))

        # Mock VAD to return speech
        vad.process_frame = lambda x: {'is_speech': True, 'speech_probability': 0.9}

        # Process chunk
        audio_source.process_chunk_with_vad(quiet_audio, time.time())

        # Get the buffered audio
        buffered_audio = audio_source.speech_buffer[0]
        buffered_rms = np.sqrt(np.mean(buffered_audio**2))

        # RMS should be unchanged (normalization disabled)
        assert abs(buffered_rms - original_rms) < 0.001, \
            "With normalization disabled, RMS should be unchanged"
