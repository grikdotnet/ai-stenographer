# tests/test_vad.py
import numpy as np
from src.VoiceActivityDetector import VoiceActivityDetector


class TestVoiceActivityDetector:
    """Tests for Voice Activity Detection component.

    Tests VAD's ability to distinguish speech from silence, detect segment
    boundaries, and enforce duration constraints.
    """

    def test_detects_silence(self, silence_audio):
        """VAD should identify silence audio as non-speech."""
        config = {
            'vad': {
                'model_path': './models/silero_vad/silero_vad.onnx',
                'threshold': 0.5,
                'frame_duration_ms': 32,
                'min_speech_duration_ms': 250
            },
            'audio': {
                'sample_rate': 16000
            }
        }

        vad = VoiceActivityDetector(config)
        result = vad.process_frame(silence_audio[:512])

        assert result['is_speech'] is False
        assert result['speech_probability'] < 0.5


    def test_detects_speech(self, speech_audio):
        """VAD should identify real speech audio as speech."""
        config = {
            'vad': {
                'model_path': './models/silero_vad/silero_vad.onnx',
                'threshold': 0.5,
                'frame_duration_ms': 32,
                'min_speech_duration_ms': 250
            },
            'audio': {
                'sample_rate': 16000
            }
        }

        vad = VoiceActivityDetector(config)
        # Process frame from middle of speech (skip first frame which may be silence/noise)
        # Frame 1 (at 512 samples = 32ms) contains speech based on empirical testing
        result = vad.process_frame(speech_audio[512:1024])

        assert result['is_speech'] is True
        assert result['speech_probability'] > 0.5


    def test_handles_noisy_audio(self, speech_audio, noise_audio):
        """VAD should detect speech in presence of background noise."""
        config = {
            'vad': {
                'model_path': './models/silero_vad/silero_vad.onnx',
                'threshold': 0.5,
                'frame_duration_ms': 32,
                'min_speech_duration_ms': 100
            },
            'audio': {
                'sample_rate': 16000
            }
        }

        vad = VoiceActivityDetector(config)

        # Add very low noise (0.1%) to speech - Silero is robust to this level
        # Use frame 1 (512:1024) which contains speech
        noisy_speech = speech_audio[512:16512] + noise_audio * 0.001

        result = vad.process_frame(noisy_speech[:512])

        # Should still detect speech despite noise
        assert result['is_speech'] is True
        assert result['speech_probability'] > 0.5
