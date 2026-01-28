# tests/test_vad.py
import numpy as np
from pathlib import Path
from src.asr.VoiceActivityDetector import VoiceActivityDetector


class TestVoiceActivityDetector:
    """Tests for Voice Activity Detection component.

    Tests VAD's ability to distinguish speech from silence, detect segment
    boundaries, and enforce duration constraints.
    """

    def test_detects_silence(self, silence_audio):
        """VAD should identify silence audio as non-speech."""
        config = {
            'vad': {
                'threshold': 0.5,
                'frame_duration_ms': 32
            },
            'audio': {
                'sample_rate': 16000
            }
        }

        model_path = Path('./models/silero_vad/silero_vad.onnx')
        vad = VoiceActivityDetector(config, model_path)
        result = vad.process_frame(silence_audio[:512])

        assert result['is_speech'] is False
        assert result['speech_probability'] < 0.5


    def test_detects_speech(self, speech_audio):
        """VAD should identify real speech audio as speech."""
        config = {
            'vad': {
                'threshold': 0.5,
                'frame_duration_ms': 32
            },
            'audio': {
                'sample_rate': 16000
            }
        }

        model_path = Path('./models/silero_vad/silero_vad.onnx')
        vad = VoiceActivityDetector(config, model_path)
        # Process frame from middle of speech (skip first frame which may be silence/noise)
        # Frame 1 (at 512 samples = 32ms) contains speech based on empirical testing
        result = vad.process_frame(speech_audio[512:1024])

        assert result['is_speech'] is True
        assert result['speech_probability'] > 0.5


    def test_handles_noisy_audio(self, speech_audio, noise_audio):
        """VAD should detect speech in presence of background noise."""
        config = {
            'vad': {
                'threshold': 0.5,
                'frame_duration_ms': 32
            },
            'audio': {
                'sample_rate': 16000
            }
        }

        model_path = Path('./models/silero_vad/silero_vad.onnx')
        vad = VoiceActivityDetector(config, model_path)

        # Add very low noise (0.1%) to speech - Silero is robust to this level
        # Use frame 1 (512:1024) which contains speech
        noisy_speech = speech_audio[512:16512] + noise_audio * 0.001

        result = vad.process_frame(noisy_speech[:512])

        # Should still detect speech despite noise
        assert result['is_speech'] is True
        assert result['speech_probability'] > 0.5


    def test_process_frame_maintains_model_state(self, speech_audio):
        """VAD should maintain ONNX model state for temporal context."""
        config = {
            'vad': {
                'threshold': 0.5,
                'frame_duration_ms': 32
            },
            'audio': {
                'sample_rate': 16000
            }
        }

        model_path = Path('./models/silero_vad/silero_vad.onnx')
        vad = VoiceActivityDetector(config, model_path)

        # Process two consecutive frames
        frame1 = speech_audio[512:1024]
        frame2 = speech_audio[1024:1536]

        result1 = vad.process_frame(frame1)
        result2 = vad.process_frame(frame2)

        # Both should detect speech
        assert result1['is_speech'] is True
        assert result2['is_speech'] is True

        # Model state should be updated (non-zero)
        assert vad.model_state is not None
        assert np.any(vad.model_state != 0), "Model state should be updated after processing"


    def test_reset_model_state_clears_onnx_state(self, speech_audio):
        """reset_model_state should clear ONNX model state only."""
        config = {
            'vad': {
                'threshold': 0.5,
                'frame_duration_ms': 32
            },
            'audio': {
                'sample_rate': 16000
            }
        }

        model_path = Path('./models/silero_vad/silero_vad.onnx')
        vad = VoiceActivityDetector(config, model_path)

        # Process frame to populate model state
        vad.process_frame(speech_audio[512:1024])
        assert np.any(vad.model_state != 0), "Model state should be non-zero after processing"

        # Reset should clear model state
        vad.reset_state()
        assert np.all(vad.model_state == 0), "Model state should be zeroed after reset"


    def test_vad_does_not_buffer_frames(self):
        """VAD should not maintain application-level buffers (SRP test)."""
        config = {
            'vad': {
                'threshold': 0.5,
                'frame_duration_ms': 32
            },
            'audio': {
                'sample_rate': 16000
            }
        }

        model_path = Path('./models/silero_vad/silero_vad.onnx')
        vad = VoiceActivityDetector(config, model_path)

        # VAD should NOT have these application-level attributes
        assert not hasattr(vad, 'speech_buffer'), "VAD should not buffer speech frames"
        assert not hasattr(vad, 'silence_frames'), "VAD should not track silence count"
        assert not hasattr(vad, 'is_speech_active'), "VAD should not track segment state"
        assert not hasattr(vad, 'speech_start_time'), "VAD should not track timing"
