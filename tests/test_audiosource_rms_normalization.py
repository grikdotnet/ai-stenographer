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
        """Buffered audio should remain raw (unnormalized) for STT, but VAD receives normalized."""
        chunk_queue = queue.Queue()
        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=vad,
            windower=mock_windower,
            config=config
        )

        # Create quiet audio chunk (RMS ~0.01)
        quiet_audio = np.random.randn(512).astype(np.float32) * 0.01
        original_rms = np.sqrt(np.mean(quiet_audio**2))

        vad.process_frame = lambda x: {'is_speech': True, 'speech_probability': 0.9}

        # Process single chunk
        audio_source.process_chunk_with_vad(quiet_audio, time.time())

        # Get the buffered audio (should be raw, not normalized)
        buffered_audio = audio_source.speech_buffer[0]
        buffered_rms = np.sqrt(np.mean(buffered_audio**2))

        # Verify buffered audio is raw (NOT boosted)
        assert abs(buffered_rms - original_rms) < 0.001, \
            "Buffered audio should be raw (unnormalized) for STT"

        # Verify gain state was updated (proves normalization happened internally for VAD)
        assert audio_source.current_gain > 1.0, \
            "Gain should be > 1.0 for quiet audio (proves normalization logic ran)"


    def test_normalizes_loud_audio_to_target_rms(self, config, vad, mock_windower):
        """Buffered audio should remain raw (unnormalized) for STT, gain state tracks loud audio."""
        chunk_queue = queue.Queue()
        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=vad,
            windower=mock_windower,
            config=config
        )

        # Create loud audio chunk (RMS ~0.2)
        loud_audio = np.random.randn(512).astype(np.float32) * 0.2
        original_rms = np.sqrt(np.mean(loud_audio**2))

        # Mock VAD to return speech
        vad.process_frame = lambda x: {'is_speech': True, 'speech_probability': 0.9}

        # Process single chunk
        audio_source.process_chunk_with_vad(loud_audio, time.time())

        # Get the buffered audio (should be raw, not normalized)
        buffered_audio = audio_source.speech_buffer[0]
        buffered_rms = np.sqrt(np.mean(buffered_audio**2))

        # Verify buffered audio is raw (NOT reduced)
        assert abs(buffered_rms - original_rms) < 0.001, \
            "Buffered audio should be raw (unnormalized) for STT"

        # Verify gain state was updated (proves normalization happened internally for VAD)
        assert audio_source.current_gain < 1.0, \
            "Gain should be < 1.0 for loud audio (proves normalization logic ran)"


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

    def test_temporal_smoothing_prevents_sudden_gain_changes(self, config, vad, mock_windower):
        """Gain should change gradually with temporal smoothing, not instantly."""
        config['audio']['rms_normalization']['gain_smoothing'] = 0.9

        chunk_queue = queue.Queue()
        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=vad,
            windower=mock_windower,
            config=config
        )

        vad.process_frame = lambda x: {'is_speech': True, 'speech_probability': 0.9}

        # Process sequence: quiet → loud
        quiet_audio = np.random.randn(512).astype(np.float32) * 0.01
        loud_audio = np.random.randn(512).astype(np.float32) * 0.2

        # First chunk: quiet (RMS ~0.01) - should increase gain
        audio_source.process_chunk_with_vad(quiet_audio, time.time())
        first_gain = audio_source.current_gain

        # Gain should be moving up from 1.0 toward ~5.0
        assert first_gain > 1.0, "Gain should increase for quiet audio"

        # Second chunk: loud (RMS ~0.2) - should decrease gain
        audio_source.process_chunk_with_vad(loud_audio, time.time())
        second_gain = audio_source.current_gain

        # With smoothing=0.9, gain shouldn't drop instantly to 0.25
        # It should be between first_gain and target_gain
        assert second_gain < first_gain, "Gain should decrease when loud audio appears"
        assert second_gain > 0.25, "Gain shouldn't instantly reach target due to smoothing"


    def test_maintains_gain_state_across_chunks(self, config, vad, mock_windower):
        """Gain state should persist across multiple chunks and converge to target."""
        config['audio']['rms_normalization']['gain_smoothing'] = 0.8

        chunk_queue = queue.Queue()
        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=vad,
            windower=mock_windower,
            config=config
        )

        vad.process_frame = lambda x: {'is_speech': True, 'speech_probability': 0.9}

        # Process multiple chunks with consistent low RMS
        target_rms = config['audio']['rms_normalization']['target_rms']  # 0.05
        input_rms = 0.01  # Quiet audio

        gain_values = []
        for _ in range(10):
            # Create new chunk with same RMS level
            chunk = np.random.randn(512).astype(np.float32) * input_rms
            audio_source.process_chunk_with_vad(chunk, time.time())
            gain_values.append(audio_source.current_gain)

        # Verify gain state persists (current_gain should be set)
        assert hasattr(audio_source, 'current_gain'), "AudioSource should have current_gain attribute"

        # Verify gain converges toward target_gain over time
        # target_gain = target_rms / input_rms = 0.05 / 0.01 = 5.0
        target_gain = target_rms / input_rms

        # Later gains should be closer to target than earlier gains
        assert abs(gain_values[-1] - target_gain) < abs(gain_values[0] - target_gain), \
            "Gain should converge closer to target over multiple chunks"

        # The last gain should be very close to target after convergence
        assert abs(gain_values[-1] - target_gain) < 1.0, \
            f"After 10 chunks, gain should be near target {target_gain}, got {gain_values[-1]:.2f}"


    def test_vad_receives_normalized_stt_receives_raw(self, config, vad, mock_windower):
        """VAD should receive normalized audio while STT receives raw audio (dual-path)."""
        chunk_queue = queue.Queue()

        # Create mock VAD to capture what it receives AND force speech detection
        captured_vad_input = []

        def mock_process_frame(audio):
            captured_vad_input.append(audio.copy())
            # Force speech detection so buffer gets populated
            return {'is_speech': True, 'speech_probability': 0.9}

        vad.process_frame = mock_process_frame

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            vad=vad,
            windower=mock_windower,
            config=config
        )

        # Create quiet audio (RMS ~0.01, should be boosted for VAD)
        quiet_audio = np.random.randn(512).astype(np.float32) * 0.01
        original_rms = np.sqrt(np.mean(quiet_audio**2))

        # Process chunk
        audio_source.process_chunk_with_vad(quiet_audio, time.time())

        # Get what VAD received
        vad_audio = captured_vad_input[0]
        vad_rms = np.sqrt(np.mean(vad_audio**2))

        # Get what's buffered for STT
        buffered_audio = audio_source.speech_buffer[0]
        buffered_rms = np.sqrt(np.mean(buffered_audio**2))

        # VAD should receive normalized audio (boosted)
        assert vad_rms > original_rms, \
            "VAD should receive normalized (boosted) audio"

        # Buffer should contain raw audio (not boosted)
        assert abs(buffered_rms - original_rms) < 0.001, \
            "STT buffer should contain raw (unnormalized) audio"

        # They should be different
        assert not np.allclose(vad_audio, buffered_audio), \
            "VAD input and STT buffer should be different (normalized vs raw)"
