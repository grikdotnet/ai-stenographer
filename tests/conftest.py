# tests/conftest.py
import pytest
import numpy as np
import torch
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session")
def download_test_audio():
    """Download Silero VAD test audio file once per test session.

    Downloads the official Silero VAD English example audio from their CDN.
    The file is cached in tests/fixtures/ to avoid repeated downloads.

    Returns:
        Path: Path to the fixtures directory containing downloaded audio
    """
    FIXTURES_DIR.mkdir(exist_ok=True)

    audio_file = FIXTURES_DIR / 'en.wav'

    if not audio_file.exists():
        url = 'https://models.silero.ai/vad_models/en.wav'
        torch.hub.download_url_to_file(url, str(audio_file))

    return FIXTURES_DIR


@pytest.fixture
def speech_audio(download_test_audio):
    """Load real speech audio for testing VAD speech detection.

    Loads the downloaded English speech sample. Audio is 16kHz mono float32.

    Args:
        download_test_audio: Fixture that ensures audio is downloaded

    Returns:
        np.ndarray: Audio data as float32 array, shape (n_samples,)
    """
    import soundfile as sf
    audio_path = download_test_audio / 'en.wav'
    audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample to 16kHz if needed (simple decimation/interpolation)
    if sample_rate != 16000:
        from scipy import signal
        num_samples = int(len(audio_data) * 16000 / sample_rate)
        audio_data = signal.resample(audio_data, num_samples).astype(np.float32)

    return audio_data


@pytest.fixture
def silence_audio():
    """Generate 1 second of silence for testing VAD silence detection.

    Creates a numpy array of zeros representing silence at 16kHz sample rate.

    Returns:
        np.ndarray: Silence audio as float32 array, shape (16000,)
    """
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def noise_audio():
    """Generate 1 second of low-level background noise for testing.

    Creates random noise at 1% amplitude to simulate quiet background noise.

    Returns:
        np.ndarray: Noise audio as float32 array, shape (16000,)
    """
    return (np.random.randn(16000) * 0.01).astype(np.float32)


@pytest.fixture
def speech_with_pause_audio(speech_audio, silence_audio):
    """Create audio pattern: speech + silence + speech for boundary testing.

    Concatenates 1 second of speech, 0.5 seconds of silence, and 1 second of
    speech to test VAD boundary detection capabilities.

    Args:
        speech_audio: Real speech audio fixture
        silence_audio: Silence audio fixture

    Returns:
        np.ndarray: Combined audio pattern as float32 array
    """
    # Take first 1 second of speech
    speech_1sec = speech_audio[:16000]

    # 0.5 second silence
    silence_half_sec = silence_audio[:8000]

    # Concatenate: speech (1s) + silence (0.5s) + speech (1s)
    return np.concatenate([speech_1sec, silence_half_sec, speech_1sec])


@pytest.fixture
def config():
    """Provide standard test configuration with VAD model path.

    Returns:
        Dict: Configuration dictionary matching production config structure
    """
    return {
        'audio': {
            'sample_rate': 16000,
            'chunk_duration': 0.032
        },
        'vad': {
            'model_path': './models/silero_vad/silero_vad.onnx',
            'frame_duration_ms': 32,
            'threshold': 0.5,
            'min_speech_duration_ms': 64,
            'max_speech_duration_ms': 3000,
            'silence_timeout_ms': 32
        },
        'windowing': {
            'window_duration': 3.0,
            'step_size': 1.0
        }
    }


@pytest.fixture
def vad(config):
    """Provide a VoiceActivityDetector instance for testing.

    Args:
        config: Configuration fixture

    Returns:
        VoiceActivityDetector: VAD instance configured with test config
    """
    from src.VoiceActivityDetector import VoiceActivityDetector
    return VoiceActivityDetector(config=config, verbose=False)
