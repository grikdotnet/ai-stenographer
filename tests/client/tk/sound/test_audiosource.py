# tests/test_audiosource.py
import pytest
import queue
import numpy as np
from unittest.mock import Mock, patch
from src.client.tk.sound.AudioSource import AudioSource


class TestAudioSource:
    """Tests for AudioSource callback for audio capture.
    AudioSource captures audio and puts raw chunks to queue.
    """

    @pytest.fixture
    def config(self):
        """Standard configuration for AudioSource."""
        return {
            'audio': {
                'sample_rate': 16000,
                'chunk_duration': 0.032  # 32ms chunks (512 samples)
            }
        }

    def test_audio_callback_puts_raw_chunk_to_queue(self, config):
        """AudioSource callback should put raw audio dict to queue.

        Logic: Callback receives audio → creates dict {audio, timestamp} → puts to queue.
        """
        chunk_queue = queue.Queue()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            config=config,
            verbose=False
        )

        # indata shape: (frames, channels)
        indata = np.random.randn(512, 1).astype(np.float32) * 0.1
        frames = 512
        time_info = None
        status = None

        # Mock time.time()
        with patch('time.time', return_value=123.456):
            audio_source.audio_callback(indata, frames, time_info, status)

        # Queue should have one item
        assert not chunk_queue.empty()
        chunk_data = chunk_queue.get()

        # Verify structure
        assert isinstance(chunk_data, dict)
        assert 'audio' in chunk_data
        assert 'timestamp' in chunk_data

        # Verify data
        assert isinstance(chunk_data['audio'], np.ndarray)
        assert chunk_data['audio'].dtype == np.float32
        assert len(chunk_data['audio']) == 512
        assert chunk_data['timestamp'] == 123.456


    def test_audio_source_start_stop(self, config):
        """AudioSource should start and stop stream correctly.

        Logic: start() → stream created and started → stop() → stream stopped.
        """
        chunk_queue = queue.Queue()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            config=config,
            verbose=False
        )

        assert not audio_source.is_running
        assert audio_source.stream is None

        # Mock sounddevice.InputStream
        mock_stream = Mock()
        mock_stream.active = True

        with patch('sounddevice.InputStream', return_value=mock_stream) as mock_input_stream:
            audio_source.start()

            # Verify InputStream was created with correct parameters
            mock_input_stream.assert_called_once_with(
                samplerate=16000,
                channels=1,
                callback=audio_source.audio_callback,
                blocksize=512
            )

            # Verify stream was started
            mock_stream.start.assert_called_once()

            # Verify AudioSource state
            assert audio_source.is_running
            assert audio_source.stream is mock_stream

            # Stop should stop stream
            audio_source.stop()
            mock_stream.stop.assert_called_once()
            assert not audio_source.is_running


    def test_pause_state_stops_and_closes_stream(self, config):
        """Test that pausing stops and closes the audio stream.

        Logic: state changes to 'paused' → stop() and close() stream to release microphone.
        """
        chunk_queue = queue.Queue()
        mock_app_state = Mock()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            config=config,
            app_state=mock_app_state,
            verbose=False
        )

        mock_stream = Mock()
        audio_source.stream = mock_stream
        audio_source.is_running = True

        audio_source.on_state_change('running', 'paused')

        # Verify stream was stopped and closed
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


    def test_resume_state_creates_new_stream(self, config):
        """Test that resuming creates a new audio stream.

        Logic: state changes from 'paused' to 'running' → start() creates new stream.
        """
        chunk_queue = queue.Queue()
        mock_app_state = Mock()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            config=config,
            app_state=mock_app_state,
            verbose=False
        )

        audio_source.is_running = False
        audio_source.stream = None

        mock_stream = Mock()

        with patch('sounddevice.InputStream', return_value=mock_stream):
            audio_source.on_state_change('paused', 'running')

            # Verify stream was started
            mock_stream.start.assert_called_once()
            assert audio_source.is_running


    def test_shutdown_state_stops_stream(self, config):
        """Test that shutdown stops the audio stream.

        Logic: state changes to 'shutdown' → stop() stream.
        """
        chunk_queue = queue.Queue()
        mock_app_state = Mock()

        audio_source = AudioSource(
            chunk_queue=chunk_queue,
            config=config,
            app_state=mock_app_state,
            verbose=False
        )

        mock_stream = Mock()
        audio_source.stream = mock_stream
        audio_source.is_running = True

        audio_source.on_state_change('running', 'shutdown')

        # Verify stream was stopped
        mock_stream.stop.assert_called_once()
