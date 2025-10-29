# src/AudioSource.py
from __future__ import annotations
import sounddevice as sd
import queue
import numpy as np
import time
import logging
from typing import Dict, Any


class AudioSource:
    """Captures audio from microphone and emits raw chunks to queue.

    AudioSource uses sounddevice to capture real-time audio from the microphone
    and immediately puts raw audio chunks to chunk_queue. No processing is done
    in the callback to prevent blocking and audio dropouts.

    The callback is designed to be minimal (<1ms) to avoid buffer overflows.
    All processing (VAD, normalization, buffering) happens in SoundPreProcessor thread.

    Args:
        chunk_queue: Queue to send raw audio chunks (dict with audio, timestamp)
        config: Configuration dictionary loaded from stt_config.json
        verbose: Enable verbose logging
    """

    def __init__(self,
                 chunk_queue: queue.Queue,
                 config: Dict[str, Any],
                 verbose: bool = False):

        self.chunk_queue: queue.Queue = chunk_queue
        self.verbose: bool = verbose

        self.sample_rate: int = config['audio']['sample_rate']
        chunk_duration = config['audio']['chunk_duration']

        self.chunk_size: int = int(self.sample_rate * chunk_duration)
        self.is_running: bool = False
        self.stream: sd.InputStream | None = None


    def audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Callback from sounddevice that processes incoming audio data.

        This callback is called automatically by sounddevice when audio data
        is available from the microphone. Strip stereo, take 1st channel.

        Args:
            indata: Input audio data as numpy array (shape: [frames, channels])
            frames: Number of audio frames
            time_info: Timing information from sounddevice
            status: Status information from sounddevice
        """
        if status:
            logging.error(f"Audio error: {status}")

        # If the sound comes in stereo, take the 1st audio channel
        audio_float: np.ndarray = indata[:, 0].astype(np.float32)
        current_time = time.time()

        chunk_data = {
            'audio': audio_float,
            'timestamp': current_time
        }

        try:
            self.chunk_queue.put_nowait(chunk_data)
        except queue.Full:
            logging.warning("chunk_queue full, dropping audio chunk")


    def start(self) -> None:
        """Start capturing audio from the microphone.

        Creates and starts a sounddevice InputStream that will begin
        capturing audio and calling the audio_callback method.
        """
        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.chunk_size
        )
        self.stream.start()


    def stop(self) -> None:
        """Stop capturing audio from the microphone.

        Stops the sounddevice InputStream.
        """
        self.is_running = False
        if self.stream:
            self.stream.stop()
