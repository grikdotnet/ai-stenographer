# src/AudioSource.py
import sounddevice as sd
import queue
import numpy as np
import time
from typing import Optional, Dict, Any

class AudioSource:
    """Captures audio from microphone and puts chunks into a queue.

    AudioSource uses sounddevice to capture real-time audio from the microphone
    and splits it into fixed-size chunks that are placed in a queue for processing
    by downstream components in the STT pipeline.

    Args:
        chunk_queue: Queue to receive audio chunks
        sample_rate: Audio sample rate in Hz
        chunk_duration: Duration of each audio chunk in seconds
    """

    def __init__(self, chunk_queue: queue.Queue, sample_rate: int = 16000, chunk_duration: float = 0.1):
        self.chunk_queue: queue.Queue = chunk_queue
        self.sample_rate: int = sample_rate
        self.chunk_size: int = int(sample_rate * chunk_duration)
        self.is_running: bool = False
        self.stream: Optional[sd.InputStream] = None

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Callback from sounddevice that processes incoming audio data.

        This callback is called automatically by sounddevice when audio data
        is available from the microphone. It converts the audio to float32
        format and puts it in the chunk queue with timestamp information.

        Args:
            indata: Input audio data as numpy array
            frames: Number of audio frames
            time_info: Timing information from sounddevice
            status: Status information from sounddevice
        """
        if status:
            print(f"Audio error: {status}")
        
        # Convert int16 to float32 and put in queue
        audio_float: np.ndarray = indata[:, 0].astype(np.float32)
        chunk_data: Dict[str, Any] = {
            'data': audio_float.copy(),
            'timestamp': time.time()
        }
        self.chunk_queue.put(chunk_data)

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

        Stops the sounddevice InputStream and sets the running flag to False.
        """
        self.is_running = False
        if self.stream:
            self.stream.stop()