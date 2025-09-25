# src/AudioSource.py
import sounddevice as sd
import queue
import numpy as np
import time
from typing import Optional, Dict, Any

class AudioSource:
    def __init__(self, chunk_queue: queue.Queue, sample_rate: int = 16000, chunk_duration: float = 0.1):
        self.chunk_queue: queue.Queue = chunk_queue
        self.sample_rate: int = sample_rate
        self.chunk_size: int = int(sample_rate * chunk_duration)
        self.is_running: bool = False
        self.stream: Optional[sd.InputStream] = None

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Callback from sounddevice - puts raw chunks in queue"""
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
        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.chunk_size
        )
        self.stream.start()

    def stop(self) -> None:
        self.is_running = False
        if self.stream:
            self.stream.stop()