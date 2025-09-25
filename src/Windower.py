# src/Windower.py
import queue
import threading
import numpy as np
from typing import Dict, Any, Optional, Union

class Windower:
    """Creates overlapping windows from audio chunks for speech recognition.

    Windower accumulates audio chunks from a queue and creates fixed-size
    overlapping windows suitable for speech recognition. Windows are created
    with a specified duration and step size to provide temporal overlap
    that improves recognition accuracy.

    Args:
        chunk_queue: Queue to read audio chunks from
        window_queue: Queue to write audio windows to
        window_duration: Duration of each window in seconds
        step_duration: Step size between windows in seconds (determines overlap)
        sample_rate: Audio sample rate in Hz
    """

    def __init__(self, chunk_queue: queue.Queue, 
                 window_queue: queue.Queue,
                 window_duration: float = 2.0, 
                 step_duration: float = 0.5, 
                 sample_rate: int = 16000
                 ):
        self.chunk_queue: queue.Queue = chunk_queue
        self.window_queue: queue.Queue = window_queue
        self.window_size: int = int(window_duration * sample_rate)
        self.step_size: int = int(step_duration * sample_rate)
        self.sample_rate: int = sample_rate
        self.buffer: np.ndarray = np.array([], dtype=np.float32)
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start processing chunks from queue in a background thread.

        Creates and starts a daemon thread that continuously reads chunks
        from the input queue and processes them using process_chunk().
        The thread runs until stop() is called.
        """
        def _queue_reader() -> None:
            while self.is_running:
                try:
                    chunk_data: Dict[str, Union[np.ndarray, float]] = self.chunk_queue.get(timeout=0.1)
                    self.process_chunk(chunk_data)
                except queue.Empty:
                    continue

        self.is_running = True
        self.thread = threading.Thread(target=_queue_reader, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.is_running = False

    def process_chunk(self, chunk_data: Dict[str, Union[np.ndarray, float]]) -> None:
        """Process a single chunk and create windows when buffer has enough data.

        Accumulates the chunk in an internal buffer and creates overlapping
        windows when sufficient audio data is available. Each window is
        placed in the output queue with timestamp and duration information.
        The buffer size is managed to prevent unbounded memory growth.

        Args:
            chunk_data: Dictionary containing 'data' (numpy array), 'timestamp', etc.
        """
        chunk: np.ndarray = chunk_data['data']

        # Accumulate chunk in buffer
        self.buffer = np.concatenate([self.buffer, chunk])

        # Create windows when buffer has sufficient data
        while len(self.buffer) >= self.window_size:
            window: np.ndarray = self.buffer[:self.window_size]

            window_data: Dict[str, Union[np.ndarray, float]] = {
                'data': window.copy(),
                'timestamp': chunk_data['timestamp'],
                'duration': self.window_size / self.sample_rate
            }
            self.window_queue.put(window_data)

            # Advance buffer by step size for overlap
            self.buffer = self.buffer[self.step_size:]

        # Limit buffer growth to prevent memory issues
        max_buffer_size: int = self.window_size + self.step_size
        if len(self.buffer) > max_buffer_size:
            self.buffer = self.buffer[-max_buffer_size:]