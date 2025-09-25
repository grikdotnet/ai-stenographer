# src/Windower.py
import queue
import threading
import numpy as np
from typing import Dict, Any, Optional

class Windower:
    def __init__(self, chunk_queue: queue.Queue, window_queue: queue.Queue,
                 window_duration: float = 2.0, step_duration: float = 0.5, sample_rate: int = 16000):
        self.chunk_queue: queue.Queue = chunk_queue
        self.window_queue: queue.Queue = window_queue
        self.window_size: int = int(window_duration * sample_rate)
        self.step_size: int = int(step_duration * sample_rate)
        self.sample_rate: int = sample_rate
        self.buffer: np.ndarray = np.array([], dtype=np.float32)
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start processing chunks from queue in a background thread"""
        def _queue_reader() -> None:
            while self.is_running:
                try:
                    chunk_data: Dict[str, Any] = self.chunk_queue.get(timeout=0.1)
                    self.process_chunk(chunk_data)
                except queue.Empty:
                    continue

        self.is_running = True
        self.thread = threading.Thread(target=_queue_reader, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.is_running = False

    def process_chunk(self, chunk_data: Dict[str, Any]) -> None:
        """Process a single chunk and create windows when buffer has enough data"""
        chunk: np.ndarray = chunk_data['data']

        # Accumulate chunk in buffer
        self.buffer = np.concatenate([self.buffer, chunk])

        # Create windows when buffer has sufficient data
        while len(self.buffer) >= self.window_size:
            window: np.ndarray = self.buffer[:self.window_size]

            window_data: Dict[str, Any] = {
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