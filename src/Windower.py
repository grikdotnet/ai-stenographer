# src/Windower.py
import queue
import threading
import numpy as np

class Windower:
    def __init__(self, chunk_queue: queue.Queue, window_queue: queue.Queue,
                 window_duration=2.0, step_duration=0.5, sample_rate=16000):
        self.chunk_queue = chunk_queue
        self.window_queue = window_queue
        self.window_size = int(window_duration * sample_rate)
        self.step_size = int(step_duration * sample_rate)
        self.sample_rate = sample_rate
        self.buffer = np.array([], dtype=np.float32)
        self.is_running = False
        
    def start(self):
        """Start processing chunks from queue in a background thread"""
        def _queue_reader():
            while self.is_running:
                try:
                    chunk_data = self.chunk_queue.get(timeout=0.1)
                    self.process_chunk(chunk_data)
                except queue.Empty:
                    continue

        self.is_running = True
        self.thread = threading.Thread(target=_queue_reader, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.is_running = False

    def process_chunk(self, chunk_data):
        """Process a single chunk and create windows when buffer has enough data"""
        chunk = chunk_data['data']

        # Accumulate chunk in buffer
        self.buffer = np.concatenate([self.buffer, chunk])

        # Create windows when buffer has sufficient data
        while len(self.buffer) >= self.window_size:
            window = self.buffer[:self.window_size]

            self.window_queue.put({
                'data': window.copy(),
                'timestamp': chunk_data['timestamp'],
                'duration': self.window_size / self.sample_rate
            })

            # Advance buffer by step size for overlap
            self.buffer = self.buffer[self.step_size:]

        # Limit buffer growth to prevent memory issues
        max_buffer_size = self.window_size + self.step_size
        if len(self.buffer) > max_buffer_size:
            self.buffer = self.buffer[-max_buffer_size:]