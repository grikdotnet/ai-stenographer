# src/AudioSource.py
import sounddevice as sd
import queue
import numpy as np
import time

class AudioSource:
    def __init__(self, chunk_queue: queue.Queue, sample_rate=16000, chunk_duration=0.1):
        self.chunk_queue = chunk_queue
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.is_running = False
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback from sounddevice - puts raw chunks in queue"""
        if status:
            print(f"Audio error: {status}")
        
        # Convert int16 to float32 and put in queue
        audio_float = indata[:, 0].astype(np.float32)
        self.chunk_queue.put({
            'data': audio_float.copy(),
            'timestamp': time.time()
        })
    
    def start(self):
        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.chunk_size
        )
        self.stream.start()
    
    def stop(self):
        self.is_running = False
        self.stream.stop()