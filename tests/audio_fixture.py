# tests/audio_fixture.py
import numpy as np

class MockAudioStream:
    """Simulates sounddevice stream for testing"""
    
    def __init__(self, duration=2.0, sample_rate=16000, chunk_duration=0.1):
        # Generate simple test audio (sine wave with pauses to simulate speech)
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create speech-like pattern: tone-silence-tone-silence
        audio = np.zeros_like(t)
        for i in range(4):
            start = int(i * 0.5 * sample_rate)
            end = int((i * 0.5 + 0.3) * sample_rate)
            if end <= len(audio):
                audio[start:end] = np.sin(2 * np.pi * (440 + i*50) * t[start:end]) * 0.5
        
        self.audio = audio.astype(np.float32)
        self.chunk_size = int(sample_rate * chunk_duration)
        self.position = 0
        
    def run_with_callback(self, callback):
        """Simulates sounddevice stream calling the callback"""
        while self.position < len(self.audio):
            # Get chunk
            chunk = self.audio[self.position:self.position + self.chunk_size]
            
            if len(chunk) == self.chunk_size:
                # Reshape to match sounddevice format (N, 1)
                indata = chunk.reshape(-1, 1)
                
                # Call the callback exactly like sounddevice does
                callback(
                    indata=indata,
                    frames=len(indata),
                    time_info=None,
                    status=None
                )
            
            self.position += self.chunk_size