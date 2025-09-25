# realtime_stt_local.py
import sounddevice as sd
import numpy as np
import onnx_asr
import queue
import threading
import time

class LocalSTT:
    def __init__(self, model_path="./models/parakeet"):
        print("Loading local model...")
        self.model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", model_path)
        self.audio_queue = queue.Queue()
        self.is_running = False
        
    def audio_callback(self, indata, frames, time_info, status):
        """Capture audio chunks"""
        if status:
            print(f"Audio error: {status}")
        self.audio_queue.put(indata.copy())
        
    def process_audio(self):
        """Process audio in separate thread"""
        buffer = []
        
        while self.is_running:
            try:
                # Get audio chunk (non-blocking)
                chunk = self.audio_queue.get(timeout=0.1)
                buffer.append(chunk)
                
                # Process every 2 seconds of audio
                if len(buffer) >= 32:  # ~2 seconds at 16kHz
                    audio_data = np.concatenate(buffer, axis=0).squeeze()
                    
                    # Recognize
                    text = self.model.recognize(audio_data)
                    
                    if text.strip():
                        print(f">>> {text}")
                    
                    # Keep last 8 chunks for context
                    buffer = buffer[-8:]
                    
            except queue.Empty:
                continue
                
    def start(self):
        """Start real-time transcription"""
        print("🎤 Starting real-time STT (Press Ctrl+C to stop)...")
        self.is_running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio, daemon=True)
        process_thread.start()
        
        # Start audio stream
        with sd.InputStream(
            samplerate=16000,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(16000 * 0.1)  # 100ms chunks
        ):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n👋 Stopping...")
                self.is_running = False

if __name__ == "__main__":
    stt = LocalSTT(model_path="./models/parakeet")
    stt.start()