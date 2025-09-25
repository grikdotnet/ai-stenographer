# src/Recognizer.py
import queue
import threading
import onnx_asr

class Recognizer:
    def __init__(self, window_queue: queue.Queue, text_queue: queue.Queue, 
                 model_path="./models/parakeet"):
        self.window_queue = window_queue
        self.text_queue = text_queue
        self.model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", model_path)
        self.is_running = False
        
    def process(self):
        """Read windows, recognize text"""
        while self.is_running:
            try:
                window_data = self.window_queue.get(timeout=0.1)
                audio = window_data['data']
                
                # Recognize
                text = self.model.recognize(audio)
                
                if text.strip():
                    self.text_queue.put({
                        'text': text,
                        'timestamp': window_data['timestamp'],
                        'window_duration': window_data['duration']
                    })
                    
            except queue.Empty:
                continue
    
    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.is_running = False