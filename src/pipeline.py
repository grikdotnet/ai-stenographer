import queue
import signal
import sys
import time
import onnx_asr

from .AudioSource import AudioSource
from .Windower import Windower
from .Recognizer import Recognizer
from .TextMatcher import TextMatcher
from .OutputHandlers import FinalTextOutput, PartialTextOutput

class STTPipeline:
    def __init__(self, model_path="./models/parakeet"):
        # Create queues
        self.chunk_queue = queue.Queue(maxsize=100)
        self.window_queue = queue.Queue(maxsize=50)
        self.text_queue = queue.Queue(maxsize=50)
        self.final_queue = queue.Queue(maxsize=50)
        self.partial_queue = queue.Queue(maxsize=50)

        # Load recognition model
        self.model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", model_path)

        # Create components
        self.audio_source = AudioSource(self.chunk_queue)
        self.windower = Windower(self.chunk_queue, self.window_queue)
        self.recognizer = Recognizer(self.window_queue, self.text_queue, self.model)
        self.text_matcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue)
        self.final_output = FinalTextOutput(self.final_queue)
        self.partial_output = PartialTextOutput(self.partial_queue)
        
        # All components
        self.components = [
            self.audio_source, self.windower, self.recognizer,
            self.text_matcher, self.final_output, self.partial_output
        ]
    
    def start(self):
        print("Starting STT Pipeline...")
        for component in self.components:
            component.start()
        print("Pipeline running. Press Ctrl+C to stop.")
    
    def stop(self):
        print("\nStopping pipeline...")
        for component in self.components:
            component.stop()
        print("Pipeline stopped.")
    
    def run(self):
        self.start()
        
        def signal_handler(sig, frame):
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()