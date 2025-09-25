# src/OutputHhandlers.py
import queue
import threading

class FinalTextOutput:
    def __init__(self, final_queue: queue.Queue):
        self.final_queue = final_queue
        self.is_running = False
        self.full_transcript = []
        
    def process(self):
        while self.is_running:
            try:
                text_data = self.final_queue.get(timeout=0.1)
                text = text_data['text']
                
                # Output finalized text
                print(f"[FINAL] {text}")
                self.full_transcript.append(text)
                
            except queue.Empty:
                continue
    
    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.is_running = False
        print("\nFull transcript:", ' '.join(self.full_transcript))


class PartialTextOutput:
    def __init__(self, partial_queue: queue.Queue):
        self.partial_queue = partial_queue
        self.is_running = False
        
    def process(self):
        while self.is_running:
            try:
                text_data = self.partial_queue.get(timeout=0.1)
                text = text_data['text']
                
                # Output partial text (could be grayed out in GUI)
                print(f"[PARTIAL] {text}", end='\r')
                
            except queue.Empty:
                continue
    
    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.is_running = False