# src/TextMatcher.py
import queue
import threading

class TextMatcher:
    def __init__(self, text_queue: queue.Queue, 
                 final_queue: queue.Queue, partial_queue: queue.Queue):
        self.text_queue = text_queue
        self.final_queue = final_queue
        self.partial_queue = partial_queue
        self.is_running = False
        
        # State for matching
        self.previous_text = ""
        self.finalized_text = ""
        
    def find_overlap(self, text1, text2):
        """Find overlapping words between two texts"""
        words1 = text1.split()
        words2 = text2.split()
        
        # Find longest common subsequence at boundary
        max_overlap = min(len(words1), len(words2))
        overlap = 0
        
        for i in range(1, max_overlap + 1):
            if words1[-i:] == words2[:i]:
                overlap = i
                
        return overlap
    
    def process(self):
        """Match overlapping text segments"""
        while self.is_running:
            try:
                text_data = self.text_queue.get(timeout=0.1)
                current_text = text_data['text']
                
                # Find overlap with previous
                if self.previous_text:
                    overlap = self.find_overlap(self.previous_text, current_text)
                    
                    if overlap > 0:
                        # Remove overlapping part
                        new_words = current_text.split()[overlap:]
                        new_text = ' '.join(new_words)
                        
                        # First part is now finalized
                        finalized_part = ' '.join(self.previous_text.split()[:-overlap])
                        if finalized_part and finalized_part != self.finalized_text:
                            self.final_queue.put({
                                'text': finalized_part,
                                'timestamp': text_data['timestamp']
                            })
                            self.finalized_text = finalized_part
                        
                        # Rest is partial
                        self.partial_queue.put({
                            'text': new_text,
                            'timestamp': text_data['timestamp']
                        })
                    else:
                        # No overlap - previous is final
                        if self.previous_text != self.finalized_text:
                            self.final_queue.put({
                                'text': self.previous_text,
                                'timestamp': text_data['timestamp']
                            })
                            self.finalized_text = self.previous_text
                        
                        # Current is partial
                        self.partial_queue.put({
                            'text': current_text,
                            'timestamp': text_data['timestamp']
                        })
                else:
                    # First text - all partial
                    self.partial_queue.put({
                        'text': current_text,
                        'timestamp': text_data['timestamp']
                    })
                
                self.previous_text = current_text
                
            except queue.Empty:
                continue
    
    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.is_running = False