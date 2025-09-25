# src/Recognizer.py
import queue
import threading
import onnx_asr
import numpy as np
from typing import Dict, Any, Optional, Union, TypedDict

class Recognizer:
    """Performs speech recognition on audio windows using an ONNX model.

    Recognizer processes audio windows from a queue and converts them to text
    using a pre-loaded speech recognition model. Only non-empty text results
    are placed in the output queue along with timing information.

    Args:
        window_queue: Queue to read audio windows from
        text_queue: Queue to write recognition results to
        model: Pre-loaded speech recognition model with recognize() method
    """

    def __init__(self, window_queue: queue.Queue, text_queue: queue.Queue, model: Any):
        self.window_queue: queue.Queue = window_queue
        self.text_queue: queue.Queue = text_queue
        self.model: Any = model
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None

    def recognize_window(self, window_data: Dict[str, Any]) -> None:
        """Process a single window and add recognized text to queue.

        Runs speech recognition on the audio data in the window and adds
        the result to the output queue if the recognized text is non-empty.
        Silent windows (empty text) are filtered out.

        Args:
            window_data: Dictionary containing 'data' (audio numpy array),
                        'timestamp', and 'duration' fields
        """
        audio: np.ndarray = window_data['data']

        # Recognize
        text: str = self.model.recognize(audio)

        if text.strip():
            text_data: Dict[str, Union[str, float]] = {
                'text': text,
                'timestamp': window_data['timestamp'],
                'window_duration': window_data['duration']
            }
            self.text_queue.put(text_data)

    def process(self) -> None:
        """Read windows from queue and process them.

        Continuously reads window data from the input queue and processes
        each window using recognize_window(). Runs until stop() is called.
        Uses a timeout to prevent blocking indefinitely on empty queues.
        """
        while self.is_running:
            try:
                window_data: Dict[str, Any] = self.window_queue.get(timeout=0.1)
                self.recognize_window(window_data)
            except queue.Empty:
                continue

    def start(self) -> None:
        """Start processing windows in a background thread.

        Creates and starts a daemon thread that continuously processes
        windows from the input queue using the process() method.
        """
        self.is_running = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the background processing thread.

        Sets the running flag to False, causing the background thread
        to exit its processing loop.
        """
        self.is_running = False