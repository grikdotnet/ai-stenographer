# src/Recognizer.py
import queue
import threading
import numpy as np
from typing import Any, Optional
from src.types import ChunkQueueItem, RecognitionResult

class Recognizer:
    """Pure speech recognition: AudioSegment → RecognitionResult

    Single Responsibility: Convert audio to text using STT model.
    Does not handle silence detection or text finalization logic.

    Args:
        chunk_queue: Queue to read AudioSegment instances from (both preliminary and finalized)
        text_queue: Queue to write RecognitionResult instances to
        model: Pre-loaded speech recognition model with recognize() method
        verbose: Enable verbose logging
    """

    def __init__(self, chunk_queue: queue.Queue,
                 text_queue: queue.Queue,
                 model: Any,
                 verbose: bool = False
                 ) -> None:
        self.chunk_queue: queue.Queue = chunk_queue
        self.text_queue: queue.Queue = text_queue
        self.model: Any = model
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.verbose: bool = verbose

    def recognize_window(self, window_data: ChunkQueueItem, is_preliminary: bool = False) -> Optional[RecognitionResult]:
        """Recognize audio and emit RecognitionResult.

        Args:
            window_data: AudioSegment to recognize
            is_preliminary: True for instant results, False for high-quality

        Returns:
            RecognitionResult if text recognized, None if empty/silence
        """
        audio: np.ndarray = window_data.data

        # Recognize
        text: str = self.model.recognize(audio)

        if self.verbose:
            print(f"recognize(): chunk_ids={window_data.chunk_ids}")
            print(f"recognize(): text: '{text}'")

        if not text or not text.strip():
            return None

        # Create RecognitionResult with type information
        result = RecognitionResult(
            text=text,
            start_time=window_data.start_time,
            end_time=window_data.end_time,
            is_preliminary=is_preliminary
        )
        self.text_queue.put(result)
        return result

    def process(self) -> None:
        """Process AudioSegments from queue continuously.

        Reads AudioSegment instances and converts them to RecognitionResult instances.
        Determines preliminary vs finalized based on segment type.
        Runs until stop() is called.
        """
        while self.is_running:
            try:
                window_data: ChunkQueueItem = self.chunk_queue.get(timeout=0.1)
                # Determine if preliminary based on type field
                is_preliminary: bool = (window_data.type == 'preliminary')
                self.recognize_window(window_data, is_preliminary=is_preliminary)
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