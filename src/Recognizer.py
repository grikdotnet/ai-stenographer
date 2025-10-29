# src/Recognizer.py
import queue
import threading
import logging
import numpy as np
from typing import Any, Optional
from src.types import ChunkQueueItem, RecognitionResult

class Recognizer:
    """Pure speech recognition: AudioSegment → RecognitionResult

    Single Responsibility: Convert audio to text using STT model.
    Does not handle silence detection or text finalization logic.

    Args:
        speech_queue: Queue to read AudioSegment instances from (both preliminary and finalized)
        text_queue: Queue to write RecognitionResult instances to
        model: Pre-loaded speech recognition model with recognize() method
        verbose: Enable verbose logging
    """

    def __init__(self, speech_queue: queue.Queue,
                 text_queue: queue.Queue,
                 model: Any,
                 verbose: bool = False
                 ) -> None:
        self.speech_queue: queue.Queue = speech_queue
        self.text_queue: queue.Queue = text_queue
        self.model: Any = model
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.verbose: bool = verbose

    def recognize_window(self, window_data: ChunkQueueItem) -> Optional[RecognitionResult]:
        """Recognize audio and emit RecognitionResult.

        Maps AudioSegment.type to RecognitionResult.status:
        - 'preliminary' → 'preliminary'
        - 'finalized' → 'final'
        - 'flush' → 'flush'

        Args:
            window_data: AudioSegment to recognize

        Returns:
            RecognitionResult if text recognized, None if empty/silence
        """
        audio: np.ndarray = window_data.data

        # Map AudioSegment.type to RecognitionResult.status
        status_map = {
            'preliminary': 'preliminary',
            'finalized': 'final',
            'flush': 'flush'
        }
        status = status_map[window_data.type]

        # Recognize
        text: str = self.model.recognize(audio)

        if self.verbose:
            logging.debug(f"recognize(): chunk_ids={window_data.chunk_ids}")
            logging.debug(f"recognize(): text: '{text}'")

        if not text or not text.strip():
            return None

        # Create RecognitionResult with mapped status
        result = RecognitionResult(
            text=text,
            start_time=window_data.start_time,
            end_time=window_data.end_time,
            status=status,
            chunk_ids=window_data.chunk_ids
        )
        self.text_queue.put(result)
        return result

    def process(self) -> None:
        """Process AudioSegments from queue continuously.

        Reads AudioSegment instances and converts them to RecognitionResult instances.
        Status is determined automatically from AudioSegment.type.
        Runs until stop() is called.
        """
        while self.is_running:
            try:
                window_data: ChunkQueueItem = self.speech_queue.get(timeout=0.1)
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