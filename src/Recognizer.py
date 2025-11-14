# src/Recognizer.py
import queue
import threading
import logging
import numpy as np
from typing import Any, Optional, TYPE_CHECKING
from src.types import ChunkQueueItem, RecognitionResult

if TYPE_CHECKING:
    from onnx_asr.adapters import TimestampedResultsAsrAdapter
from onnx_asr.asr import TimestampedResult


class Recognizer:
    """Pure speech recognition: AudioSegment → RecognitionResult

    Single Responsibility: Convert audio to text using STT model.
    Does not handle silence detection or text finalization logic.

    Args:
        speech_queue: Queue to read AudioSegment instances from (both preliminary and finalized)
        text_queue: Queue to write RecognitionResult instances to
        model: Pre-loaded speech recognition model with recognize() method (timestamped)
        sample_rate: Audio sample rate in Hz (default: 16000)
        verbose: Enable verbose logging
    """

    def __init__(self, speech_queue: queue.Queue,
                 text_queue: queue.Queue,
                 model: "TimestampedResultsAsrAdapter",
                 sample_rate: int = 16000,
                 verbose: bool = False
                 ) -> None:
        self.speech_queue: queue.Queue = speech_queue
        self.text_queue: queue.Queue = text_queue
        self.model: "TimestampedResultsAsrAdapter" = model
        self.sample_rate: int = sample_rate
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.verbose: bool = verbose

    def recognize_window(self, window_data: ChunkQueueItem) -> Optional[RecognitionResult]:
        """Recognize audio with context and filter tokens by timestamp.

        Algorithm:
        1. Concatenate left_context + data + right_context for recognition
        2. Recognize with timestamps using TimestampedResultsAsrAdapter
        3. Calculate data region boundaries in seconds
        4. Filter tokens to only include those within data region
        5. Return RecognitionResult with filtered text and original timing

        Maps AudioSegment.type to RecognitionResult.status:
        - 'preliminary' → 'preliminary'
        - 'finalized' → 'final'
        - 'flush' → 'flush'

        Args:
            window_data: AudioSegment to recognize

        Returns:
            RecognitionResult if text recognized, None if empty/silence
        """
        # Concatenate context + data + context for recognition
        audio_parts = []
        if window_data.left_context.size > 0:
            audio_parts.append(window_data.left_context)
        audio_parts.append(window_data.data)
        if window_data.right_context.size > 0:
            audio_parts.append(window_data.right_context)

        full_audio = np.concatenate(audio_parts) if len(audio_parts) > 1 else window_data.data

        # Map AudioSegment.type to RecognitionResult.status
        status_map = {
            'preliminary': 'preliminary',
            'finalized': 'final',
            'flush': 'flush'
        }
        status = status_map[window_data.type]

        # Recognize with timestamps
        result: TimestampedResult = self.model.recognize(full_audio)
        duration_with_context = len(full_audio) / self.sample_rate

        # Calculate context boundaries in seconds
        data_start = len(window_data.left_context) / self.sample_rate
        data_duration = len(window_data.data) / self.sample_rate
        data_end = data_start + data_duration

        if self.verbose:
            logging.debug(f"recognize() dump:")
            logging.debug(f"  type: {window_data.type}")
            logging.debug(f"  chunk_ids={window_data.chunk_ids}")
            logging.debug(f"  audio duration: {duration_with_context}")
            logging.debug(f"  data_start: {data_start}")
            logging.debug(f"  data_duration: {data_duration}")
            logging.debug(f"  text: '{result.text}'")
            logging.debug(f"  tokens: {result.tokens}")
            logging.debug(f"  timestamps: {result.timestamps}")

        if not result.text or not result.text.strip():
            return None

        # Filter tokens within data region
        filtered_text = self._filter_tokens_by_timestamp(
            result.text,
            result.tokens,
            result.timestamps,
            data_start,
            data_end
        )

        if self.verbose:
            logging.debug(f"  filtered text: '{filtered_text}'")

        if not filtered_text or not filtered_text.strip():
            return None

        # Create RecognitionResult with filtered text, original timing
        recognition_result = RecognitionResult(
            text=filtered_text,
            start_time=window_data.start_time,
            end_time=window_data.end_time,
            status=status,
            chunk_ids=window_data.chunk_ids
        )
        self.text_queue.put(recognition_result)
        return recognition_result

    def _filter_tokens_by_timestamp(self,
                                     text: str,
                                     tokens: Optional[list[str]],
                                     timestamps: Optional[list[float]],
                                     data_start: float,
                                     data_end: float) -> str:
        """Filter tokens to only those within data region (exclude context).

        Algorithm:
        1. If no timestamps available, return full text (fallback)
        2. Filter tokens where (data_start - tolerance) <= timestamp <= data_end
           - Tolerance accounts for VAD warm-up delay that may place first speech chunk in left_context
        3. Reconstruct text from filtered tokens (tokens use space prefix for word boundaries)
        4. Strip and return

        Args:
            text: Full recognized text
            tokens: Token list from TimestampedResult (subword units with space prefix)
            timestamps: Timestamp list (in seconds, relative to audio start)
            data_start: Start of data region in seconds
            data_end: End of data region in seconds

        Returns:
            Filtered text containing only tokens from data region
        """
        if not tokens or not timestamps:
            # No timestamps available - return full text (fallback)
            return text

        # Filter tokens within data boundaries
        # Include tokens slightly before data_start to account for VAD RMS normalization delay
        boundary_tolerance = 0.37
        filtered_tokens = []
        for token, ts in zip(tokens, timestamps):
            if (data_start - boundary_tolerance) <= ts <= data_end:
                filtered_tokens.append(token)

        if not filtered_tokens:
            return ""

        # Reconstruct text from filtered tokens
        reconstructed = ''.join(filtered_tokens)
        reconstructed = reconstructed.strip()

        return reconstructed

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