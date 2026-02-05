# src/asr/Recognizer.py
import queue
import threading
import logging
import numpy as np
from typing import Any, Optional, TYPE_CHECKING
from ..types import ChunkQueueItem, RecognitionResult

if TYPE_CHECKING:
    from onnx_asr.adapters import TimestampedResultsAsrAdapter
    from src.ApplicationState import ApplicationState
from onnx_asr.asr import TimestampedResult


def _format_chunk_ids(chunk_ids: list) -> str:
    return f"[{chunk_ids[0]}-{chunk_ids[-1]}]" if chunk_ids else "[]"


class Recognizer:
    """Pure speech recognition: AudioSegment → RecognitionResult

    Single Responsibility: Convert audio to text using STT model.
    Does not handle silence detection or text finalization logic.

    Observer Pattern: Subscribes to ApplicationState for shutdown events.
    Components can be stopped via observer notification.

    Args:
        speech_queue: Queue to read AudioSegment instances from (both preliminary and finalized)
        text_queue: Queue to write RecognitionResult instances to
        model: Pre-loaded speech recognition model with recognize() method (timestamped)
        sample_rate: Audio sample rate in Hz (default: 16000)
        app_state: ApplicationState for observer pattern (REQUIRED)
        verbose: Enable verbose logging
    """

    # Filler words to exclude when including previous complete word before data_start
    FILLER_WORDS = {' um', ' oh', ' uh', ' ah'}

    def __init__(self, speech_queue: queue.Queue,
                 text_queue: queue.Queue,
                 model: "TimestampedResultsAsrAdapter",
                 sample_rate: int = 16000,
                 app_state: 'ApplicationState' = None,
                 verbose: bool = False
                 ) -> None:
        self.speech_queue: queue.Queue = speech_queue
        self.text_queue: queue.Queue = text_queue
        self.model: "TimestampedResultsAsrAdapter" = model
        self.sample_rate: int = sample_rate
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.verbose: bool = verbose
        self.app_state: 'ApplicationState' = app_state
        self.dropped_results: int = 0

        # Register as component observer
        self.app_state.register_component_observer(self.on_state_change)

    def recognize_window(self, window_data: ChunkQueueItem) -> Optional[RecognitionResult]:
        """Recognize audio with context and filter tokens by timestamp.

        Algorithm:
        1. Concatenate left_context + data + right_context for recognition
        2. Recognize with timestamps using TimestampedResultsAsrAdapter
        3. Calculate data region boundaries in seconds
        4. Filter tokens to only include those within data region
        5. Return RecognitionResult with filtered text and original timing
           (confidence fields populated with defaults until new API integration)

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

        # Confidence extraction not yet implemented with new API
        token_confidences = []

        # Calculate context boundaries in seconds
        data_start = len(window_data.left_context) / self.sample_rate
        data_duration = len(window_data.data) / self.sample_rate
        data_end = data_start + data_duration

        if self.verbose:
            logging.debug(f"recognize() dump:")
            logging.debug(f"  type: {window_data.type}")
            logging.debug(f"  chunk_ids={_format_chunk_ids(window_data.chunk_ids)}")
            logging.debug(f"  audio duration: {duration_with_context}")
            logging.debug(f"  data_start: {data_start}")
            logging.debug(f"  data_duration: {data_duration}")
            logging.debug(f"  text: '{result.text}'")
            logging.debug(f"  tokens: {result.tokens}")
            logging.debug(f"  timestamps: {result.timestamps}")

        if not result.text or not result.text.strip():
            return None

        # Filter tokens and confidences within data region
        filtered_text, filtered_confidences = self._filter_tokens_with_confidence(
            result.text,
            result.tokens,
            result.timestamps,
            token_confidences,
            data_start,
            data_end
        )

        if not filtered_text or not filtered_text.strip():
            return None

        # Compute segment-level confidence statistics
        avg_confidence = float(np.mean(filtered_confidences)) if filtered_confidences else 0.0
        audio_rms = float(np.sqrt(np.mean(window_data.data ** 2)))
        confidence_variance = float(np.var(filtered_confidences)) if len(filtered_confidences) > 1 else 0.0

        if self.verbose:
            logging.debug(f"  filtered text: '{filtered_text}'")
            logging.debug(f"  audio_rms: {audio_rms:.3f}")

        recognition_result = RecognitionResult(
            text=filtered_text,
            start_time=window_data.start_time,
            end_time=window_data.end_time,
            status=status,
            chunk_ids=window_data.chunk_ids,
            confidence=avg_confidence,
            token_confidences=filtered_confidences,
            audio_rms=audio_rms,
            confidence_variance=confidence_variance
        )
        self._put_result_nonblocking(recognition_result)
        return recognition_result

    def _put_result_nonblocking(self, result: RecognitionResult) -> None:
        """Enqueue recognition result to text_queue with drop-on-full behavior."""
        try:
            self.text_queue.put_nowait(result)
        except queue.Full:
            self.dropped_results += 1
            if self.verbose:
                logging.warning(
                    f"Recognizer: text_queue full, dropped result "
                    f"(total drops: {self.dropped_results})"
                )

    def _filter_tokens_with_confidence(self,
                                        text: str,
                                        tokens: Optional[list[str]],
                                        timestamps: Optional[list[float]],
                                        confidences: list[float],
                                        data_start: float,
                                        data_end: float) -> tuple[str, list[float]]:
        """Filter tokens and their confidences to only those within data region (exclude context).

        Algorithm:
        1. If no timestamps available, return full text with empty confidences (fallback)
        2. Timestamp filtering
        3. Backtrack for split words (prevents word corruption from boundary cuts):
           - Parakeet splits words into subword tokens: [" O", "ne"] for "One"
           - Word-start tokens have leading space, continuation tokens do not
           - If first filtered token lacks leading space, backtrack to find word-start token
           - Include all tokens from word-start through original filtered set
        4. Include previous complete word if first filtered token is complete:
           - If first filtered token has space prefix (complete word)
           - Check if previous token also has space prefix (also complete word)
           - Exclude filler words (um, oh, uh, ah) - case-insensitive check
           - Prepend previous token to capture short words at speech start
        5. Reconstruct text from filtered tokens (tokens use space prefix for word boundaries)
        6. Strip and return (text, confidences)

        Args:
            text: Full recognized text
            tokens: Token list from TimestampedResult (subword units with space prefix)
            timestamps: Timestamp list (in seconds, relative to audio start)
            confidences: Confidence scores parallel to tokens
            data_start: Start of data region in seconds
            data_end: End of data region in seconds

        Returns:
            tuple: (filtered_text, filtered_confidences)
        """
        if not tokens or not timestamps:
            # No timestamps available - return full text with empty confidences (fallback)
            return text, []

        # Step 1: Strict timestamp filtering (data_start <= ts <= data_end)
        filtered_tokens = []
        filtered_confidences = []

        has_confidences = len(confidences) == len(tokens)

        for i, (token, ts) in enumerate(zip(tokens, timestamps)):
            if data_start <= ts <= data_end:
                filtered_tokens.append(token)
                if has_confidences:
                    filtered_confidences.append(confidences[i])

        if not filtered_tokens:
            return "", []

         # Include previous complete word (if not a filler word)
        if filtered_tokens and filtered_tokens[0].startswith(' '):
            first_filtered_idx = tokens.index(filtered_tokens[0])
            if first_filtered_idx > 0:
                prev_token = tokens[first_filtered_idx - 1]
                if prev_token.startswith(' '):
                    if prev_token.lower() not in self.FILLER_WORDS:
                        filtered_tokens.insert(0, prev_token)
                        if has_confidences:
                            filtered_confidences.insert(0, confidences[first_filtered_idx - 1])

       # Backtrack for split words
        if filtered_tokens and not filtered_tokens[0].startswith(' '):
            first_filtered_idx = tokens.index(filtered_tokens[0])

            word_start_idx = first_filtered_idx
            for idx in range(first_filtered_idx - 1, -1, -1):
                if tokens[idx].startswith(' '):
                    word_start_idx = idx
                    break

            if word_start_idx < first_filtered_idx:
                for idx in range(word_start_idx, first_filtered_idx):
                    filtered_tokens.insert(idx - word_start_idx, tokens[idx])
                    if has_confidences:
                        filtered_confidences.insert(idx - word_start_idx, confidences[idx])

        # Reconstruct text from filtered tokens
        reconstructed = ''.join(filtered_tokens)
        reconstructed = reconstructed.strip()

        return reconstructed, filtered_confidences

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
        """Stop the background processing thread and cleanup resources.

        Sets the running flag to False, causing the background thread
        to exit its processing loop.

        This method is idempotent - safe to call multiple times.
        """
        if not self.is_running:
            return

        self.is_running = False

    def on_state_change(self, old_state: str, new_state: str) -> None:
        """Handle application state changes.

        Observes ApplicationState and reacts to state transitions:
        - * -> shutdown: stop processing and cleanup resources

        Args:
            old_state: Previous state
            new_state: New state
        """
        if new_state == 'shutdown':
            self.stop()
