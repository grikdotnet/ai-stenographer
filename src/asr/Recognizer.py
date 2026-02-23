# src/asr/Recognizer.py
import queue
import threading
import logging
import numpy as np
from typing import Optional, TYPE_CHECKING
from ..types import (
    AudioSegment,
    RecognitionResult,
    RecognitionTextMessage,
    RecognizerAck,
    RecognizerOutputItem,
)

if TYPE_CHECKING:
    from onnx_asr.adapters import TimestampedResultsAsrAdapter
    from src.ApplicationState import ApplicationState
from onnx_asr.asr import TimestampedResult


def _format_chunk_ids(chunk_ids: list) -> str:
    return f"[{chunk_ids[0]}-{chunk_ids[-1]}]" if chunk_ids else "[]"


class Recognizer:
    """Pure speech recognition: AudioSegment -> recognizer output protocol.

    Single Responsibility: Convert audio to text using STT model.
    Does not handle silence detection or text finalization logic.

    Observer Pattern: Subscribes to ApplicationState for shutdown events.
    Components can be stopped via observer notification.

    Args:
        input_queue: Queue to read AudioSegment instances from
        output_queue: Queue to write RecognitionTextMessage/RecognizerAck
        model: Pre-loaded speech recognition model with recognize() method (timestamped)
        sample_rate: Audio sample rate in Hz (default: 16000)
        app_state: ApplicationState for observer pattern (REQUIRED)
        verbose: Enable verbose logging
    """

    FILLER_WORDS: frozenset[str] = frozenset({"um", "uh", "ah", "oh", "yeah", "mm-hmm", "okay"})

    def __init__(self,
                 *,
                 model: Optional["TimestampedResultsAsrAdapter"] = None,
                 input_queue: queue.Queue | None = None,
                 output_queue: queue.Queue | None = None,
                 sample_rate: int = 16000,
                 app_state: 'ApplicationState' = None,
                 verbose: bool = False,
                 ) -> None:
        self.input_queue: queue.Queue = input_queue
        self.output_queue: queue.Queue = output_queue
        if model is None:
            raise ValueError("Recognizer requires model")
        if self.input_queue is None or self.output_queue is None:
            raise ValueError("Recognizer requires input/output queues")
        self.model: "TimestampedResultsAsrAdapter" = model
        self.sample_rate: int = sample_rate
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.verbose: bool = verbose
        self.app_state: 'ApplicationState' = app_state
        self.dropped_results: int = 0

        # Register as component observer
        self.app_state.register_component_observer(self.on_state_change)

    def _extract_token_confidences(self, result: TimestampedResult) -> list[float]:
        """Convert per-token log-probabilities to probability confidences in [0, 1].

        Algorithm:
        1. Return [] if logprobs is None, empty, or length-mismatched with tokens.
        2. Return [] if any logprob value is non-finite (nan or inf).
        3. Convert each logprob: conf = 1.0 if lp >= 0 else exp(lp), clamp to [0.0, 1.0].
           Positive logprobs (should not occur in practice) are clamped to 1.0 to avoid
           overflow in exp() and to keep semantics consistent.

        Args:
            result: TimestampedResult containing logprobs and tokens

        Returns:
            Per-token confidence list parallel to result.tokens, or [] on any fallback condition.
        """
        logprobs = result.logprobs
        tokens = result.tokens

        if logprobs is None or len(logprobs) == 0:
            return []
        if not tokens or len(logprobs) != len(tokens):
            return []
        if not all(np.isfinite(lp) for lp in logprobs):
            return []

        return [float(np.clip(1.0 if lp >= 0 else np.exp(lp), 0.0, 1.0)) for lp in logprobs]

    def recognize_window(self, window_data: AudioSegment) -> Optional[RecognitionResult]:
        """Recognize audio with context and filter tokens by timestamp.

        Algorithm:
        1. Concatenate left_context + data + right_context for recognition
        2. Recognize with timestamps using TimestampedResultsAsrAdapter
        3. Calculate data region boundaries in seconds
        4. Filter tokens to only include those within data region
        5. Extract per-token confidences from logprobs via exp(logprob) → [0,1].
           Falls back to [] if logprobs is None, misaligned, empty, or contains
           non-finite values. Filters confidences in lockstep with token filtering.

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

        # Recognize with timestamps
        result: TimestampedResult = self.model.recognize(full_audio)
        duration_with_context = len(full_audio) / self.sample_rate

        token_confidences = self._extract_token_confidences(result)

        data_start = len(window_data.left_context) / self.sample_rate

        if self.verbose:
            logging.debug(f"recognize() dump:")
            logging.debug(f"  type: incremental")
            logging.debug(f"  chunk_ids={_format_chunk_ids(window_data.chunk_ids)}")
            logging.debug(f"  audio duration: {duration_with_context}")
            logging.debug(f"  data_start: {data_start}")
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
            data_start
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
            chunk_ids=window_data.chunk_ids,
            confidence=avg_confidence,
            token_confidences=filtered_confidences,
            audio_rms=audio_rms,
            confidence_variance=confidence_variance,
        )
        return recognition_result

    def _put_output_nonblocking(self, item: RecognizerOutputItem | RecognitionResult) -> None:
        """Enqueue recognizer output message with drop-on-full behavior."""
        try:
            self.output_queue.put_nowait(item)
        except queue.Full:
            self.dropped_results += 1
            if self.verbose:
                logging.warning(
                    f"Recognizer: output_queue full, dropped message "
                    f"(total drops: {self.dropped_results})"
                )

    def _filter_tokens_with_confidence(self,
                                        text: str,
                                        tokens: Optional[list[str]],
                                        timestamps: Optional[list[float]],
                                        confidences: list[float],
                                        data_start: float) -> tuple[str, list[float]]:
        """Filter tokens and their confidences to only those within data region (exclude context).

        Algorithm:
        1. If tokens/timestamps missing or lengths mismatched: return (text, []) as fallback.
        2. Group tokens into words by index. A word starts at index 0 or when the token has
           a leading space. Word timestamp = timestamp of its first token.
        3. For each word, include if:
           - word_ts >= data_start  (in-range), OR
           - word_ts < data_start AND normalized word not in FILLER_WORDS  (left non-filler)
        4. If no words selected: return ('', []).
        5. Reconstruct text: join tokens in original order, strip, return with aligned confidences.

        Args:
            text: Full recognized text
            tokens: Token list from TimestampedResult (subword units with space prefix)
            timestamps: Timestamp list (in seconds, relative to audio start)
            confidences: Confidence scores parallel to tokens
            data_start: Start of data region in seconds

        Returns:
            tuple: (filtered_text, filtered_confidences)
        """
        if not tokens or not timestamps or len(tokens) != len(timestamps):
            return text, []

        has_confidences = len(confidences) == len(tokens)

        words: list[list[int]] = []
        for i, token in enumerate(tokens):
            if i == 0 or token.startswith(' '):
                words.append([i])
            else:
                words[-1].append(i)

        final_indices: list[int] = []
        for word_indices in words:
            word_ts = timestamps[word_indices[0]]
            if word_ts >= data_start:
                final_indices.extend(word_indices)
            else:
                normalized = ''.join(tokens[i] for i in word_indices).strip().lower()
                if normalized not in self.FILLER_WORDS:
                    final_indices.extend(word_indices)

        if not final_indices:
            return "", []

        filtered_tokens = [tokens[i] for i in final_indices]
        filtered_confidences = [confidences[i] for i in final_indices] if has_confidences else []
        return ''.join(filtered_tokens).strip(), filtered_confidences

    def process(self) -> None:
        """Process AudioSegments from queue continuously.

        For each consumed segment:
        - emit RecognitionTextMessage if non-empty text recognized
        - always emit terminal RecognizerAck
        Runs until stop() is called.
        """
        while self.is_running:
            try:
                window_data: AudioSegment = self.input_queue.get(timeout=0.1)
                message_id = window_data.message_id
                if message_id is None:
                    self._put_output_nonblocking(
                        RecognizerAck(message_id=-1, ok=False, error="missing message_id")
                    )
                    continue

                try:
                    recognition_result = self.recognize_window(window_data)
                    if recognition_result is not None:
                        self._put_output_nonblocking(
                            RecognitionTextMessage(
                                result=recognition_result,
                                message_id=message_id
                            )
                        )
                    self._put_output_nonblocking(RecognizerAck(message_id=message_id, ok=True))
                except Exception as exc:
                    self._put_output_nonblocking(
                        RecognizerAck(message_id=message_id, ok=False, error=str(exc))
                    )
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
