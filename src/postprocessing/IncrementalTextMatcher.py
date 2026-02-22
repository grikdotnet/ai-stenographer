# src/postprocessing/IncrementalTextMatcher.py
import queue
import threading
import logging
from typing import Optional, TYPE_CHECKING
from .TextNormalizer import TextNormalizer
from ..types import MatcherQueueItem, RecognitionResult, SpeechEndSignal
from ..RecognitionResultPublisher import RecognitionResultPublisher
from dataclasses import replace

if TYPE_CHECKING:
    from src.ApplicationState import ApplicationState


def _format_chunk_ids(chunk_ids: list) -> str:
    return f"[{chunk_ids[0]}-{chunk_ids[-1]}]" if chunk_ids else "[]"


class IncrementalTextMatcher:
    """Matches incremental recognition results from growing windows.

    IncrementalTextMatcher processes sequential recognition results from
    growing windows. Uses word overlap detection to identify stable (finalized)
    vs unstable (preliminary) regions.

    Observer Pattern:
    - Subscribes to ApplicationState for shutdown events
    - Publishes recognition results via RecognitionResultPublisher

    Args:
        text_queue: Queue to read recognition results from
        publisher: RecognitionResultPublisher for distributing results to subscribers
        text_normalizer: Optional text normalizer for overlap detection
        app_state: ApplicationState for observer pattern (REQUIRED)
        verbose: Enable verbose logging
    """

    def __init__(self, text_queue: queue.Queue,
                 publisher: RecognitionResultPublisher,
                 text_normalizer: Optional[TextNormalizer] = None,
                 app_state: 'ApplicationState' = None,
                 verbose: bool = False
                 ) -> None:
        self.text_queue: queue.Queue = text_queue
        self.publisher: RecognitionResultPublisher = publisher
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.verbose: bool = verbose
        self.app_state: 'ApplicationState' = app_state

        # Text normalization for overlap detection
        self.text_normalizer: TextNormalizer = text_normalizer if text_normalizer is not None else TextNormalizer()

        # State for incremental matching
        self.previous_result: Optional[RecognitionResult] = None
        self._prev_normalized_words: Optional[list[str]] = None
        self.prev_finalized_words: int = 0 # Number of words in previous_result that were finalized

        # Register as component observer
        self.app_state.register_component_observer(self.on_state_change)

    def _split_and_normalize(self, text: str) -> list[str]:
        """Split text into words and normalize each for overlap comparison.

        Args:
            text: Input text to split and normalize

        Returns:
            List of normalized words
        """
        return [self.text_normalizer.normalize_text(w) for w in text.split()]

    def _prefix_match_length(self, curr_normalized: list[str]) -> int:
        """Return the common prefix length between previous and current normalized words.

        Args:
            curr_normalized: Pre-normalized words for the current result.

        Returns:
            Number of words that match at the start of both sequences.
        """
        prev = self._prev_normalized_words or []
        length = 0
        while (length < len(prev) and length < len(curr_normalized)
               and prev[length] == curr_normalized[length]):
            length += 1
        return length

    def find_word_overlap(self, curr_normalized: list[str]) -> tuple[int, int, int]:
        """Find longest overlapping word sequence between previous and current result.

        Uses cached normalized words for self.previous_result and the provided
        normalized words for the current result.

        Args:
            curr_normalized: Pre-normalized words for current result.text

        Returns:
            Tuple of (start_idx_prev, start_idx_curr, common_length)
            - start_idx_prev: Starting index in previous result where common sequence begins
            - start_idx_curr: Starting index in current result where common sequence begins
            - common_length: Length of the common word sequence
        """
        if self._prev_normalized_words is None:
            self._prev_normalized_words = self._split_and_normalize(self.previous_result.text)

        max_length: int = 0
        best_i: int = 0
        best_j: int = 0

        for i in range(len(self._prev_normalized_words)):
            for j in range(len(curr_normalized)):
                length: int = 0
                while (i + length < len(self._prev_normalized_words) and
                       j + length < len(curr_normalized) and
                       self._prev_normalized_words[i + length] == curr_normalized[j + length]):
                    length += 1

                if length > max_length:
                    max_length = length
                    best_i, best_j = i, j

        return best_i, best_j, max_length

    def process_item(self, item: MatcherQueueItem) -> None:
        """Route matcher input item by message type."""
        if isinstance(item, RecognitionResult):
            self.process_incremental(item)
            return
        self.process_speech_end(item)

    def process_incremental(self, result: RecognitionResult) -> None:
        """Process incremental result using prefix comparison for stability detection.

        Algorithm:
        1. If no previous_result: publish entire text as preliminary, store, return.
        2. Split current text into words and normalize.
        3. Determine matching strategy:
           - If either chunk_ids is empty: force reset (no finalization, safe fallback).
           - If chunk_ids[0] matches (same growing window): prefix match at (0, 0).
           - Otherwise (sliding window): find_word_overlap() arbitrary position.
        4. If common_length == 0: reset state, replace preliminary, return.
        5. If common_length > 0:
           a. Stable region ends at overlap_start_curr + common_length in current result.
           b. Newly finalized = stable words beyond prev_finalized_words.
           c. Unstable tail = words after stable region.
           d. Publish finalized and preliminary accordingly.

        Args:
            result: RecognitionResult
        """
        if self.verbose:
            logging.debug(f"IncrementalTextMatcher.process_incremental() received '{result.text}'")
            logging.debug(f"  chunk_ids={_format_chunk_ids(result.chunk_ids)}")

        # First result: all preliminary
        if self.previous_result is None:
            if self.verbose:
                logging.debug(f"  first result, publishing as preliminary")
            self.publisher.publish_partial_update(result)
            self.previous_result = result
            self._prev_normalized_words = self._split_and_normalize(result.text)
            return

        curr_words = result.text.split()
        curr_normalized = self._split_and_normalize(result.text)

        if self.verbose:
            logging.debug(f"  previous finalized_word_count={self.prev_finalized_words}")
            logging.debug(f"  current:  '{result.text}'")

        prev_start = self.previous_result.chunk_ids[0] if self.previous_result.chunk_ids else None
        curr_start = result.chunk_ids[0] if result.chunk_ids else None
        same_start = prev_start is not None and curr_start is not None and prev_start == curr_start
        either_empty = prev_start is None or curr_start is None

        if either_empty:
            overlap_start_prev, overlap_start_curr, common_length = 0, 0, 0
        elif same_start:
            overlap_start_prev, overlap_start_curr, common_length = 0, 0, self._prefix_match_length(curr_normalized)
        else:
            overlap_start_prev, overlap_start_curr, common_length = self.find_word_overlap(curr_normalized)

        if self.verbose:
            logging.debug(f"  overlap: idx_prev={overlap_start_prev}, idx_curr={overlap_start_curr}, length={common_length}")

        # No match: replace preliminary entirely
        if common_length == 0:
            if self.verbose:
                logging.debug(f"  no overlap, resetting word count")
            self.publisher.publish_partial_update(result)
            self.previous_result = result
            self._prev_normalized_words = curr_normalized
            self.prev_finalized_words = 0
            return

        stable_end = overlap_start_curr + common_length

        # garbled prefix or stability shrinks at the start
        # - shifted/garbled prefix: overlap_start_curr > 0 and overlaps earlier than finalized count
        # - shrinking prefix: overlap_start_curr == 0 and stable_end < prev_finalized_words
        if overlap_start_curr < self.prev_finalized_words and (overlap_start_curr > 0 or stable_end < self.prev_finalized_words):
            if self.verbose:
                logging.debug(
                    f"  overlap shifted (idx_curr={overlap_start_curr}), "
                    f"  prev_finalized_words = {self.prev_finalized_words}"
                )
            self.prev_finalized_words = overlap_start_curr

        # Newly finalized words: from prev_finalized_words to stable_end
        # Note: words before overlap_start_curr are ignored (unreliable after sliding)
        finalize_start = max(self.prev_finalized_words, overlap_start_curr)
        if stable_end > finalize_start:
            newly_finalized_words = curr_words[finalize_start:stable_end]
            newly_finalized_text = ' '.join(newly_finalized_words)

            if newly_finalized_text.strip():
                if self.verbose:
                    logging.debug(f"  finalization('{newly_finalized_text}') words {finalize_start} to {stable_end}")

                finalized_result = replace(result, text=newly_finalized_text)
                self.publisher.publish_finalization(finalized_result)

            self.prev_finalized_words = stable_end

        # Unstable tail: from stable_end onwards
        unstable_words = curr_words[stable_end:]
        if unstable_words:
            unstable_text = ' '.join(unstable_words)
            if self.verbose:
                logging.debug(f"  preliminary('{unstable_text}')")

            preliminary_result = replace(result, text=unstable_text)
            self.publisher.publish_partial_update(preliminary_result)

        self.previous_result = result
        self._prev_normalized_words = curr_normalized

    def process_speech_end(self, signal: SpeechEndSignal) -> None:
        """Finalize pending text when receiving utterance boundary.

        Algorithm:
        1. Finalize any remaining non-finalized text.
        2. Reset state for next utterance.

        Args:
            signal: End-of-speech boundary.
        """
        if self.verbose:
            logging.debug(
                f"IncrementalTextMatcher.process_speech_end() utterance_id={signal.utterance_id}, "
                f"pending='{self.previous_result.text if self.previous_result else None}'"
            )

        if self.previous_result and self.prev_finalized_words < len(self.previous_result.text.split()):
            words = self.previous_result.text.split()
            remaining_words = words[self.prev_finalized_words:]

            if remaining_words:
                remaining_text = ' '.join(remaining_words)
                if self.verbose:
                    logging.debug(f"IncrementalTextMatcher.finalize_pending() finalizing '{remaining_text}'")

                finalized_result = replace(self.previous_result, text=remaining_text)
                self.publisher.publish_finalization(finalized_result)

        self.previous_result = None
        self._prev_normalized_words = None
        self.prev_finalized_words = 0

    def process(self) -> None:
        """Read texts from queue and process them continuously.

        Runs in a loop reading text recognition results from the input queue
        and processing each one using process_item(). Continues until stop()
        is called.
        """
        while self.is_running:
            try:
                item: MatcherQueueItem = self.text_queue.get(timeout=0.1)
                self.process_item(item)
            except queue.Empty:
                continue

    def start(self) -> None:
        """Start processing texts in a background thread.

        Creates and starts a daemon thread that continuously processes
        text recognition results from the input queue.
        """
        self.is_running = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the background processing thread.

        Sets the running flag to False, causing the background thread
        to exit its processing loop.
        """
        if not self.is_running:
            return

        self.is_running = False

    def on_state_change(self, old_state: str, new_state: str) -> None:
        """Observes ApplicationState and reacts to shutdown."""
        if new_state == 'shutdown':
            self.stop()
            # Finalize any pending preliminary text before shutdown.
            if self.previous_result and self.prev_finalized_words < len(self.previous_result.text.split()):
                words = self.previous_result.text.split()
                remaining_words = words[self.prev_finalized_words:]
                if remaining_words:
                    remaining_text = ' '.join(remaining_words)
                    finalized_result = replace(self.previous_result, text=remaining_text)
                    self.publisher.publish_finalization(finalized_result)

            # Reset state to avoid leaking pending text across sessions.
            self.previous_result = None
            self._prev_normalized_words = None
            self.prev_finalized_words = 0
