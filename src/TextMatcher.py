# src/TextMatcher.py
import queue
import threading
import logging
from typing import Dict, Any, Optional, Union
from .TextNormalizer import TextNormalizer
from .GuiWindow import GuiWindow
from .types import RecognitionResult
from dataclasses import replace


def _format_chunk_ids(chunk_ids: list) -> str:
    return f"[{chunk_ids[0]}-{chunk_ids[-1]}]" if chunk_ids else "[]"


class TextMatcher:
    """Matches overlapping speech recognition results from sliding windows.

    TextMatcher processes sequential speech recognition results that may contain
    overlapping words due to sliding window recognition. Routes preliminary results
    (word-level VAD) directly to GUI, and finalized results (sliding windows) through
    overlap resolution before displaying.

    Args:
        text_queue: Queue to read recognition results from
        gui_window: GUI window instance with update_partial() and finalize_text() methods
        text_normalizer: Optional text normalizer for overlap detection
        time_threshold: Time threshold for duplicate detection (finalized only)
        verbose: Enable verbose logging
    """

    def __init__(self, text_queue: queue.Queue,
                 gui_window: GuiWindow,
                 text_normalizer: Optional[TextNormalizer] = None,
                 time_threshold: float = 0.6,
                 verbose: bool = False
                 ) -> None:
        self.text_queue: queue.Queue = text_queue
        self.gui_window: GuiWindow = gui_window
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.verbose: bool = verbose

        # Enhanced duplicate detection parameters (for finalized path)
        self.text_normalizer: TextNormalizer = text_normalizer if text_normalizer is not None else TextNormalizer()
        self.time_threshold: float = time_threshold

        # State only needed for final/flush paths (overlap resolution)
        # Preliminary path has no state - VAD guarantees distinct segments
        self.previous_result: Optional[RecognitionResult] = None

    def find_word_overlap(self, words1: list[str], words2: list[str]) -> tuple[int, int, int]:
        """Find longest overlapping word sequence between two word lists.

        Args:
            words1: First word list
            words2: Second word list

        Returns:
            Tuple of (start_idx1, start_idx2, common_length)
            - start_idx1: Starting index in words1 where common sequence begins
            - start_idx2: Starting index in words2 where common sequence begins
            - common_length: Length of the common word sequence
        """
        max_length: int = 0
        best_i: int = 0
        best_j: int = 0

        for i in range(len(words1)):
            for j in range(len(words2)):
                # Check for matching sequence starting at positions i, j
                length: int = 0
                while (i + length < len(words1) and
                       j + length < len(words2) and
                       words1[i + length] == words2[j + length]):
                    length += 1

                if length > max_length:
                    max_length = length
                    best_i, best_j = i, j

        return best_i, best_j, max_length

    def resolve_overlap(self, window1: tuple[str, float], window2: tuple[str, float]) -> tuple[str, tuple[str, float]]:
        """Resolve overlap between two recognition windows.

        Uses text normalization to find overlaps even when punctuation or case differs,
        while preserving the original text formatting in the output.

        Args:
            window1: (text_string, timestamp) for first window
            window2: (text_string, timestamp) for second window

        Returns:
            Tuple of (finalized_text, remaining_window)
            - finalized_text: string to add to output (reliable part)
            - remaining_window: (text_string, timestamp) for next iteration

        Raises:
            Exception: if no overlap found between windows
        """
        # Split into original words (preserve original formatting)
        words1: list[str] = window1[0].split()
        words2: list[str] = window2[0].split()

        # Normalize words for comparison (strip punctuation, lowercase, etc.)
        normalized_words1: list[str] = [self.text_normalizer.normalize_text(word) for word in words1]
        normalized_words2: list[str] = [self.text_normalizer.normalize_text(word) for word in words2]

        # Find longest common subsequence using normalized words
        common_start_idx1, common_start_idx2, common_length = self.find_word_overlap(
            normalized_words1, normalized_words2
        )

        if common_length == 0:
            # will be caught in process_finalized()
            raise Exception("No overlap found between windows")

        # Extract finalized part using original words (preserve formatting)
        finalized_words: list[str] = words1[:common_start_idx1 + common_length]
        finalized_text: str = " ".join(finalized_words)

        # Prepare remaining window using original words (preserve formatting)
        remaining_start_idx: int = common_start_idx2 + common_length
        remaining_words: list[str] = words2[remaining_start_idx:]
        remaining_text: str = " ".join(remaining_words)

        return finalized_text, (remaining_text, window2[1])

    def process_text(self, result: RecognitionResult) -> None:
        """Process a single text segment using appropriate path.

        Routes based on status:
        - 'preliminary': Pass through to GUI directly
        - 'final': Overlap resolution, then to GUI
        - 'flush': Trigger finalize_pending() to flush buffered text

        Args:
            result: RecognitionResult with text, timing, and chunk_ids
        """
        if result.status == 'preliminary':
            self.gui_window.update_partial(result)
        elif result.status == 'final':
            self.process_finalized(result)
        elif result.status == 'flush':
            self.process_flush(result)


    def process_finalized(self, result: RecognitionResult) -> None:
        """Process finalized result using window-based overlap resolution.

        Finalized results come from sliding windows with overlap. Uses the
        resolve_overlap algorithm to find reliable common subsequences and
        finalize the reliable parts while discarding boundary errors.

        Args:
            result: RecognitionResult with finalized text, chunk_ids, and confidence
        """
        current_text: str = result.text

        if self.verbose:
            logging.debug(f"TextMatcher.process_finalized() received '{current_text}'")
            logging.debug(f"  chunk_ids={_format_chunk_ids(result.chunk_ids)}")
            logging.debug(f"  confidence={result.confidence:.3f}")
            logging.debug(f"  previous_text: '{self.previous_result.text if self.previous_result else None}'")

        # Enhanced duplicate detection using text normalization
        # This handles punctuation variants ("Hello." vs "Hello?") while preventing
        # false positives when user says the same word multiple times
        if self.previous_result:
            # Normalize both texts for comparison
            current_normalized = self.text_normalizer.normalize_text(current_text)
            previous_normalized = self.text_normalizer.normalize_text(self.previous_result.text)
            time_diff = abs(result.start_time - self.previous_result.start_time)

            if (current_normalized == previous_normalized and
                time_diff < self.time_threshold):
                if self.verbose:
                    logging.debug(f"  duplicate within {time_diff:.3f}s detected, skipping: '{current_text}'")
                    logging.debug(f"  Original: '{self.previous_result.text}' → Normalized: '{previous_normalized}'")
                    logging.debug(f"  Current: '{current_text}' → Normalized: '{current_normalized}'")
                # Skip normalized duplicate - it's likely from overlapping windows
                return

        # Process windows in pairs using the overlap resolution algorithm
        if self.previous_result:
            window1: tuple[str, float] = (self.previous_result.text, result.start_time)
            window2: tuple[str, float] = (current_text, result.start_time)

            try:
                # Resolve overlap between windows
                finalized_text, remaining_window = self.resolve_overlap(window1, window2)

                # Send finalized text to GUI - create RecognitionResult with resolved text from previous window
                if finalized_text.strip():
                    if self.verbose:
                        logging.debug(f"  finalize_text('{finalized_text}') chunk_ids={_format_chunk_ids(self.previous_result.chunk_ids)}")

                    finalized_result = replace(self.previous_result,text=finalized_text)
                    self.gui_window.finalize_text(finalized_result)

                self.previous_result = replace(result,text=remaining_window[0])
                
            except Exception:
                # No overlap found. This is a poor recognition result, just show the previous result as is.
                if self.verbose:
                    logging.debug(f"  no overlap, finalize_text('{self.previous_result.text}')")

                self.gui_window.finalize_text(self.previous_result)
                self.previous_result = result

        else:
            # First segment - store for overlap resolution
            if self.verbose:
                logging.debug(f"  first text, storing '{current_text}'")
            self.previous_result = result

    def process(self) -> None:
        """Read texts from queue and process them continuously.

        Runs in a loop reading text recognition results from the input queue
        and processing each one using process_text(). Continues until stop()
        is called.
        """
        while self.is_running:
            try:
                result: RecognitionResult = self.text_queue.get(timeout=0.1)
                self.process_text(result)
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

    def finalize_pending(self) -> None:
        """Finalize any remaining partial finalized text.
        Called explicitly by Pipeline.stop() or self.process_flush().
        """
        if self.previous_result:
            if self.verbose:
                logging.debug(f"TextMatcher.finalize_pending() previous finalized text: '{self.previous_result.text}'")

            # Finalize the pending result directly
            self.gui_window.finalize_text(self.previous_result)
            self.previous_result = None

    def process_flush(self, result: RecognitionResult) -> None:
        """Process flush signal by finalizing any pending text.
        Flush signals indicate end of speech segment (e.g., silence timeout).

        Args:
            result: RecognitionResult with status='flush'
        """
        if self.verbose:
            logging.debug(f"TextMatcher.process_flush() text='{result.text}', pending='{self.previous_result.text if self.previous_result else None}'")

        if result.text and result.text.strip():
            final_result = replace(result, status='final')
            self.process_finalized(final_result)

        self.finalize_pending()

    def stop(self) -> None:
        """Stop the background processing thread.

        Sets the running flag to False, causing the background thread
        to exit its processing loop.
        """
        self.is_running = False