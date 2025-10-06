# src/TextMatcher.py
import queue
import threading
from typing import Dict, Any, Optional, Union
from .TextNormalizer import TextNormalizer

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
                 gui_window,
                 text_normalizer: Optional[TextNormalizer] = None,
                 time_threshold: float = 0.6,
                 verbose: bool = False
                 ) -> None:
        self.text_queue: queue.Queue = text_queue
        self.gui_window = gui_window  # GuiWindow instance
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.verbose: bool = verbose

        # Enhanced duplicate detection parameters (for finalized path)
        self.text_normalizer: TextNormalizer = text_normalizer if text_normalizer is not None else TextNormalizer()
        self.time_threshold: float = time_threshold

        # State only needed for finalized path (overlap resolution)
        # Preliminary path has no state - VAD guarantees distinct segments
        self.previous_finalized_text: str = ""
        self.previous_finalized_timestamp: float = 0.0

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
            raise Exception("No overlap found between windows")

        # Extract finalized part using original words (preserve formatting)
        finalized_words: list[str] = words1[:common_start_idx1 + common_length]
        finalized_text: str = " ".join(finalized_words)

        # Prepare remaining window using original words (preserve formatting)
        remaining_start_idx: int = common_start_idx2 + common_length
        remaining_words: list[str] = words2[remaining_start_idx:]
        remaining_text: str = " ".join(remaining_words)

        return finalized_text, (remaining_text, window2[1])

    def process_text(self, text_data: Dict[str, Union[str, float, bool]]) -> None:
        """Process a single text segment using appropriate path.

        Routes to preliminary or finalized processing based on is_preliminary flag.

        Args:
            text_data: Dictionary containing 'text', 'timestamp', 'is_preliminary'
        """
        is_preliminary = text_data.get('is_preliminary', False)

        if is_preliminary:
            self.process_preliminary(text_data)
        else:
            self.process_finalized(text_data)

    def process_preliminary(self, text_data: Dict[str, Union[str, float, bool]]) -> None:
        """Process preliminary result - no overlap resolution needed.

        Preliminary results come from word-level VAD segmentation with clean
        boundaries. They don't overlap, so we skip the overlap resolution
        logic and go straight to GUI.

        Args:
            text_data: Dictionary containing 'text' and 'timestamp'
        """
        current_text: str = text_data['text']

        if self.verbose:
            print(f"TextMatcher.process_preliminary() received '{current_text}'")

        # No duplicate detection needed - VAD guarantees distinct segments
        # If duplicates appear, it's a bug in AudioSource/Recognizer

        # Direct GUI call (no overlap resolution needed)
        if self.verbose:
            print(f"TextMatcher.process_preliminary() → update_partial('{current_text}')")
        self.gui_window.update_partial(current_text)

    def process_finalized(self, text_data: Dict[str, Union[str, float, bool]]) -> None:
        """Process finalized result using window-based overlap resolution.

        Finalized results come from sliding windows with overlap. Uses the
        resolve_overlap algorithm to find reliable common subsequences and
        finalize the reliable parts while discarding boundary errors.

        Args:
            text_data: Dictionary containing 'text' and 'timestamp'
        """
        current_text: str = text_data['text']

        if self.verbose:
            print(f"TextMatcher.process_finalized() current_text: received '{current_text}'")
            print(f"TextMatcher.process_finalized() previous_text: '{self.previous_finalized_text}'")

        # Enhanced duplicate detection using text normalization
        # This handles punctuation variants ("Hello." vs "Hello?") while preventing
        # false positives when user says the same word multiple times
        if self.previous_finalized_text and self.previous_finalized_timestamp > 0:
            # Normalize both texts for comparison
            current_normalized = self.text_normalizer.normalize_text(current_text)
            previous_normalized = self.text_normalizer.normalize_text(self.previous_finalized_text)
            time_diff = abs(text_data['timestamp'] - self.previous_finalized_timestamp)

            if (current_normalized == previous_normalized and
                time_diff < self.time_threshold):
                if self.verbose:
                    print(f"TextMatcher.process_finalized() duplicate within {time_diff:.3f}s detected, skipping: '{current_text}'")
                    print(f"  Original: '{self.previous_finalized_text}' → Normalized: '{previous_normalized}'")
                    print(f"  Current: '{current_text}' → Normalized: '{current_normalized}'")
                # Skip normalized duplicate - it's likely from overlapping windows
                return

        # Process windows in pairs using the overlap resolution algorithm
        if self.previous_finalized_text:
            window1: tuple[str, float] = (self.previous_finalized_text, text_data['timestamp'])
            window2: tuple[str, float] = (current_text, text_data['timestamp'])

            try:
                # Resolve overlap between windows
                finalized_text, remaining_window = self.resolve_overlap(window1, window2)

                # Send finalized text to GUI
                if finalized_text.strip():
                    if self.verbose:
                        print(f"TextMatcher.process_finalized(): finalize_text('{finalized_text}')")
                    self.gui_window.finalize_text(finalized_text)

                # Send remaining part as partial to GUI
                if remaining_window[0].strip():
                    if self.verbose:
                        print(f"TextMatcher.process_finalized(): update_partial('{remaining_window[0]}')")
                    self.gui_window.update_partial(remaining_window[0])

                # Update state with remaining window text
                self.previous_finalized_text = remaining_window[0]
                self.previous_finalized_timestamp = text_data['timestamp']

            except Exception:
                # No overlap found - finalize previous and start new
                if self.verbose:
                    print(f"TextMatcher.process_finalized(): no overlap, finalize_text('{self.previous_finalized_text}')")
                self.gui_window.finalize_text(self.previous_finalized_text)

                # Current text becomes partial
                if self.verbose:
                    print(f"TextMatcher.process_finalized(): update_partial('{current_text}')")
                self.gui_window.update_partial(current_text)

                self.previous_finalized_text = current_text
                self.previous_finalized_timestamp = text_data['timestamp']
        else:
            # First text - all partial
            if self.verbose:
                print(f"TextMatcher.process_finalized(): first text, update_partial('{current_text}')")
            self.gui_window.update_partial(current_text)
            self.previous_finalized_text = current_text
            self.previous_finalized_timestamp = text_data['timestamp']

    def process(self) -> None:
        """Read texts from queue and process them continuously.

        Runs in a loop reading text recognition results from the input queue
        and processing each one using process_text(). Continues until stop()
        is called.
        """
        while self.is_running:
            try:
                text_data: Dict[str, Union[str, float]] = self.text_queue.get(timeout=0.1)
                self.process_text(text_data)
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

        Called explicitly by Pipeline.stop() or external observers
        to flush pending partial text to GUI as final.

        This implements explicit finalization control, allowing
        external components to trigger finalization without
        TextMatcher managing its own timeout logic (SRP).
        """
        if self.previous_finalized_text:
            if self.verbose:
                print(f"TextMatcher.finalize_pending(): finalize_text('{self.previous_finalized_text}')")
            self.gui_window.finalize_text(self.previous_finalized_text)
            self.previous_finalized_text = ""
            self.previous_finalized_timestamp = 0.0

    def stop(self) -> None:
        """Stop the background processing thread.

        Sets the running flag to False, causing the background thread
        to exit its processing loop.
        """
        self.is_running = False