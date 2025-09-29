# src/TextMatcher.py
import queue
import threading
from typing import Dict, Any, Optional, Union

class TextMatcher:
    """Matches overlapping speech recognition results from sliding windows.

    TextMatcher processes sequential speech recognition results that may contain
    overlapping words due to sliding window recognition. It identifies overlaps,
    finalizes non-overlapping portions, and maintains partial results for
    streaming output.

    Args:
        text_queue: Queue to read recognition results from
        final_queue: Queue to write finalized text segments to
        partial_queue: Queue to write partial/streaming text to
    """

    def __init__(self, text_queue: queue.Queue,
                 final_queue: queue.Queue, partial_queue: queue.Queue, verbose: bool = False):
        self.text_queue: queue.Queue = text_queue
        self.final_queue: queue.Queue = final_queue
        self.partial_queue: queue.Queue = partial_queue
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.verbose: bool = verbose

        # State for matching
        self.previous_text: str = ""
        self.finalized_text: str = ""
        self.previous_text_timestamp: float = 0.0
        

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
        words1: list[str] = window1[0].split()
        words2: list[str] = window2[0].split()

        # Find longest common subsequence of words
        common_start_idx1, common_start_idx2, common_length = self.find_word_overlap(words1, words2)

        if common_length == 0:
            raise Exception("No overlap found between windows")

        # Extract finalized part (start of window1 + common part)
        finalized_words: list[str] = words1[:common_start_idx1 + common_length]
        finalized_text: str = " ".join(finalized_words)

        # Prepare remaining window (part of window2 after common part)
        remaining_start_idx: int = common_start_idx2 + common_length
        remaining_words: list[str] = words2[remaining_start_idx:]
        remaining_text: str = " ".join(remaining_words)

        return finalized_text, (remaining_text, window2[1])

    def process_text(self, text_data: Dict[str, Union[str, float, bool]]) -> None:
        """Process a single text segment using window-based overlap resolution.

        Uses the resolve_overlap algorithm to properly handle STT window overlaps
        by finding reliable common subsequences and finalizing the reliable parts
        while discarding boundary errors.

        Args:
            text_data: Dictionary containing 'text', 'timestamp', and optionally 'is_silence'
        """
        current_text: str = text_data['text']
        is_silence: bool = text_data.get('is_silence', False)

        if self.verbose:
            print(f"process_text() current_text: received '{current_text}'")
            print(f"process_text() previous_text: '{self.previous_text}'")

        # Handle silence signal - finalize any remaining partial text
        if is_silence:
            if self.previous_text:
                final_data: Dict[str, Union[str, float]] = {
                    'text': self.previous_text,
                    'timestamp': text_data['timestamp']
                }
                if self.verbose:
                    print(f"sending to final_queue: {final_data}")
                self.final_queue.put(final_data)
            # Reset state for new phrase after silence
            self.previous_text = ""
            self.previous_text_timestamp = 0.0
            return

        # Handle identical text from overlapping windows (common case)
        # Only skip if text is identical AND timestamps are very close (< 1 second apart)
        # This prevents false positives when user says the same word multiple times
        if (self.previous_text and current_text == self.previous_text and
            self.previous_text_timestamp > 0 and
            abs(text_data['timestamp'] - self.previous_text_timestamp) < 1.0):
            if self.verbose:
                time_diff = abs(text_data['timestamp'] - self.previous_text_timestamp)
                print(f"process_text() identical text within {time_diff:.3f}s detected, skipping: '{current_text}'")
            # Skip identical text - it's likely from overlapping windows
            return

        # Process windows in pairs using the new overlap resolution algorithm
        if self.previous_text:
            window1: tuple[str, float] = (self.previous_text, text_data['timestamp'])
            window2: tuple[str, float] = (current_text, text_data['timestamp'])

            try:
                # Resolve overlap between windows
                finalized_text, remaining_window = self.resolve_overlap(window1, window2)

                # Queue finalized text
                if finalized_text.strip():
                    final_data: Dict[str, Union[str, float]] = {
                        'text': finalized_text,
                        'timestamp': text_data['timestamp']
                    }
                    if self.verbose:
                        print(f"sending to final_queue: {final_data}")
                    self.final_queue.put(final_data)

                # Queue remaining part as partial
                if remaining_window[0].strip():
                    partial_data: Dict[str, Union[str, float]] = {
                        'text': remaining_window[0],
                        'timestamp': remaining_window[1]
                    }
                    if self.verbose:
                        print(f"sending to partial_queue: {partial_data}")
                    self.partial_queue.put(partial_data)

                # Update state with remaining window text
                self.previous_text = remaining_window[0]
                self.previous_text_timestamp = text_data['timestamp']

            except Exception:
                # No overlap found - finalize previous and start new
                final_data: Dict[str, Union[str, float]] = {
                    'text': self.previous_text,
                    'timestamp': text_data['timestamp']
                }
                if self.verbose:
                    print(f"sending to final_queue: {final_data}")
                self.final_queue.put(final_data)

                # Current text becomes partial
                partial_data: Dict[str, Union[str, float]] = {
                    'text': current_text,
                    'timestamp': text_data['timestamp']
                }
                if self.verbose:
                    print(f"sending to partial_queue: {partial_data}")
                self.partial_queue.put(partial_data)

                self.previous_text = current_text
                self.previous_text_timestamp = text_data['timestamp']
        else:
            # First text - all partial
            partial_data: Dict[str, Union[str, float]] = {
                'text': current_text,
                'timestamp': text_data['timestamp']
            }
            if self.verbose:
                print(f"sending to partial_queue: {partial_data}")
            self.partial_queue.put(partial_data)
            self.previous_text = current_text
            self.previous_text_timestamp = text_data['timestamp']

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

    def stop(self) -> None:
        """Stop the background processing thread.

        Sets the running flag to False, causing the background thread
        to exit its processing loop.
        """
        self.is_running = False