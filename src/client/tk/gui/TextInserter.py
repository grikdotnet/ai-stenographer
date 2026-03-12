import threading
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.client.tk.gui.KeyboardSimulator import KeyboardSimulator
    from src.types import RecognitionResult


class TextInserter:
    """Subscribes for recognized speech to RecognitionResultPublisher
and inserts finalized text into the active application via keyboard simulation.

    Implements TextRecognitionSubscriber protocol (structural typing).
    Thread-safe - callbacks may come from background threads.

    Only finalized text is inserted. Partial/preliminary results are ignored
    because they are unstable and may change.

    Example:
        >>> inserter = TextInserter(keyboard_simulator)
        >>> inserter.enable()
        >>> # Now when RecognitionResultPublisher publishes finalized results,
        >>> # the text will be typed into the active window
    """

    def __init__(self, keyboard_simulator: 'KeyboardSimulator', verbose: bool = False) -> None:
        """Initialize TextInserter.

        Args:
            keyboard_simulator: Keyboard simulation backend for typing text
            verbose: Enable verbose logging
        """
        self._keyboard_simulator = keyboard_simulator
        self._verbose = verbose
        self._enabled = False
        self._lock = threading.Lock()

    def enable(self) -> None:
        """Enable text insertion.
        Finalized recognition results will be typed into the active application window.
        """
        with self._lock:
            self._enabled = True
            if self._verbose:
                logging.info("TextInserter enabled")

    def disable(self) -> None:
        """Disable text insertion, recognition results are ignored."""
        with self._lock:
            self._enabled = False
            if self._verbose:
                logging.info("TextInserter disabled")

    def is_enabled(self) -> bool:
        """Check if text insertion is currently enabled."""
        with self._lock:
            return self._enabled

    def on_partial_update(self, result: 'RecognitionResult') -> None:
        # Intentionally do nothing - partial results are unstable
        pass

    def on_finalization(self, result: 'RecognitionResult') -> None:
        """
        Insert the recognized text into the active application if:
        - Insertion is enabled
        - Text is not empty or whitespace-only

        A trailing space is added after each insertion for word separation.

        Args:
            result: Finalized RecognitionResult
        """
        # Prevent deadlock if keyboard simulation blocks
        with self._lock:
            enabled = self._enabled

        if not enabled:
            return

        text = result.text.strip()
        if not text:
            return

        self._keyboard_simulator.type_text(result.text + " ")

        if self._verbose:
            logging.info(f"TextInserter: inserted '{result.text}'")
