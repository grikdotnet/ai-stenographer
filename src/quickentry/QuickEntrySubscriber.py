import logging
import threading
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.types import RecognitionResult


class QuickEntrySubscriber:

    def __init__(
        self,
        on_text_change: Callable[[str, str], None],
        verbose: bool = False
    ) -> None:
        self._on_text_change = on_text_change
        self._verbose = verbose
        self._lock = threading.Lock()
        self._finalized_text = ""
        self._partial_text = ""
        self._active = False

    def activate(self) -> None:
        with self._lock:
            self._active = True
            self._finalized_text = ""
            self._partial_text = ""

        if self._verbose:
            logging.info("QuickEntrySubscriber: activated")

    def deactivate(self) -> None:
        with self._lock:
            self._active = False

        if self._verbose:
            logging.info("QuickEntrySubscriber: deactivated")

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def get_accumulated_text(self) -> str:
        with self._lock:
            return self._finalized_text

    def get_display_text(self) -> str:
        with self._lock:
            if self._partial_text:
                return f"{self._finalized_text} {self._partial_text}".strip()
            return self._finalized_text

    def on_partial_update(self, result: 'RecognitionResult') -> None:
        with self._lock:
            if not self._active:
                return
            self._partial_text = result.text
            finalized = self._finalized_text
            partial = self._partial_text

        # Callback outside lock to prevent deadlocks
        self._on_text_change(finalized, partial)

        if self._verbose:
            logging.info(f"QuickEntrySubscriber: partial '{result.text}'")

    def on_finalization(self, result: 'RecognitionResult') -> None:
        with self._lock:
            if not self._active:
                return

            if self._finalized_text:
                self._finalized_text += " " + result.text
            else:
                self._finalized_text = result.text

            self._partial_text = ""
            finalized = self._finalized_text

        # Callback outside lock to prevent deadlocks
        self._on_text_change(finalized, "")

        if self._verbose:
            logging.info(f"QuickEntrySubscriber: finalized '{result.text}', "
                        f"total='{finalized}'")
