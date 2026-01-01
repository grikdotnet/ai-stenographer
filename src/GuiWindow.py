import tkinter as tk
from tkinter import scrolledtext
from typing import List, Set
from src.types import RecognitionResult


class GuiWindow:
    """Manages two-stage text display for real-time speech recognition.

    Provides update_partial() and finalize_text() methods for displaying
    preliminary (gray/italic) and final (black/normal) text in a tkinter widget.
    Uses chunk-ID tracking for accurate partial finalization.

    Text widget must be pre-configured with 'preliminary' and 'final' tags
    before passing to GuiWindow constructor.
    """
    PARAGRAPH_PAUSE_THRESHOLD: float = 2.0  # seconds

    def __init__(self, text_widget: scrolledtext.ScrolledText, root: tk.Tk = None) -> None:
        self.text_widget: scrolledtext.ScrolledText = text_widget
        self.root: tk.Tk = root
        self.preliminary_start_pos: str = "1.0"
        self.last_finalized_text: str = ""
        self.last_finalized_end_time: float = 0.0

        # Chunk-ID-based tracking
        self.preliminary_results: List[RecognitionResult] = []
        self.finalized_chunk_ids: Set[int] = set()
        self.finalized_text: str = ""

    def _is_main_thread(self) -> bool:
        """Check if we're running on the main thread."""
        import threading
        return threading.current_thread() is threading.main_thread()

    def update_partial(self, result: RecognitionResult) -> None:
        """Append new preliminary text.

        Args:
            result: RecognitionResult object with chunk_ids
        """
        if self.root and not self._is_main_thread():
            # Schedule GUI update on main thread when called from background thread
            self.root.after(0, self._update_partial_safe, result)
        else:
            # Direct call when on main thread or in tests
            self._update_partial_safe(result)

    def _update_partial_safe(self, result: RecognitionResult) -> None:
        """Thread-safe implementation of preliminary text update.

        Stores RecognitionResult object and appends text to widget.
        Detects audio pauses and inserts paragraph breaks when pause exceeds threshold.
        """
        self.preliminary_results.append(result)
        
        if (self.last_finalized_end_time != 0
            and (result.start_time - self.last_finalized_end_time) >= self.PARAGRAPH_PAUSE_THRESHOLD
            and self.last_finalized_text != ""
            and self.finalized_text and not self.finalized_text.endswith('\n')):
                # Pause detected - insert paragraph break and re-render
                self.finalized_text += "\n\n"
                self._rerender_all()
                self.text_widget.see(tk.END)
                return

        # Verify preliminary_start_pos is valid, reset if corrupted
        try:
            self.text_widget.index(self.preliminary_start_pos)
        except tk.TclError:
            # Position is invalid, reset to end of text
            self.preliminary_start_pos = self.text_widget.index("end-1c")

        # Get current end position to check if there's existing preliminary text
        current_end = self.text_widget.index("end-1c")

        # Check if there's existing preliminary text
        if self.text_widget.compare(self.preliminary_start_pos, "!=", current_end):
            text_to_insert = " " + result.text
        else:
            # No existing preliminary text, insert without space
            text_to_insert = result.text

        # Insert new preliminary text at the end
        self.text_widget.insert(tk.END, text_to_insert, "preliminary")

        # Reset the last finalized text tracker since we have new preliminary text
        self.last_finalized_text = ""

        self.text_widget.see(tk.END)

    def finalize_text(self, result: RecognitionResult) -> None:
        """Convert preliminary text to final permanent text.

        Uses chunk-ID tracking to finalize only matched chunks and preserve
        unmatched preliminary text. Re-renders finalized + remaining preliminary.

        Args:
            result: RecognitionResult with chunk_ids
        """
        if self.root and not self._is_main_thread():
            self.root.after(0, self._finalize_text_safe, result)
        else:
            self._finalize_text_safe(result)

    def _finalize_text_safe(self, result: RecognitionResult) -> None:
        """Thread-safe implementation of text finalization using chunk-ID tracking.

        Marks chunks as finalized, re-renders all text (finalized + remaining preliminary).
        """
        # Prevent duplicate finalization of the same text
        if result.text == self.last_finalized_text:
            return

        self.finalized_chunk_ids.update(result.chunk_ids)

        # Append to finalized text
        if self.finalized_text:
            self.finalized_text += " " + result.text
        else:
            self.finalized_text = result.text

        self._rerender_all()

        self.last_finalized_text = result.text
        
        # Track timing for pause detection
        self.last_finalized_end_time = result.end_time  

        self.text_widget.see(tk.END)

    def _rerender_all(self) -> None:
        """Re-render all text: finalized (black) + remaining preliminary (gray).

        Filters out preliminary results whose chunks have been finalized.
        """
        self.text_widget.delete("1.0", tk.END)

        if self.finalized_text:
            self.text_widget.insert(tk.END, self.finalized_text + " ", "final")

        # Update preliminary_start_pos to end of finalized text
        self.preliminary_start_pos = self.text_widget.index("end-1c")

        # Filter out preliminary results whose chunks have been finalized
        remaining = [
            r for r in self.preliminary_results
            if not any(cid in self.finalized_chunk_ids for cid in r.chunk_ids)
        ]

        # Add remaining preliminary text (gray)
        for i, r in enumerate(remaining):
            separator = " " if i > 0 else ""
            self.text_widget.insert(tk.END, separator + r.text, "preliminary")