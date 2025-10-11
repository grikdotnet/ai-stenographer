import tkinter as tk
from tkinter import scrolledtext
from typing import Tuple, List, Set, Union
from src.types import RecognitionResult
from src.GuiFactory import GuiFactory


class GuiWindow:
    """Manages two-stage text display for real-time speech recognition.

    Provides update_partial() and finalize_text() methods for displaying
    preliminary (gray/italic) and final (black/normal) text in a tkinter widget.
    Uses chunk-ID tracking for accurate partial finalization.
    """

    def __init__(self, text_widget: scrolledtext.ScrolledText, root: tk.Tk = None) -> None:
        self.text_widget: scrolledtext.ScrolledText = text_widget
        self.root: tk.Tk = root
        self.preliminary_start_pos: str = "1.0"
        self.last_finalized_text: str = ""

        # Chunk-ID-based tracking
        self.preliminary_results: List[RecognitionResult] = [] 
        self.finalized_chunk_ids: Set[int] = set()
        self.finalized_text: str = ""

        self._setup_text_styles()

    def _is_main_thread(self) -> bool:
        """Check if we're running on the main thread."""
        import threading
        return threading.current_thread() is threading.main_thread()

    def _setup_text_styles(self) -> None:
        """Configure visual styling for preliminary and final text display."""
        self.text_widget.tag_configure("preliminary", foreground="gray", font=("TkDefaultFont", 10, "italic"))
        self.text_widget.tag_configure("final", foreground="black", font=("TkDefaultFont", 10, "normal"))

    def update_partial(self, result: Union[str, RecognitionResult]) -> None:
        """Append new preliminary text to existing preliminary text.

        Appends new recognition results to the preliminary text region
        while preserving all finalized text. The preliminary text appears
        in gray/italic styling to indicate it may change.

        Args:
            result: RecognitionResult object with chunk_ids, or str for backward compatibility
        """
        # Convert str to RecognitionResult for backward compatibility with old tests
        if isinstance(result, str):
            result = RecognitionResult(
                text=result,
                start_time=0.0,
                end_time=0.0,
                status='preliminary',
                chunk_ids=[]
            )

        if self.root and not self._is_main_thread():
            # Schedule GUI update on main thread when called from background thread
            self.root.after(0, self._update_partial_safe, result)
        else:
            # Direct call when on main thread or in tests
            self._update_partial_safe(result)

    def _update_partial_safe(self, result: RecognitionResult) -> None:
        """Thread-safe implementation of preliminary text update.

        Stores RecognitionResult object and appends text to widget.
        """
        # Store RecognitionResult object for chunk-ID tracking
        self.preliminary_results.append(result)

        # Verify preliminary_start_pos is valid, reset if corrupted
        try:
            self.text_widget.index(self.preliminary_start_pos)
        except tk.TclError:
            # Position is invalid, reset to end of final text
            self.preliminary_start_pos = self.text_widget.index("end-1c")

        # Get current end position to check if there's existing preliminary text
        current_end = self.text_widget.index("end-1c")

        # Check if there's existing preliminary text
        if self.text_widget.compare(self.preliminary_start_pos, "!=", current_end):
            # Append with space separator
            text_to_insert = " " + result.text
        else:
            # No existing preliminary text, insert without space
            text_to_insert = result.text

        # Insert new preliminary text at the end
        self.text_widget.insert(tk.END, text_to_insert, "preliminary")

        # Reset the last finalized text tracker since we have new preliminary text
        self.last_finalized_text = ""

        self.text_widget.see(tk.END)

    def finalize_text(self, result: Union[str, RecognitionResult]) -> None:
        """Convert preliminary text to final permanent text.

        Uses chunk-ID tracking to finalize only matched chunks and preserve
        unmatched preliminary text. Re-renders finalized + remaining preliminary.

        Args:
            result: RecognitionResult with chunk_ids, or str for backward compatibility
        """
        # Convert str to RecognitionResult for backward compatibility with old tests
        if isinstance(result, str):
            result = RecognitionResult(
                text=result,
                start_time=0.0,
                end_time=0.0,
                status='final',
                chunk_ids=[]
            )

        if self.root and not self._is_main_thread():
            # Schedule GUI update on main thread when called from background thread
            self.root.after(0, self._finalize_text_safe, result)
        else:
            # Direct call when on main thread or in tests
            self._finalize_text_safe(result)

    def _finalize_text_safe(self, result: RecognitionResult) -> None:
        """Thread-safe implementation of text finalization using chunk-ID tracking.

        Marks chunks as finalized, re-renders all text (finalized + remaining preliminary).
        """
        # Prevent duplicate finalization of the same text
        if result.text == self.last_finalized_text:
            return

        # Mark chunks as finalized (if chunk_ids provided)
        if result.chunk_ids:
            self.finalized_chunk_ids.update(result.chunk_ids)
        else:
            # Old behavior: clear all preliminary when no chunk_ids (backward compatibility)
            self.preliminary_results = []

        # Append to finalized text
        if self.finalized_text:
            self.finalized_text += " " + result.text
        else:
            self.finalized_text = result.text

        # Re-render everything
        self._rerender_all()

        # Update last finalized tracker
        self.last_finalized_text = result.text

        self.text_widget.see(tk.END)

    def _rerender_all(self) -> None:
        """Re-render all text: finalized (black) + remaining preliminary (gray).

        Filters out preliminary results whose chunks have been finalized.
        """
        # Clear everything
        self.text_widget.delete("1.0", tk.END)

        # Add finalized text (black)
        if self.finalized_text:
            self.text_widget.insert(tk.END, self.finalized_text + " ", "final")

        # Update preliminary_start_pos to end of finalized text
        self.preliminary_start_pos = self.text_widget.index("end-1c")

        # Find preliminary results with unfinalized chunks
        # Handle backward compatibility: if chunk_ids are empty (old tests), filter by matching text
        if self.finalized_chunk_ids:
            # New behavior: filter by chunk_ids
            remaining = [
                r for r in self.preliminary_results
                if not any(cid in self.finalized_chunk_ids for cid in r.chunk_ids)
            ]
        else:
            # Old behavior: clear all preliminary when finalizing (no chunk tracking)
            remaining = []

        # Add remaining preliminary text (gray)
        for i, r in enumerate(remaining):
            separator = " " if i > 0 else ""
            self.text_widget.insert(tk.END, separator + r.text, "preliminary")


def create_stt_window() -> Tuple[tk.Tk, scrolledtext.ScrolledText]:
    """Creates and returns a configured tkinter window with text widget for STT display."""
    # Use GuiFactory to create main window
    root = GuiFactory.create_window("Speech-to-Text Display", "800x600")

    # Create main frame
    main_frame = tk.Frame(root, padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Create title label
    title_label = tk.Label(main_frame, text="Real-time Speech Recognition",
                          font=("Arial", 16, "bold"))
    title_label.pack(pady=(0, 10))

    # Create status label
    status_label = tk.Label(main_frame, text="Gray text: Preliminary (may change) | Black text: Final",
                           font=("Arial", 10), fg="gray")
    status_label.pack(pady=(0, 10))

    # Use GuiFactory to create scrolled text widget
    text_widget = GuiFactory.create_scrolled_text(
        main_frame,
        wrap=tk.WORD,
        width=80,
        height=30,
        font=("Arial", 12),
        bg="white",
        fg="black"
    )
    text_widget.pack(fill=tk.BOTH, expand=True)

    # Make text widget read-only for users but allow programmatic updates
    text_widget.config(state=tk.NORMAL)

    return root, text_widget


def run_gui_loop(root: tk.Tk) -> None:
    """Starts the tkinter main loop."""
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            root.quit()
        except tk.TclError:
            pass
        try:
            root.destroy()
        except tk.TclError:
            pass