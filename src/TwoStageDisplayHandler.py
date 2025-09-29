# src/OutputHhandlers.py
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext
from typing import Any

class TwoStageDisplayHandler:
    """
    Manages two-stage text display for real-time speech recognition.

    Implements a visual display system where preliminary speech recognition results
    appear in gray/italic text that updates in real-time, and finalized results
    appear in black/normal text that becomes permanent. Uses tkinter text marks
    to manage boundaries between preliminary and final text regions.
    """

    def __init__(self, final_queue: queue.Queue, partial_queue: queue.Queue, text_widget: scrolledtext.ScrolledText, root: tk.Tk = None) -> None:
        self.final_queue: queue.Queue = final_queue
        self.partial_queue: queue.Queue = partial_queue
        self.text_widget: scrolledtext.ScrolledText = text_widget
        self.root: tk.Tk = root
        self.is_running: bool = False
        self.thread: threading.Thread
        self.preliminary_start_pos: str = "1.0"
        self.last_finalized_text: str = ""  # Track last finalized text to prevent duplicates

        self._setup_text_styles()

    def _is_main_thread(self) -> bool:
        """Check if we're running on the main thread."""
        import threading
        return threading.current_thread() is threading.main_thread()

    def _setup_text_styles(self) -> None:
        """Configure visual styling for preliminary and final text display."""
        self.text_widget.tag_configure("preliminary", foreground="gray", font=("TkDefaultFont", 10, "italic"))
        self.text_widget.tag_configure("final", foreground="black", font=("TkDefaultFont", 10, "normal"))

    def update_preliminary_text(self, text: str) -> None:
        """
        Replace current preliminary text with new recognition results.

        Uses position tracking to replace only the preliminary text region
        while preserving all finalized text. The preliminary text appears
        in gray/italic styling to indicate it may change.
        """
        if self.root and not self._is_main_thread():
            # Schedule GUI update on main thread when called from background thread
            self.root.after(0, self._update_preliminary_text_safe, text)
        else:
            # Direct call when on main thread or in tests
            self._update_preliminary_text_safe(text)

    def _update_preliminary_text_safe(self, text: str) -> None:
        """Thread-safe implementation of preliminary text update."""
        # Verify preliminary_start_pos is valid, reset if corrupted
        try:
            self.text_widget.index(self.preliminary_start_pos)
        except tk.TclError:
            # Position is invalid, reset to end of final text
            self.preliminary_start_pos = self.text_widget.index("end-1c")

        # Delete any existing preliminary text from the stored position to end
        self.text_widget.delete(self.preliminary_start_pos, "end-1c")

        # Insert new preliminary text at the stored position
        self.text_widget.insert(self.preliminary_start_pos, text, "preliminary")

        # Reset the last finalized text tracker since we have new preliminary text
        self.last_finalized_text = ""

        self.text_widget.see(tk.END)

    def finalize_text(self, final_text: str) -> None:
        """
        Convert preliminary text to final permanent text.

        Removes any existing preliminary text and replaces it with finalized
        text in black/normal styling. Updates position tracking to prepare for
        the next preliminary text sequence.
        """
        if self.root and not self._is_main_thread():
            # Schedule GUI update on main thread when called from background thread
            self.root.after(0, self._finalize_text_safe, final_text)
        else:
            # Direct call when on main thread or in tests
            self._finalize_text_safe(final_text)

    def _finalize_text_safe(self, final_text: str) -> None:
        """Thread-safe implementation of text finalization."""
        # Prevent duplicate finalization of the same text
        if final_text == self.last_finalized_text:
            return

        # Verify preliminary_start_pos is valid, reset if corrupted
        try:
            self.text_widget.index(self.preliminary_start_pos)
        except tk.TclError:
            # Position is invalid, reset to end of final text
            self.preliminary_start_pos = self.text_widget.index("end-1c")

        # Get the current preliminary text (if any) to check if finalization is needed
        current_end = self.text_widget.index("end-1c")
        preliminary_text = ""

        # Only proceed if there's actually preliminary text to finalize
        if self.text_widget.compare(self.preliminary_start_pos, "!=", current_end):
            preliminary_text = self.text_widget.get(self.preliminary_start_pos, current_end)

            # Replace preliminary text with finalized text
            self.text_widget.delete(self.preliminary_start_pos, current_end)
            final_with_space = final_text + " "
            self.text_widget.insert(self.preliminary_start_pos, final_with_space, "final")
            self.preliminary_start_pos = self.text_widget.index(f"{self.preliminary_start_pos}+{len(final_with_space)}c")
            self.last_finalized_text = final_text
        else:
            # No preliminary text exists - check if we should add final text
            # Only add if this is a different final text than the last one
            if final_text != self.last_finalized_text:
                final_with_space = final_text + " "
                self.text_widget.insert(self.preliminary_start_pos, final_with_space, "final")
                self.preliminary_start_pos = self.text_widget.index(f"{self.preliminary_start_pos}+{len(final_with_space)}c")
                self.last_finalized_text = final_text

        self.text_widget.see(tk.END)

    def process_queues(self) -> None:
        """
        Main processing loop that handles both final and partial text queues.

        Continuously monitors both queues with short timeouts to provide
        real-time updates. Processes finalization events and preliminary
        text updates as they arrive from the speech recognition pipeline.
        """
        while self.is_running:
            try:
                final_data = self.final_queue.get(timeout=0.05)
                text = final_data['text']
                self.finalize_text(text)
            except queue.Empty:
                pass

            try:
                partial_data = self.partial_queue.get(timeout=0.05)
                text = partial_data['text']
                self.update_preliminary_text(text)
            except queue.Empty:
                pass

    def start(self) -> None:
        self.is_running = True
        self.thread = threading.Thread(target=self.process_queues, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.is_running = False