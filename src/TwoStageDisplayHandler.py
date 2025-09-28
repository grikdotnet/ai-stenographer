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

    def __init__(self, final_queue: queue.Queue, partial_queue: queue.Queue, text_widget: scrolledtext.ScrolledText) -> None:
        self.final_queue: queue.Queue = final_queue
        self.partial_queue: queue.Queue = partial_queue
        self.text_widget: scrolledtext.ScrolledText = text_widget
        self.is_running: bool = False
        self.thread: threading.Thread

        self._setup_text_styles()
        self._initialize_marks()

    def _setup_text_styles(self) -> None:
        """Configure visual styling for preliminary and final text display."""
        self.text_widget.tag_configure("preliminary", foreground="gray", font=("TkDefaultFont", 10, "italic"))
        self.text_widget.tag_configure("final", foreground="black", font=("TkDefaultFont", 10, "normal"))

    def _initialize_marks(self) -> None:
        """Set up text marks for tracking preliminary text boundaries."""
        self.text_widget.mark_set("preliminary_start", tk.END)
        self.text_widget.mark_set("preliminary_end", tk.END)

    def update_preliminary_text(self, text: str) -> None:
        """
        Replace current preliminary text with new recognition results.

        Uses mark-based boundary tracking to replace only the preliminary text
        region while preserving all finalized text. The preliminary text appears
        in gray/italic styling to indicate it may change.
        """
        self.text_widget.delete("preliminary_start", "preliminary_end")
        start_pos = self.text_widget.index("preliminary_start")
        self.text_widget.insert(start_pos, text, "preliminary")
        self.text_widget.mark_set("preliminary_end", tk.END)
        self.text_widget.see(tk.END)

    def finalize_text(self, final_text: str) -> None:
        """
        Convert preliminary text to final permanent text.

        Removes any existing preliminary text and replaces it with finalized
        text in black/normal styling. Resets boundary marks to prepare for
        the next preliminary text sequence.
        """
        self.text_widget.delete("preliminary_start", "preliminary_end")
        start_pos = self.text_widget.index("preliminary_start")
        self.text_widget.insert(start_pos, final_text + " ", "final")
        self.text_widget.mark_set("preliminary_start", tk.END)
        self.text_widget.mark_set("preliminary_end", tk.END)
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