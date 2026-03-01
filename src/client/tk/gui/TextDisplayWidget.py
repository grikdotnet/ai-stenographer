import tkinter as tk
from tkinter import scrolledtext
import logging
from src.types import DisplayInstructions


class TextDisplayWidget:
    """Renders DisplayInstructions to tkinter.
    Thread-safe rendering.
    Maintains only GUI state.
    """

    def __init__(self, text_widget: scrolledtext.ScrolledText,
                 root: tk.Tk,
                 verbose: bool = False) -> None:
        """
        Args:
            text_widget: Pre-configured ScrolledText widget
            root: tkinter root window
            verbose: flag
        """
        self.text_widget: scrolledtext.ScrolledText = text_widget
        self.root: tk.Tk = root
        self.verbose: bool = verbose
        self.preliminary_start_pos: str = "1.0"  # GUI state (cursor tracking)

    def apply_instructions(self, instructions: DisplayInstructions) -> None:
        """Schedules on main thread.

        Args:
            instructions: DisplayInstructions from TextFormatter
        """
        if self.root and not self._is_main_thread():
            self.root.after(0, self._apply_instructions_safe, instructions)
        else:
            self._apply_instructions_safe(instructions)

    def _is_main_thread(self) -> bool:
        """Returns:
            True if on main thread, False otherwise
        """
        import threading
        return threading.current_thread() is threading.main_thread()

    def _apply_instructions_safe(self, instructions: DisplayInstructions) -> None:
        """Must be called on main thread.
        Dispatches to appropriate rendering method based on action type.

        Args:
            instructions: DisplayInstructions to render
        """
        if instructions.action == 'update_partial':
            self._render_partial_update(instructions)
        else:
            self._rerender_all(instructions.finalized_text, instructions.preliminary_segments)

        # Scroll to end
        self.text_widget.see(tk.END)

    def _render_partial_update(self, instructions: DisplayInstructions) -> None:
        """Render partial text update.
        Args:
            instructions: DisplayInstructions with pre-formatted text_to_append
        """
        # Verify preliminary_start_pos is valid, reset if corrupted
        try:
            self.text_widget.index(self.preliminary_start_pos)
        except tk.TclError:
            self.preliminary_start_pos = self.text_widget.index("end-1c")

        # append the pre-formatted text
        self.text_widget.insert(tk.END, instructions.text_to_append, "preliminary")

        if self.verbose:
            logging.debug(f"TextDisplayWidget: partial update '{instructions.text_to_append}'")

    def _rerender_all(self, finalized_text: str, preliminary_segments: list) -> None:
        """Re-render all text: finalized (black) + remaining preliminary (gray).

        Args:
            finalized_text: Complete finalized text buffer
            preliminary_segments: List of RecognitionResults not yet finalized
        """
        if self.verbose:
            logging.debug(f"TextDisplayWidget: rerender_all() finalized_len={len(finalized_text)} "
                         f"preliminary_count={len(preliminary_segments)}")

        self.text_widget.delete("1.0", tk.END)

        if finalized_text:
            self.text_widget.insert(tk.END, finalized_text + " ", "final")

        # Update preliminary_start_pos to end of finalized text
        self.preliminary_start_pos = self.text_widget.index("end-1c")

        # Render remaining preliminary text
        for i, result in enumerate(preliminary_segments):
            separator = " " if i > 0 else ""
            self.text_widget.insert(tk.END, separator + result.text, "preliminary")
