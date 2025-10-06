import tkinter as tk
from tkinter import scrolledtext
from typing import Tuple


class GuiWindow:
    """Manages two-stage text display for real-time speech recognition.

    Provides update_partial() and finalize_text() methods for displaying
    preliminary (gray/italic) and final (black/normal) text in a tkinter widget.
    """

    def __init__(self, text_widget: scrolledtext.ScrolledText, root: tk.Tk = None) -> None:
        self.text_widget: scrolledtext.ScrolledText = text_widget
        self.root: tk.Tk = root
        self.preliminary_start_pos: str = "1.0"
        self.last_finalized_text: str = ""

        self._setup_text_styles()

    def _is_main_thread(self) -> bool:
        """Check if we're running on the main thread."""
        import threading
        return threading.current_thread() is threading.main_thread()

    def _setup_text_styles(self) -> None:
        """Configure visual styling for preliminary and final text display."""
        self.text_widget.tag_configure("preliminary", foreground="gray", font=("TkDefaultFont", 10, "italic"))
        self.text_widget.tag_configure("final", foreground="black", font=("TkDefaultFont", 10, "normal"))

    def update_partial(self, text: str) -> None:
        """Replace current preliminary text with new recognition results.

        Uses position tracking to replace only the preliminary text region
        while preserving all finalized text. The preliminary text appears
        in gray/italic styling to indicate it may change.

        Args:
            text: Preliminary text to display
        """
        if self.root and not self._is_main_thread():
            # Schedule GUI update on main thread when called from background thread
            self.root.after(0, self._update_partial_safe, text)
        else:
            # Direct call when on main thread or in tests
            self._update_partial_safe(text)

    def _update_partial_safe(self, text: str) -> None:
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
        """Convert preliminary text to final permanent text.

        Removes any existing preliminary text and replaces it with finalized
        text in black/normal styling. Updates position tracking to prepare for
        the next preliminary text sequence.

        Args:
            final_text: Final text to display permanently
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


def create_stt_window() -> Tuple[tk.Tk, scrolledtext.ScrolledText]:
    """Creates and returns a configured tkinter window with text widget for STT display."""
    root = tk.Tk()
    root.title("Speech-to-Text Display")
    root.geometry("800x600")

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

    # Create text widget with scrollbar
    text_widget = scrolledtext.ScrolledText(
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