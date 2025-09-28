import tkinter as tk
from tkinter import scrolledtext
from typing import Tuple


def create_stt_window() -> Tuple[tk.Tk, scrolledtext.ScrolledText]:
    """Creates and returns a configured tkinter window with text widget for STT display."""
    root = tk.Tk()
    root.title("Speech-to-Text - Two Stage Display")
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