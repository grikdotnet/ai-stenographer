"""
HeaderPanel - Title display component.

Header row with title on left and space for buttons on right.
"""
import tkinter as tk


class HeaderPanel:
    """Header panel with title.

    Creates a horizontal frame with title on the left.
    Exposes frame for adding buttons on the right.
    Single responsibility: Header row layout management.
    """

    def __init__(self, parent: tk.Frame):
        """Initialize HeaderPanel with title label.

        Args:
            parent: Parent frame to contain the header row
        """
        # Create container frame for header row
        self.frame = tk.Frame(parent)
        self.frame.pack(fill=tk.X, pady=(0, 10))

        # Title label on the left
        self.title_label = tk.Label(
            self.frame,
            text="Real-time Speech Recognition",
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(side=tk.LEFT)
