"""
HeaderPanel - Title display component.

Simple panel displaying application title.
"""
import tkinter as tk


class HeaderPanel:
    """Header panel with title.

    Displays the application title with consistent styling.
    Single responsibility: Title display only.
    """

    def __init__(self, parent: tk.Frame):
        """Initialize HeaderPanel with title label.

        Args:
            parent: Parent frame to contain the title label
        """
        self.title_label = tk.Label(
            parent,
            text="Real-time Speech Recognition",
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(pady=(0, 10))
