"""
TextDisplayPanel - Text display area component.

Panel containing scrolled text widget and GuiWindow for text rendering.
"""
import tkinter as tk
from typing import Dict
from src.GuiFactory import GuiFactory
from src.GuiWindow import GuiWindow


class TextDisplayPanel:
    """Text display panel containing scrolled text widget.

    Owns the text widget and GuiWindow for text rendering logic.
    Single responsibility: Text display area management.
    """

    def __init__(self, parent: tk.Frame, root: tk.Tk, config: Dict):
        """Initialize TextDisplayPanel with text widget and GuiWindow.

        Args:
            parent: Parent frame to contain the text widget
            root: Root tk.Tk window for thread-safe GUI updates
            config: Application configuration dictionary
        """
        # Create scrolled text widget using factory
        self.text_widget = GuiFactory.create_scrolled_text(
            parent,
            wrap=tk.WORD,
            width=80,
            height=30,
            font=("Arial", 12),
            bg="white",
            fg="black"
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        self.text_widget.config(state=tk.NORMAL)

        # Create GuiWindow for text rendering logic
        self.gui_window = GuiWindow(self.text_widget, root)
