import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.controllers.InsertionController import InsertionController


class InsertionModePanel(ttk.Frame):
    """Toggle button widget for text insertion mode.

    Attributes:
        controller: InsertionController instance
        button: Toggle button widget
    """

    def __init__(self, parent: tk.Widget, controller: 'InsertionController') -> None:
        """Initialize InsertionModePanel.

        Args:
            parent: Parent tkinter widget
            controller: InsertionController instance for toggling
        """
        super().__init__(parent)

        self.controller = controller

        # Create toggle button
        self.button = ttk.Button(
            self,
            text='Insert: OFF',
            command=self._on_button_click
        )
        self.button.pack()

        # Set initial state
        self._update_button_state()

    def _on_button_click(self) -> None:
        """Handle button click - toggle insertion mode."""
        self.controller.toggle()
        self._update_button_state()

    def _update_button_state(self) -> None:
        """Update button text based on current insertion state."""
        if self.controller.is_enabled():
            self.button.config(text='Insert: ON')
        else:
            self.button.config(text='Insert: OFF')
