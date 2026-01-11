"""
ApplicationWindow - Top-level application window.

Owns tkinter root and creates all GUI components directly.
"""
import tkinter as tk
from tkinter import scrolledtext
from typing import Dict
from src.ApplicationState import ApplicationState
from src.gui.HeaderPanel import HeaderPanel
from src.gui.ControlPanel import ControlPanel
from src.controllers.PauseController import PauseController
from src.gui.TextDisplayWidget import TextDisplayWidget
from src.gui.TextFormatter import TextFormatter


class ApplicationWindow:
    """Top-level application window.
    Owns tkinter root and creates all GUI components directly.
    """

    def __init__(self, app_state: ApplicationState, config: Dict, verbose: bool = False):
        """Initialize TK root and all GUI components.

        Args:
            app_state: ApplicationState instance for state management
            config: Application configuration dictionary
            verbose: Enable verbose logging
        """
        self.root = tk.Tk()
        self.root.title("Speech-to-Text Display")
        self.root.geometry("800x600")

        app_state.setTkRoot(self.root)

        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        header_panel = HeaderPanel(main_frame)

        # Create control panel
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        pause_controller = PauseController(app_state)
        control_panel = ControlPanel(
            button_frame,
            app_state,
            pause_controller
        )
        control_panel.pack(side=tk.LEFT, anchor='w')

        status_label = tk.Label(
            button_frame,
            text="Gray text: Preliminary (may change) | Black text: Final",
            font=("Arial", 10),
            fg="gray"
        )
        status_label.pack(side=tk.RIGHT, padx=(20, 50))

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
        text_widget.config(state=tk.NORMAL)

        # Configure text styles for preliminary and final display
        text_widget.tag_configure(
            "preliminary",
            foreground="gray",
            font=("TkDefaultFont", 10, "normal")
        )
        text_widget.tag_configure(
            "final",
            foreground="black",
            font=("TkDefaultFont", 10, "normal")
        )

        # display (view) → formatter (controller)
        self.display = TextDisplayWidget(text_widget, self.root, verbose=verbose)
        self.formatter = TextFormatter(display=self.display, verbose=verbose)

    def get_formatter(self) -> TextFormatter:
        """Returns TextFormatter for pipeline integration.

        Returns:
            TextFormatter instance for TextMatcher to call
        """
        return self.formatter

    def get_display(self) -> TextDisplayWidget:
        """Returns TextDisplayWidget for testing.

        Returns:
            TextDisplayWidget instance
        """
        return self.display

    def get_root(self) -> tk.Tk:
        """Access root window.

        Returns:
            tk.Tk root window instance
        """
        return self.root
