"""
ApplicationWindow - Top-level application window.

Owns tkinter root and orchestrates MainView creation.
"""
import tkinter as tk
from typing import Dict
from src.ApplicationState import ApplicationState
from src.gui.GuiFactory import GuiFactory
from src.gui.MainView import MainView
from src.GuiWindow import GuiWindow


class ApplicationWindow:
    """Top-level application window.
    Owns tkinter root and orchestrates MainView creation.
    """

    def __init__(self, app_state: ApplicationState, config: Dict):
        """Initialize ApplicationWindow with root and main view.

        Args:
            app_state: ApplicationState instance for state management
            config: Application configuration dictionary
        """
        self.app_state = app_state
        self.config = config

        # Create root window
        self.root = GuiFactory.create_window(
            "Speech-to-Text Display",
            "800x600"
        )

        # Set root in ApplicationState for thread-safe callbacks
        self.app_state.setTkRoot(self.root)

        # Create main view (layout + all panels)
        self.main_view = MainView(
            parent=self.root,
            app_state=self.app_state,
            config=self.config
        )

    def get_gui_window(self) -> GuiWindow:
        """Returns GuiWindow for pipeline integration."""
        return self.main_view.text_display_panel.gui_window

    def get_root(self) -> tk.Tk:
        """Access root window.

        Returns:
            tk.Tk root window instance
        """
        return self.root
