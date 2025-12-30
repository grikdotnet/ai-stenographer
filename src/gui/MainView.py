"""
MainView - Main application view orchestrating all panels.

Responsible for layout of header, controls, and text display.
"""
import tkinter as tk
from typing import Dict
from src.ApplicationState import ApplicationState
from src.gui.HeaderPanel import HeaderPanel
from src.gui.TextDisplayPanel import TextDisplayPanel
from src.gui.ControlPanel import ControlPanel
from src.controllers.PauseController import PauseController


class MainView:

    def __init__(self, parent: tk.Tk, app_state: ApplicationState, config: Dict):
        """Initialize panels.

        Args:
            parent: Parent tk.Tk window
            app_state: ApplicationState instance for state management
            config: Application configuration dictionary
        """
        self.parent = parent
        self.app_state = app_state
        self.config = config

        # Create main frame
        self.main_frame = tk.Frame(parent, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create header panel
        self.header_panel = HeaderPanel(self.main_frame)

        # Create control panel container
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=(0, 10))

        # Create control panel with pause/resume
        pause_controller = PauseController(app_state)
        self.control_panel = ControlPanel(
            self.button_frame,
            app_state,
            pause_controller
        )
        self.control_panel.pack(side=tk.LEFT, anchor='w')

        # Create status label
        self.status_label = tk.Label(
            self.button_frame,
            text="Gray text: Preliminary (may change) | Black text: Final",
            font=("Arial", 10),
            fg="gray"
        )
        self.status_label.pack(side=tk.RIGHT, padx=(20, 50))

        # Create text display panel
        self.text_display_panel = TextDisplayPanel(
            parent=self.main_frame,
            root=parent,
            config=config
        )
