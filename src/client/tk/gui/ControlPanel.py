import tkinter as tk
from tkinter import ttk
from src.ApplicationState import ApplicationState
from src.client.tk.controllers.PauseController import PauseController


class ControlPanel(ttk.Frame):
    """
    Control panel widget with pause/resume button.
    Observes ApplicationState and updates its appearance accordingly.
    Delegates actions to PauseController following proper MVC architecture.

    Attributes:
        app_state: ApplicationState instance
        controller: PauseController instance
        button: Pause/Resume button widget
    """

    def __init__(
        self,
        parent: tk.Widget,
        app_state: ApplicationState,
        controller: PauseController
    ):
        """
        Initialize ControlPanel.

        Args:
            parent: Parent tkinter widget
            app_state: ApplicationState instance
            controller: PauseController instance
        """
        super().__init__(parent)

        self.app_state = app_state
        self.controller = controller

        # Create Pause/Resume button
        self.button = ttk.Button(
            self,
            text='Loading...',
            command=self._on_button_click,
            state='disabled'
        )
        self.button.pack(side=tk.LEFT)

        # Register as GUI observer
        self.app_state.register_gui_observer(self._on_state_change)

        # Update initial state
        current_state = self.app_state.get_state()
        self._update_button_for_state(current_state)

    def _on_state_change(self, old_state: str, new_state: str) -> None:
        """
        Handle state changes from ApplicationState.

        Updates button appearance based on new state.

        Args:
            old_state: Previous state
            new_state: New state
        """
        self._update_button_for_state(new_state)

    def _update_button_for_state(self, state: str) -> None:
        """
        Update button appearance for given state.

        Args:
            state: Application state
        """
        if state == 'running':
            self.button.config(text='Pause', state='normal')
        elif state == 'paused':
            self.button.config(text='Resume', state='normal')
        elif state in ('starting', 'shutdown'):
            self.button.config(state='disabled')

    def _on_button_click(self) -> None:
        """
        Handle button click.

        Disables button immediately to prevent rapid clicking,
        then delegates to controller. Button is re-enabled by
        observer when state change completes.
        """
        # Disable button immediately
        self.button.config(state='disabled')

        # Delegate to controller
        self.controller.toggle()
