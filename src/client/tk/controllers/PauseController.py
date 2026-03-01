"""
PauseController - Minimal controller for pause/resume functionality.

This controller ONLY updates ApplicationState. It does not manipulate
components directly - proper MVC separation with observer pattern.
"""
from src.ApplicationState import ApplicationState


class PauseController:
    """
    Controller for pause/resume operations.

    This controller follows proper MVC architecture:
    - ONLY updates ApplicationState
    - Does NOT manipulate components directly
    - Components observe ApplicationState and react independently

    Attributes:
        app_state: ApplicationState instance to manage
    """

    def __init__(self, app_state: ApplicationState):
        """
        Initialize PauseController.

        Args:
            app_state: ApplicationState instance
        """
        self.app_state = app_state

    def pause(self) -> None:
        """
        Pause the application if currently running.

        Guards against invalid transitions - only transitions from 'running' to 'paused'.
        """
        if self.app_state.get_state() == 'running':
            self.app_state.set_state('paused')

    def resume(self) -> None:
        """
        Resume the application if currently paused.

        Guards against invalid transitions - only transitions from 'paused' to 'running'.
        """
        if self.app_state.get_state() == 'paused':
            self.app_state.set_state('running')

    def toggle(self) -> None:
        """
        Toggle between running and paused states.

        Convenience method for UI toggle buttons.
        Does nothing if state is neither running nor paused.
        """
        current_state = self.app_state.get_state()
        if current_state == 'running':
            self.app_state.set_state('paused')
        elif current_state == 'paused':
            self.app_state.set_state('running')
