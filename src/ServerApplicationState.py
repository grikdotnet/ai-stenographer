"""Server application lifecycle state machine.

Three-state machine: starting → running → shutdown.
No GUI observers, no tkinter, no paused state.
"""

import threading
from typing import Callable


class ServerApplicationState:
    """Manages server lifecycle with thread-safe observer notification.

    State machine: starting → running → shutdown (terminal).
    Component observers are called directly in the thread that sets the state.

    Intentionally minimal: no GUI scheduling, no paused state.
    Used by ServerApp, RecognizerService, and SessionManager.
    """

    _VALID_TRANSITIONS: dict[str, set[str]] = {
        "starting": {"running", "shutdown"},
        "running": {"shutdown"},
        "shutdown": set(),
    }

    def __init__(self) -> None:
        self._state = "starting"
        self._lock = threading.Lock()
        self._observers: list[Callable[[str, str], None]] = []

    def get_state(self) -> str:
        """Return the current state (thread-safe).

        Returns:
            One of ``"starting"``, ``"running"``, or ``"shutdown"``.
        """
        with self._lock:
            return self._state

    def set_state(self, new_state: str) -> None:
        """Transition to new_state and notify all component observers.

        Args:
            new_state: Target state.

        Raises:
            ValueError: If the transition is not permitted by the state machine.
        """
        with self._lock:
            old_state = self._state
            if new_state not in self._VALID_TRANSITIONS[old_state]:
                raise ValueError(f"Invalid state transition: {old_state} -> {new_state}")
            self._state = new_state
            observers = list(self._observers)

        for observer in observers:
            observer(old_state, new_state)

    def register_component_observer(self, observer: Callable[[str, str], None]) -> None:
        """Register a callback to receive (old_state, new_state) on every transition.

        Args:
            observer: Callable that accepts two positional string arguments.
        """
        with self._lock:
            self._observers.append(observer)
