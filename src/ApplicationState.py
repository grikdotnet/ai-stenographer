"""
ApplicationState - State manager with observer pattern.

This class manages application state transitions and notifies observers.
It supports two types of observers:
- GUI observers: notified via root.after() when called from background threads
- Component observers: notified directly with (old_state, new_state)

State mutations are protected by threading.Lock.

State Machine:
- starting -> running, shutdown
- running -> paused, shutdown
- paused -> running, shutdown
- shutdown -> (terminal state)
"""
import threading
from typing import Callable, Dict, List, Optional, Set


class ApplicationState:
    """
    Manages application state with observer pattern.

    Attributes:
        config: Application configuration dictionary
        _state: Current state ('starting', 'running', 'paused', 'shutdown')
        _lock: Thread lock for state mutations
        _component_observers: List of component observers (receive old_state, new_state)
        _gui_observers: List of GUI observers (receive old_state, new_state)
        _root: Optional tkinter root for GUI observer scheduling
    """

    # Valid state transitions
    _VALID_TRANSITIONS: Dict[str, Set[str]] = {
        'starting': {'running', 'shutdown'},
        'running': {'paused', 'shutdown'},
        'paused': {'running', 'shutdown'},
        'shutdown': set()
    }

    def __init__(self, config: Dict, root=None):
        """
        Initialize ApplicationState.

        Args:
            config: Application configuration dictionary
            root: Optional tkinter root for GUI observer scheduling
        """
        self.config = config
        self._state = 'starting'
        self._lock = threading.Lock()
        self._component_observers: List[Callable[[str, str], None]] = []
        self._gui_observers: List[Callable[[str, str], None]] = []
        self._root = root

    def setTkRoot(self, root) -> None:
        """Set tkinter root for thread-safe GUI observer callbacks.

        Must be called after GUI window is created to enable scheduling
        GUI observer callbacks on the main thread.

        Args:
            root: tkinter root window
        """
        self._root = root

    def get_state(self) -> str:
        """
        Get current state (thread-safe).

        Returns:
            Current state string
        """
        with self._lock:
            return self._state

    def set_state(self, new_state: str) -> None:
        """
        Set new state and notify observers (thread-safe).

        Args:
            new_state: New state to transition to

        Raises ValueError
        """
        with self._lock:
            old_state = self._state

            # Validate transition
            if new_state not in self._VALID_TRANSITIONS[old_state]:
                raise ValueError(
                    f"Invalid state transition: {old_state} -> {new_state}"
                )

            self._state = new_state

        # Notify observers outside the lock to avoid deadlocks
        self._notify_observers(old_state, new_state)

    def register_component_observer(self, observer: Callable[[str, str], None]) -> None:
        """
        Args:
            observer: Callable that receives (old_state, new_state)
        """
        with self._lock:
            self._component_observers.append(observer)

    def register_gui_observer(self, observer: Callable[[str, str], None]) -> None:
        """
        Args:
            observer: Callable that receives (old_state, new_state)
        """
        with self._lock:
            self._gui_observers.append(observer)

    def _notify_observers(self, old_state: str, new_state: str) -> None:
        """
        Notify all observers of state change.

        Component observers are called directly.
        GUI observers are scheduled via root.after() if called from background thread.

        Args:
            old_state: Previous state
            new_state: New state
        """
        for observer in self._component_observers:
            observer(old_state, new_state)

        is_main_thread = threading.current_thread() is threading.main_thread()

        for observer in self._gui_observers:
            if is_main_thread or self._root is None:
                observer(old_state, new_state)
            else:
                try:
                    self._root.after(0, observer, old_state, new_state)
                except RuntimeError:
                    # Fallback: call directly if main loop not running (e.g., in tests)
                    observer(old_state, new_state)
