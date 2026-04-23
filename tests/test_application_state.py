"""Tests for ApplicationState — server lifecycle state machine."""

import threading
import unittest
from unittest.mock import Mock

from src.ApplicationState import ApplicationState


class TestApplicationState(unittest.TestCase):
    """Test ApplicationState observer pattern and thread safety."""

    def test_initial_state_is_starting(self):
        """ApplicationState initializes to 'starting'."""
        app_state = ApplicationState()
        self.assertEqual(app_state.get_state(), "starting")

    def test_component_observers_receive_old_and_new_state(self):
        """Component observers are called with (old_state, new_state)."""
        app_state = ApplicationState()
        observer = Mock()
        app_state.register_component_observer(observer)

        app_state.set_state("running")

        observer.assert_called_once_with("starting", "running")

    def test_multiple_observers_all_notified(self):
        """All registered observers are notified of state changes."""
        app_state = ApplicationState()
        observer1 = Mock()
        observer2 = Mock()
        observer3 = Mock()
        app_state.register_component_observer(observer1)
        app_state.register_component_observer(observer2)
        app_state.register_component_observer(observer3)

        app_state.set_state("running")

        observer1.assert_called_once_with("starting", "running")
        observer2.assert_called_once_with("starting", "running")
        observer3.assert_called_once_with("starting", "running")

    def test_valid_transition_starting_to_waiting_for_model(self):
        """starting → waiting_for_model is a valid transition."""
        app_state = ApplicationState()
        app_state.set_state("waiting_for_model")
        self.assertEqual(app_state.get_state(), "waiting_for_model")

    def test_valid_transition_waiting_for_model_to_running(self):
        """waiting_for_model → running is a valid transition."""
        app_state = ApplicationState()
        app_state.set_state("waiting_for_model")
        app_state.set_state("running")
        self.assertEqual(app_state.get_state(), "running")

    def test_valid_transition_waiting_for_model_to_shutdown(self):
        """waiting_for_model → shutdown is a valid transition."""
        app_state = ApplicationState()
        app_state.set_state("waiting_for_model")
        app_state.set_state("shutdown")
        self.assertEqual(app_state.get_state(), "shutdown")

    def test_invalid_transition_waiting_for_model_to_starting(self):
        """waiting_for_model → starting raises ValueError."""
        app_state = ApplicationState()
        app_state.set_state("waiting_for_model")
        with self.assertRaises(ValueError):
            app_state.set_state("starting")

    def test_invalid_transition_rejected(self):
        """Invalid state transitions raise ValueError."""
        app_state = ApplicationState()
        with self.assertRaises(ValueError):
            app_state.set_state("invalid_state")
        # Verify valid path: starting -> running -> shutdown
        app_state.set_state("running")
        app_state.set_state("shutdown")
        self.assertEqual(app_state.get_state(), "shutdown")

    def test_shutdown_is_terminal(self):
        """No transitions out of shutdown."""
        app_state = ApplicationState()
        app_state.set_state("running")
        app_state.set_state("shutdown")
        with self.assertRaises(ValueError):
            app_state.set_state("running")

    def test_thread_safe_state_access(self):
        """get_state() is safe to call from multiple threads concurrently."""
        app_state = ApplicationState()
        app_state.set_state("running")
        errors = []

        def read_state():
            for _ in range(100):
                state = app_state.get_state()
                if state not in ("starting", "waiting_for_model", "running", "shutdown"):
                    errors.append(state)

        threads = [threading.Thread(target=read_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
