"""
Tests for ApplicationState - State manager with observer pattern.

TDD Phase 1.1: RED - These tests should fail until ApplicationState is implemented.
"""
import unittest
import threading
from unittest.mock import Mock


class TestApplicationState(unittest.TestCase):
    """Test ApplicationState with observer pattern and thread safety."""

    def setUp(self):
        """Set up test fixtures."""
        # Import here to allow test discovery even if module doesn't exist yet
        try:
            from src.ApplicationState import ApplicationState
            self.ApplicationState = ApplicationState
        except ImportError:
            self.skipTest("ApplicationState not implemented yet")

    def test_initial_state_is_starting(self):
        """Test that ApplicationState initializes to 'starting' state."""
        app_state = self.ApplicationState(config={})
        self.assertEqual(app_state.get_state(), 'starting')

    def test_component_observers_receive_old_and_new_state(self):
        """Test that component observers are called with (old_state, new_state)."""
        app_state = self.ApplicationState(config={})

        observer = Mock()
        app_state.register_component_observer(observer)

        app_state.set_state('running')

        observer.assert_called_once_with('starting', 'running')

    def test_thread_safe_state_access(self):
        """Test that state transitions are thread-safe."""
        app_state = self.ApplicationState(config={})
        app_state.set_state('running')

        results = []

        def toggle_state():
            for _ in range(10):
                current = app_state.get_state()
                if current == 'running':
                    app_state.set_state('paused')
                else:
                    app_state.set_state('running')
                results.append(app_state.get_state())

        threads = [threading.Thread(target=toggle_state) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be valid states
        for state in results:
            self.assertIn(state, ['running', 'paused'])

    def test_invalid_transition_rejected(self):
        """Test that invalid state transitions are rejected."""
        app_state = self.ApplicationState(config={})

        # Can't go from starting to paused (must go to running first)
        with self.assertRaises(ValueError):
            app_state.set_state('paused')

    def test_multiple_observers_all_notified(self):
        """Test that all registered observers are notified of state changes."""
        app_state = self.ApplicationState(config={})

        observer1 = Mock()
        observer2 = Mock()
        observer3 = Mock()

        app_state.register_component_observer(observer1)
        app_state.register_component_observer(observer2)
        app_state.register_component_observer(observer3)

        app_state.set_state('running')

        observer1.assert_called_once_with('starting', 'running')
        observer2.assert_called_once_with('starting', 'running')
        observer3.assert_called_once_with('starting', 'running')

    def test_config_ownership(self):
        """Test that ApplicationState holds the application config."""
        config = {'audio': {'sample_rate': 16000}}
        app_state = self.ApplicationState(config=config)

        self.assertEqual(app_state.config, config)


if __name__ == '__main__':
    unittest.main()
