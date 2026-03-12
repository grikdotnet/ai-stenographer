"""
Tests for ClientApplicationState - client-side state manager with observer pattern.
"""
import threading
import unittest
from unittest.mock import Mock

from src.client.tk.ClientApplicationState import ClientApplicationState


class TestClientApplicationState(unittest.TestCase):
    """Test ClientApplicationState with observer pattern and thread safety."""

    def test_initial_state_is_starting(self):
        """Test that ClientApplicationState initializes to 'starting' state."""
        app_state = ClientApplicationState(config={})
        self.assertEqual(app_state.get_state(), 'starting')

    def test_config_ownership(self):
        """Test that ClientApplicationState holds the application config."""
        config = {'audio': {'sample_rate': 16000}}
        app_state = ClientApplicationState(config=config)
        self.assertEqual(app_state.config, config)

    def test_valid_transition_starting_to_running(self):
        """Test transition from starting to running."""
        app_state = ClientApplicationState(config={})
        app_state.set_state('running')
        self.assertEqual(app_state.get_state(), 'running')

    def test_valid_transition_running_to_paused(self):
        """Test transition from running to paused."""
        app_state = ClientApplicationState(config={})
        app_state.set_state('running')
        app_state.set_state('paused')
        self.assertEqual(app_state.get_state(), 'paused')

    def test_valid_transition_paused_to_running(self):
        """Test transition from paused back to running."""
        app_state = ClientApplicationState(config={})
        app_state.set_state('running')
        app_state.set_state('paused')
        app_state.set_state('running')
        self.assertEqual(app_state.get_state(), 'running')

    def test_invalid_transition_rejected(self):
        """Test that invalid state transitions raise ValueError."""
        app_state = ClientApplicationState(config={})
        with self.assertRaises(ValueError):
            app_state.set_state('paused')

    def test_shutdown_is_terminal(self):
        """Test that shutdown state cannot be exited."""
        app_state = ClientApplicationState(config={})
        app_state.set_state('shutdown')
        with self.assertRaises(ValueError):
            app_state.set_state('running')

    def test_component_observer_receives_old_and_new_state(self):
        """Test that component observers are called with (old_state, new_state)."""
        app_state = ClientApplicationState(config={})
        observer = Mock()
        app_state.register_component_observer(observer)
        app_state.set_state('running')
        observer.assert_called_once_with('starting', 'running')

    def test_multiple_component_observers_all_notified(self):
        """Test that all registered component observers are notified."""
        app_state = ClientApplicationState(config={})
        observers = [Mock(), Mock(), Mock()]
        for obs in observers:
            app_state.register_component_observer(obs)
        app_state.set_state('running')
        for obs in observers:
            obs.assert_called_once_with('starting', 'running')

    def test_gui_observer_called_on_main_thread(self):
        """Test that GUI observers are called directly when on main thread."""
        app_state = ClientApplicationState(config={})
        observer = Mock()
        app_state.register_gui_observer(observer)
        app_state.set_state('running')
        observer.assert_called_once_with('starting', 'running')

    def test_gui_observer_scheduled_via_root_after_from_background_thread(self):
        """Test that GUI observers are scheduled via root.after() from background threads."""
        root = Mock()
        app_state = ClientApplicationState(config={}, root=root)
        observer = Mock()
        app_state.register_gui_observer(observer)

        scheduled = []

        def background():
            app_state.set_state('running')

        t = threading.Thread(target=background)
        t.start()
        t.join()

        root.after.assert_called_once_with(0, observer, 'starting', 'running')
        observer.assert_not_called()

    def test_set_tk_root(self):
        """Test that setTkRoot assigns root for GUI observer scheduling."""
        app_state = ClientApplicationState(config={})
        root = Mock()
        app_state.setTkRoot(root)
        observer = Mock()
        app_state.register_gui_observer(observer)

        def background():
            app_state.set_state('running')

        t = threading.Thread(target=background)
        t.start()
        t.join()

        root.after.assert_called_once_with(0, observer, 'starting', 'running')

    def test_thread_safe_state_access(self):
        """Test that state transitions are thread-safe under concurrent access."""
        app_state = ClientApplicationState(config={})
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

        for state in results:
            self.assertIn(state, ['running', 'paused'])


if __name__ == '__main__':
    unittest.main()
