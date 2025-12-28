"""
Tests for PauseController - Minimal controller that updates ApplicationState.
"""
import unittest
from unittest.mock import Mock


class TestPauseController(unittest.TestCase):
    """Test PauseController state management."""

    def setUp(self):
        """Set up test fixtures."""
        # Import here to allow test discovery even if module doesn't exist yet
        try:
            from src.controllers.PauseController import PauseController
            self.PauseController = PauseController
        except ImportError:
            self.skipTest("PauseController not implemented yet")

        # Mock ApplicationState
        self.mock_app_state = Mock()
        self.mock_app_state.get_state = Mock(return_value='running')

    def test_pause_sets_state_to_paused_when_running(self):
        """Test that pause() sets state to 'paused' when currently running."""
        self.mock_app_state.get_state.return_value = 'running'
        controller = self.PauseController(self.mock_app_state)

        controller.pause()

        self.mock_app_state.set_state.assert_called_once_with('paused')

    def test_pause_ignored_when_not_running(self):
        """Test that pause() does nothing when state is not 'running'."""
        self.mock_app_state.get_state.return_value = 'starting'
        controller = self.PauseController(self.mock_app_state)

        controller.pause()

        self.mock_app_state.set_state.assert_not_called()

    def test_resume_sets_state_to_running_when_paused(self):
        """Test that resume() sets state to 'running' when currently paused."""
        self.mock_app_state.get_state.return_value = 'paused'
        controller = self.PauseController(self.mock_app_state)

        controller.resume()

        self.mock_app_state.set_state.assert_called_once_with('running')

    def test_resume_ignored_when_not_paused(self):
        """Test that resume() does nothing when state is not 'paused'."""
        self.mock_app_state.get_state.return_value = 'running'
        controller = self.PauseController(self.mock_app_state)

        controller.resume()

        self.mock_app_state.set_state.assert_not_called()

    def test_toggle_switches_between_running_and_paused(self):
        """Test that toggle() switches between 'running' and 'paused' states."""
        controller = self.PauseController(self.mock_app_state)

        # Test: running -> paused
        self.mock_app_state.get_state.return_value = 'running'
        controller.toggle()
        self.mock_app_state.set_state.assert_called_with('paused')

        self.mock_app_state.set_state.reset_mock()

        # Test: paused -> running
        self.mock_app_state.get_state.return_value = 'paused'
        controller.toggle()
        self.mock_app_state.set_state.assert_called_with('running')

    def test_toggle_ignored_when_invalid_state(self):
        """Test that toggle() does nothing when state is neither running nor paused."""
        self.mock_app_state.get_state.return_value = 'starting'
        controller = self.PauseController(self.mock_app_state)

        controller.toggle()

        self.mock_app_state.set_state.assert_not_called()


if __name__ == '__main__':
    unittest.main()
