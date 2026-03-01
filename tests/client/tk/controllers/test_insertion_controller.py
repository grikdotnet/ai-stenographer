import unittest
from unittest.mock import Mock


class TestInsertionController(unittest.TestCase):

    def setUp(self):
        """Create InsertionController with mocked TextInserter."""
        from src.client.tk.controllers.InsertionController import InsertionController
        self.mock_inserter = Mock()
        self.mock_inserter.is_enabled = Mock(return_value=False)
        self.controller = InsertionController(self.mock_inserter)

    # --- Delegation Tests ---

    def test_toggle_enables_when_disabled(self):
        self.mock_inserter.is_enabled.return_value = False

        self.controller.toggle()

        self.mock_inserter.enable.assert_called_once()
        self.mock_inserter.disable.assert_not_called()

    def test_toggle_disables_when_enabled(self):
        self.mock_inserter.is_enabled.return_value = True

        self.controller.toggle()

        self.mock_inserter.disable.assert_called_once()
        self.mock_inserter.enable.assert_not_called()

    def test_is_enabled_delegates_to_inserter(self):
        self.mock_inserter.is_enabled.return_value = True
        self.assertTrue(self.controller.is_enabled())

        self.mock_inserter.is_enabled.return_value = False
        self.assertFalse(self.controller.is_enabled())

    def test_enable_delegates_to_inserter(self):
        self.controller.enable()
        self.mock_inserter.enable.assert_called_once()


if __name__ == '__main__':
    unittest.main()
