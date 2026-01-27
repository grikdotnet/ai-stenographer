import unittest
from unittest.mock import Mock, patch


class TestKeyboardSimulator(unittest.TestCase):
    """Test KeyboardSimulator wrapper around pynput."""

    def test_creates_pynput_controller_on_init(self):
        with patch('src.gui.KeyboardSimulator.Controller') as MockController:
            MockController.return_value = Mock()

            from src.gui.KeyboardSimulator import KeyboardSimulator
            simulator = KeyboardSimulator()

            MockController.assert_called_once()

    def test_type_text_delegates_to_pynput_controller(self):
        """Verify type_text() calls pynput's type() method."""
        mock_keyboard = Mock()
        with patch('src.gui.KeyboardSimulator.Controller', return_value=mock_keyboard):
            from src.gui.KeyboardSimulator import KeyboardSimulator
            simulator = KeyboardSimulator()

            simulator.type_text("hello world")

            mock_keyboard.type.assert_called_once_with("hello world")

    def test_type_text_handles_empty_string(self):
        """Verify empty string is passed through to pynput."""
        mock_keyboard = Mock()
        with patch('src.gui.KeyboardSimulator.Controller', return_value=mock_keyboard):
            from src.gui.KeyboardSimulator import KeyboardSimulator
            simulator = KeyboardSimulator()

            simulator.type_text("")

            mock_keyboard.type.assert_called_once_with("")

    def test_type_text_handles_unicode(self):
        mock_keyboard = Mock()
        with patch('src.gui.KeyboardSimulator.Controller', return_value=mock_keyboard):
            from src.gui.KeyboardSimulator import KeyboardSimulator
            simulator = KeyboardSimulator()

            simulator.type_text("Привет мир 你好世界")

            mock_keyboard.type.assert_called_once_with("Привет мир 你好世界")

if __name__ == '__main__':
    unittest.main()
