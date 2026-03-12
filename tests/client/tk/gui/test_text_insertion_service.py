import unittest
from unittest.mock import Mock, patch


class TestTextInsertionService(unittest.TestCase):
    """Test TextInsertionService wiring and integration."""

    def test_creates_text_inserter_with_keyboard_simulator(self):
        mock_publisher = Mock()
        mock_keyboard = Mock()

        with patch('src.client.tk.gui.TextInsertionService.KeyboardSimulator', return_value=mock_keyboard):
            with patch('src.client.tk.gui.TextInsertionService.TextInserter') as MockInserter:
                from src.client.tk.gui.TextInsertionService import TextInsertionService
                service = TextInsertionService(mock_publisher, verbose=True)

                MockInserter.assert_called_once_with(mock_keyboard, True)

    def test_creates_insertion_controller_with_text_inserter(self):
        """Verify InsertionController is created with the text inserter."""
        mock_publisher = Mock()
        mock_inserter = Mock()

        with patch('src.client.tk.gui.TextInsertionService.KeyboardSimulator'):
            with patch('src.client.tk.gui.TextInsertionService.TextInserter', return_value=mock_inserter):
                with patch('src.client.tk.gui.TextInsertionService.InsertionController') as MockController:
                    from src.client.tk.gui.TextInsertionService import TextInsertionService
                    service = TextInsertionService(mock_publisher)

                    MockController.assert_called_once_with(mock_inserter)

    def test_subscribes_text_inserter_to_publisher(self):
        """Verify TextInserter is subscribed to the recognition publisher."""
        mock_publisher = Mock()
        mock_inserter = Mock()

        with patch('src.client.tk.gui.TextInsertionService.KeyboardSimulator'):
            with patch('src.client.tk.gui.TextInsertionService.TextInserter', return_value=mock_inserter):
                with patch('src.client.tk.gui.TextInsertionService.InsertionController'):
                    from src.client.tk.gui.TextInsertionService import TextInsertionService
                    service = TextInsertionService(mock_publisher)

                    mock_publisher.subscribe.assert_called_once_with(mock_inserter)

    def test_controller_property_returns_controller(self):
        """Verify controller property returns the InsertionController."""
        mock_publisher = Mock()
        mock_controller = Mock()

        with patch('src.client.tk.gui.TextInsertionService.KeyboardSimulator'):
            with patch('src.client.tk.gui.TextInsertionService.TextInserter'):
                with patch('src.client.tk.gui.TextInsertionService.InsertionController', return_value=mock_controller):
                    from src.client.tk.gui.TextInsertionService import TextInsertionService
                    service = TextInsertionService(mock_publisher)

                    self.assertIs(service.controller, mock_controller)


class TestTextInsertionServiceIntegration(unittest.TestCase):
    """Integration tests with real components."""

    def test_full_integration_creates_working_service(self):
        """Verify service creates real components that work together."""
        mock_publisher = Mock()

        # Only mock pynput's Controller to avoid actual keyboard control
        with patch('src.client.tk.gui.KeyboardSimulator.Controller'):
            from src.client.tk.gui.TextInsertionService import TextInsertionService
            from src.client.tk.controllers.InsertionController import InsertionController

            service = TextInsertionService(mock_publisher, verbose=False)

            self.assertIsInstance(service.controller, InsertionController)

            # Publisher should have been called to subscribe
            mock_publisher.subscribe.assert_called_once()

            # Controller should delegate to inserter
            self.assertFalse(service.controller.is_enabled())
            service.controller.enable()
            self.assertTrue(service.controller.is_enabled())


if __name__ == '__main__':
    unittest.main()
