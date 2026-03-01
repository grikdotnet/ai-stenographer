from typing import TYPE_CHECKING

from src.client.tk.gui.KeyboardSimulator import KeyboardSimulator
from src.client.tk.gui.TextInserter import TextInserter
from src.client.tk.controllers.InsertionController import InsertionController

if TYPE_CHECKING:
    from src.RecognitionResultPublisher import RecognitionResultPublisher


class TextInsertionService:
    """ Facade for text insertion subsystem.
    Creates components and subscribes to the publisher.
    """

    def __init__(self, publisher: 'RecognitionResultPublisher', verbose: bool = False) -> None:
        """
        Args:
            publisher: RecognitionResultPublisher to subscribe TextInserter to
            verbose: Enable verbose logging for TextInserter
        """
        self._keyboard_simulator = KeyboardSimulator()
        self._text_inserter = TextInserter(self._keyboard_simulator, verbose)
        self._controller = InsertionController(self._text_inserter)

        # Subscribe inserter to receive recognition results
        publisher.subscribe(self._text_inserter)

    @property
    def controller(self) -> InsertionController:
        """
        Returns:
            InsertionController instance for GUI binding
        """
        return self._controller
