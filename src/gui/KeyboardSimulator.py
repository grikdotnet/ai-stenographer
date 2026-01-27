from pynput.keyboard import Controller


class KeyboardSimulator:
    """ An abstraction over pynput's keyboard Controller for dependency injection.
    All keyboard operations are delegated to pynput Controller.

    Example:
        >>> simulator = KeyboardSimulator()
        >>> simulator.type_text("Hello World")  # Types into active window
    """

    def __init__(self) -> None:
        self._keyboard: Controller = Controller()

    def type_text(self, text: str) -> None:
        """Type text character by character into the active window.

        The text is typed at the current cursor position in whatever application has focus.

        Args:
            text: The string to type.
        """
        self._keyboard.type(text)
