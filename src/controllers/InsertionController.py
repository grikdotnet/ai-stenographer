from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.gui.TextInserter import TextInserter


class InsertionController:
    """Controller for text insertion mode."""

    def __init__(self, text_inserter: 'TextInserter') -> None:
        self._text_inserter = text_inserter

    def enable(self) -> None:
        self._text_inserter.enable()

    def disable(self) -> None:
        self._text_inserter.disable()

    def toggle(self) -> None:
        if self._text_inserter.is_enabled():
            self._text_inserter.disable()
        else:
            self._text_inserter.enable()

    def is_enabled(self) -> bool:
        return self._text_inserter.is_enabled()
