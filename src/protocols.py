"""Protocol definitions for speech-to-text system components.

This module defines structural interfaces using Python's Protocol for duck typing.
"""

from typing import Protocol
from src.types import RecognitionResult


class TextRecognitionSubscriber(Protocol):
    """Subscriber interface for text recognition events.

    Components implementing this protocol can receive notifications about
    speech recognition results. The protocol uses structural subtyping,
    so classes don't need explicit inheritance - just matching method signatures.

    Thread Safety:
        Implementations must handle calls from background threads.
        TextMatcher invokes these methods from its daemon thread.
    """

    def on_partial_update(self, result: RecognitionResult) -> None:
        """Handle preliminary recognition result.

        Called when speech recognition produces an intermediate result that
        may be refined or overwritten by subsequent results.

        Args:
            result: RecognitionResult with status='preliminary'
        """
        ...

    def on_finalization(self, result: RecognitionResult) -> None:
        """Handle finalized recognition result after overlap resolution.

        Called when TextMatcher has resolved overlapping chunks and produced
        a stable recognition result. This text should be treated as final.

        Args:
            result: RecognitionResult with status='final' or 'flush'
        """
        ...
