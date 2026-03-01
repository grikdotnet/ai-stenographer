"""Protocol for publishing text recognition events.

Defines the minimal interface that any recognition result publisher must satisfy,
enabling server-side components (IncrementalTextMatcher, WsResultSender) and
client-side components (RecognitionResultFanOut) to be typed uniformly.
"""

from typing import TYPE_CHECKING
from typing import Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.types import RecognitionResult


@runtime_checkable
class RecognitionResultPublisher(Protocol):
    """Protocol for publishing text recognition results.

    Implemented by WsResultSender (server) and RecognitionResultFanOut (client).
    Any object with publish_partial_update and publish_finalization satisfies it.
    """

    def publish_partial_update(self, result: 'RecognitionResult') -> None:
        """Publish a preliminary recognition result.

        Args:
            result: Preliminary RecognitionResult
        """
        ...

    def publish_finalization(self, result: 'RecognitionResult') -> None:
        """Publish a finalized recognition result.

        Args:
            result: Finalized RecognitionResult
        """
        ...
