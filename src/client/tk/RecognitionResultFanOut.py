"""Concrete fan-out publisher for text recognition events.

Manages a registry of subscribers and broadcasts recognition results to all
of them, decoupling RemoteRecognitionPublisher from downstream consumers
(TextInsertionService, QuickEntryService, etc.).
"""

import threading
import logging
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.protocols import TextRecognitionSubscriber
    from src.types import RecognitionResult


class RecognitionResultFanOut:
    """Broadcasts text recognition events to all registered subscribers.

    Implements the RecognitionResultPublisher protocol, satisfying the interface
    required by RemoteRecognitionPublisher. Manages subscriber lifecycle with
    thread-safe registration and notification.

    Thread Safety:
        - Subscription management uses a lock for thread-safe registration
        - Subscriber list is copied before iteration (lock released during callbacks)
        - No locks held during subscriber callbacks (prevents deadlocks)

    Error Handling:
        - Each subscriber notification is wrapped in try-except
        - Exceptions logged but don't affect other subscribers
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize fan-out publisher.

        Args:
            verbose: Enable verbose logging for subscription events
        """
        self._subscribers: List['TextRecognitionSubscriber'] = []
        self._lock: threading.Lock = threading.Lock()
        self._verbose: bool = verbose

    def subscribe(self, subscriber: 'TextRecognitionSubscriber') -> None:
        """Register a subscriber for text recognition events.

        Thread-safe and idempotent — registering the same subscriber multiple
        times has no additional effect.

        Args:
            subscriber: Object implementing TextRecognitionSubscriber protocol
        """
        with self._lock:
            if subscriber not in self._subscribers:
                self._subscribers.append(subscriber)
                if self._verbose:
                    logging.info(f"Subscriber registered: {subscriber.__class__.__name__}")

    def unsubscribe(self, subscriber: 'TextRecognitionSubscriber') -> None:
        """Unregister a subscriber.

        Thread-safe. Unregistering a non-existent subscriber is a no-op.

        Args:
            subscriber: Previously registered subscriber
        """
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
                if self._verbose:
                    logging.info(f"Subscriber unregistered: {subscriber.__class__.__name__}")

    def subscriber_count(self) -> int:
        """Return current number of subscribers (thread-safe).

        Returns:
            Number of registered subscribers
        """
        with self._lock:
            return len(self._subscribers)

    def publish_partial_update(self, result: 'RecognitionResult') -> None:
        """Publish preliminary recognition result to all subscribers.

        Algorithm:
            1. Copy subscriber list under lock
            2. Call on_partial_update() on each subscriber outside the lock
            3. Log and continue on individual subscriber failures

        Args:
            result: Preliminary RecognitionResult
        """
        with self._lock:
            subscribers = list(self._subscribers)

        for subscriber in subscribers:
            try:
                subscriber.on_partial_update(result)
            except Exception as e:
                logging.error(
                    f"Subscriber {subscriber.__class__.__name__} failed on_partial_update: {e}",
                    exc_info=True
                )

    def publish_finalization(self, result: 'RecognitionResult') -> None:
        """Publish finalized recognition result to all subscribers.

        Algorithm:
            1. Copy subscriber list under lock
            2. Call on_finalization() on each subscriber outside the lock
            3. Log and continue on individual subscriber failures

        Args:
            result: Finalized RecognitionResult
        """
        with self._lock:
            subscribers = list(self._subscribers)

        for subscriber in subscribers:
            try:
                subscriber.on_finalization(result)
            except Exception as e:
                logging.error(
                    f"Subscriber {subscriber.__class__.__name__} failed on_finalization: {e}",
                    exc_info=True
                )
