"""Publisher for text recognition events with thread-safe subscriber management.

This module implements the Observer pattern's publisher component, enabling
multiple subscribers to receive speech recognition results independently.
The publisher isolates TextMatcher from its consumers.
"""

import threading
import logging
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.protocols import TextRecognitionSubscriber
    from src.types import RecognitionResult


class RecognitionResultPublisher:
    """Manages subscribers and publishes text recognition events.

    This class decouples TextMatcher from its consumers (GUI, API, etc.)
    by managing a registry of subscribers and notifying them of recognition
    events. Each subscriber receives events independently.

    Thread Safety:
        - Subscription management uses a lock for thread-safe registration
        - Subscriber list is copied before iteration (lock released during callbacks)
        - No locks held during subscriber callbacks (prevents deadlocks)

    Error Handling:
        - Each subscriber notification is wrapped in try-except
        - Exceptions logged but don't affect other subscribers
        - Publishing continues even if individual subscribers fail

    Example:
        >>> publisher = RecognitionResultPublisher(verbose=True)
        >>> publisher.subscribe(formatter)
        >>> publisher.subscribe(api_subscriber)
        >>> publisher.publish_partial_update(result)  # Both receive event
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize publisher.

        Args:
            verbose: Enable verbose logging for subscription events
        """
        self._subscribers: List['TextRecognitionSubscriber'] = []
        self._lock: threading.Lock = threading.Lock()
        self._verbose: bool = verbose

    def subscribe(self, subscriber: 'TextRecognitionSubscriber') -> None:
        """Register a subscriber for text recognition events.

        Thread-safe and idempotent - registering the same subscriber multiple
        times has no additional effect.

        Args:
            subscriber: Object implementing TextRecognitionSubscriber protocol
                       (must have on_partial_update and on_finalization methods)
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

    def publish_partial_update(self, result: 'RecognitionResult') -> None:
        """Publish preliminary recognition result to all subscribers.

        Calls on_partial_update() on each registered subscriber. If a subscriber
        raises an exception, the error is logged and other subscribers continue
        to receive the event.

        Thread Safety:
            The subscriber list is copied under lock, then callbacks are made
            outside the lock to prevent deadlocks if subscribers need to
            register/unregister during callbacks.

        Args:
            result: Preliminary RecognitionResult
        """
        # Copy subscriber list under lock, then notify outside lock
        # This prevents deadlocks if subscribers modify the registry
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

        Calls on_finalization() on each registered subscriber. If a subscriber
        raises an exception, the error is logged and other subscribers continue
        to receive the event.

        Thread Safety:
            Same strategy as publish_partial_update - copy list under lock,
            then notify outside lock.

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

    def subscriber_count(self) -> int:
        """Get current number of subscribers (thread-safe).

        Useful for debugging and testing.

        Returns:
            Number of registered subscribers
        """
        with self._lock:
            return len(self._subscribers)
