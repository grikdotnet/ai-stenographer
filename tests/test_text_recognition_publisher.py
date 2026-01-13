"""Tests for RecognitionResultPublisher (Observer pattern implementation).

These tests define the expected behavior of the publisher component
before implementation (TDD RED phase).
"""

import pytest
import logging
import threading
import time
from unittest.mock import Mock, call
from src.RecognitionResultPublisher import RecognitionResultPublisher
from src.types import RecognitionResult


# Test Fixtures

@pytest.fixture
def publisher():
    """Create a publisher instance."""
    return RecognitionResultPublisher(verbose=False)


@pytest.fixture
def mock_subscriber():
    """Create a mock subscriber with protocol methods."""
    subscriber = Mock()
    subscriber.on_partial_update = Mock()
    subscriber.on_finalization = Mock()
    return subscriber


@pytest.fixture
def sample_preliminary_result():
    """Create a sample preliminary recognition result."""
    return RecognitionResult(
        text="hello world",
        status="preliminary",
        chunk_ids=[1, 2],
        start_time=0.0,
        end_time=1.0
    )


@pytest.fixture
def sample_final_result():
    """Create a sample final recognition result."""
    return RecognitionResult(
        text="hello world",
        status="final",
        chunk_ids=[1, 2],
        start_time=0.0,
        end_time=1.0
    )


# Subscription Management Tests

def test_subscribe_adds_subscriber(publisher, mock_subscriber):
    """Verify subscriber registration adds subscriber to internal list."""
    assert publisher.subscriber_count() == 0

    publisher.subscribe(mock_subscriber)

    assert publisher.subscriber_count() == 1


def test_unsubscribe_removes_subscriber(publisher, mock_subscriber):
    """Verify unregister removes subscriber from internal list."""
    publisher.subscribe(mock_subscriber)
    assert publisher.subscriber_count() == 1

    publisher.unsubscribe(mock_subscriber)

    assert publisher.subscriber_count() == 0


def test_duplicate_subscribe_ignored(publisher, mock_subscriber):
    """Verify subscribing the same object twice is idempotent."""
    publisher.subscribe(mock_subscriber)
    publisher.subscribe(mock_subscriber)  # Second subscribe
    publisher.subscribe(mock_subscriber)  # Third subscribe

    # Should only be registered once
    assert publisher.subscriber_count() == 1


def test_unsubscribe_nonexistent_subscriber_safe(publisher, mock_subscriber):
    """Verify unsubscribing non-registered subscriber doesn't raise error."""
    # Should not raise exception
    publisher.unsubscribe(mock_subscriber)
    assert publisher.subscriber_count() == 0


# Notification Tests

def test_publish_partial_update_notifies_all(publisher, sample_preliminary_result):
    """Verify publish_partial_update calls on_partial_update on all subscribers."""
    # Create 3 mock subscribers
    sub1 = Mock()
    sub2 = Mock()
    sub3 = Mock()

    publisher.subscribe(sub1)
    publisher.subscribe(sub2)
    publisher.subscribe(sub3)

    # Publish event
    publisher.publish_partial_update(sample_preliminary_result)

    # All subscribers should receive the call
    sub1.on_partial_update.assert_called_once_with(sample_preliminary_result)
    sub2.on_partial_update.assert_called_once_with(sample_preliminary_result)
    sub3.on_partial_update.assert_called_once_with(sample_preliminary_result)


def test_publish_finalization_notifies_all(publisher, sample_final_result):
    """Verify publish_finalization calls on_finalization on all subscribers."""
    # Create 3 mock subscribers
    sub1 = Mock()
    sub2 = Mock()
    sub3 = Mock()

    publisher.subscribe(sub1)
    publisher.subscribe(sub2)
    publisher.subscribe(sub3)

    # Publish event
    publisher.publish_finalization(sample_final_result)

    # All subscribers should receive the call
    sub1.on_finalization.assert_called_once_with(sample_final_result)
    sub2.on_finalization.assert_called_once_with(sample_final_result)
    sub3.on_finalization.assert_called_once_with(sample_final_result)


def test_publish_after_unsubscribe_skips_unsubscribed(publisher, sample_preliminary_result):
    """Verify unsubscribed subscribers don't receive events."""
    sub1 = Mock()
    sub2 = Mock()

    publisher.subscribe(sub1)
    publisher.subscribe(sub2)

    # Unsubscribe one
    publisher.unsubscribe(sub1)

    # Publish event
    publisher.publish_partial_update(sample_preliminary_result)

    # Only sub2 should receive the call
    sub1.on_partial_update.assert_not_called()
    sub2.on_partial_update.assert_called_once_with(sample_preliminary_result)


# Error Isolation Tests

def test_subscriber_exception_isolation(publisher, sample_preliminary_result, caplog):
    """Verify exception in one subscriber doesn't affect others."""
    # Create subscribers where middle one raises exception
    sub1 = Mock()
    sub2 = Mock()
    sub2.on_partial_update.side_effect = RuntimeError("Subscriber failed!")
    sub3 = Mock()

    publisher.subscribe(sub1)
    publisher.subscribe(sub2)
    publisher.subscribe(sub3)

    # Publish event - should not raise exception
    publisher.publish_partial_update(sample_preliminary_result)

    # sub1 and sub3 should still receive calls
    sub1.on_partial_update.assert_called_once_with(sample_preliminary_result)
    sub3.on_partial_update.assert_called_once_with(sample_preliminary_result)

    # Error should be logged
    assert "failed on_partial_update" in caplog.text


def test_finalization_exception_isolation(publisher, sample_final_result, caplog):
    """Verify exception in finalization doesn't affect other subscribers."""
    sub1 = Mock()
    sub2 = Mock()
    sub2.on_finalization.side_effect = ValueError("Bad data!")
    sub3 = Mock()

    publisher.subscribe(sub1)
    publisher.subscribe(sub2)
    publisher.subscribe(sub3)

    # Publish event
    publisher.publish_finalization(sample_final_result)

    # sub1 and sub3 should still receive calls
    sub1.on_finalization.assert_called_once_with(sample_final_result)
    sub3.on_finalization.assert_called_once_with(sample_final_result)

    # Error should be logged
    assert "failed on_finalization" in caplog.text


# Thread Safety Tests

def test_thread_safe_subscription(publisher):
    """Verify concurrent subscribe/unsubscribe from multiple threads is safe."""
    subscribers = [Mock() for _ in range(10)]
    errors = []

    def subscribe_unsubscribe_loop(sub):
        """Subscribe and unsubscribe repeatedly."""
        try:
            for _ in range(20):
                publisher.subscribe(sub)
                time.sleep(0.001)  # Small delay to increase contention
                publisher.unsubscribe(sub)
        except Exception as e:
            errors.append(e)

    # Create threads
    threads = [threading.Thread(target=subscribe_unsubscribe_loop, args=(sub,))
               for sub in subscribers]

    # Start all threads
    for t in threads:
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    # Should complete without errors
    assert len(errors) == 0
    # Final count should be 0 (all unsubscribed)
    assert publisher.subscriber_count() == 0


def test_thread_safe_notification_during_subscription(publisher, sample_preliminary_result):
    """Verify notifications work correctly when subscribers are added/removed concurrently."""
    call_counts = {'count': 0}
    lock = threading.Lock()

    def counting_subscriber():
        """Subscriber that counts calls."""
        sub = Mock()

        def on_partial_update(result):
            with lock:
                call_counts['count'] += 1

        sub.on_partial_update = on_partial_update
        return sub

    subscribers = [counting_subscriber() for _ in range(5)]

    def publish_loop():
        """Publish events repeatedly."""
        for _ in range(10):
            publisher.publish_partial_update(sample_preliminary_result)
            time.sleep(0.001)

    def subscribe_loop():
        """Subscribe all subscribers."""
        for sub in subscribers:
            publisher.subscribe(sub)
            time.sleep(0.002)

    # Start both operations concurrently
    pub_thread = threading.Thread(target=publish_loop)
    sub_thread = threading.Thread(target=subscribe_loop)

    pub_thread.start()
    sub_thread.start()

    pub_thread.join()
    sub_thread.join()

    # Should complete without crashes
    # Call count should be > 0 (some subscribers received some events)
    assert call_counts['count'] > 0


# Verbose Logging Tests

def test_verbose_logging(caplog, sample_preliminary_result):
    """Verify verbose mode logs subscription events."""
    # Set log level to capture INFO messages
    caplog.set_level(logging.INFO)

    publisher = RecognitionResultPublisher(verbose=True)
    sub = Mock()

    # Subscribe
    publisher.subscribe(sub)
    assert "Subscriber registered" in caplog.text

    # Unsubscribe
    publisher.unsubscribe(sub)
    assert "Subscriber unregistered" in caplog.text


def test_non_verbose_no_logging(caplog, sample_preliminary_result):
    """Verify non-verbose mode doesn't log subscription events."""
    publisher = RecognitionResultPublisher(verbose=False)
    sub = Mock()

    # Clear any existing logs
    caplog.clear()

    # Subscribe
    publisher.subscribe(sub)

    # Should not log (verbose=False)
    assert "Subscriber registered" not in caplog.text
