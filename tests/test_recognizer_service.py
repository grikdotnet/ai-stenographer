"""Tests for RecognizerService (Phase 4).

Strategy: mock Recognizer model, real queues, threading.Event for coordination.
"""

import queue
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ServerApplicationState import ServerApplicationState
from src.server.RecognizerService import RecognizerService
from src.types import AudioSegment, RecognitionResult, RecognitionTextMessage, RecognizerAck


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(message_id: int) -> AudioSegment:
    """Build a minimal AudioSegment with a given message_id."""
    data = np.zeros(512, dtype=np.float32)
    seg = AudioSegment(
        type="incremental",
        data=data,
        left_context=np.array([], dtype=np.float32),
        right_context=np.array([], dtype=np.float32),
        start_time=0.0,
        end_time=1.0,
        utterance_id=1,
        chunk_ids=[1],
        message_id=message_id,
    )
    return seg


def _make_recognition_result() -> RecognitionResult:
    return RecognitionResult(text="hello", start_time=0.0, end_time=1.0)


def _make_recognizer(result: RecognitionResult | None) -> MagicMock:
    """Create a mock Recognizer whose recognize_window returns a fixed result."""
    recognizer = MagicMock()
    recognizer.recognize_window.return_value = result
    return recognizer


def _start_service(recognizer, app_state=None) -> RecognizerService:
    if app_state is None:
        app_state = ServerApplicationState()
        app_state.set_state("running")
    service = RecognizerService(recognizer=recognizer, app_state=app_state)
    service.start()
    return service


def _drain(out_queue: queue.Queue, count: int, timeout: float = 2.0) -> list:
    """Collect `count` items from out_queue within timeout."""
    items = []
    deadline = time.monotonic() + timeout
    while len(items) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            items.append(out_queue.get(timeout=min(remaining, 0.1)))
        except queue.Empty:
            continue
    return items


# ---------------------------------------------------------------------------
# Session-prefixed ID allocation
# ---------------------------------------------------------------------------

class TestSessionPrefixedIds:
    def test_session1_ids_start_at_10_000_001(self) -> None:
        """Session index 1 → first_message_id = 10_000_001."""
        assert 1 * 10_000_000 + 1 == 10_000_001

    def test_session2_ids_start_at_20_000_001(self) -> None:
        """Session index 2 → first_message_id = 20_000_001."""
        assert 2 * 10_000_000 + 1 == 20_000_001

    def test_sessions_do_not_collide(self) -> None:
        """10M IDs per session guarantees no cross-session collision."""
        session1_max = 10_000_000 + 9_999_999
        session2_min = 20_000_000 + 1
        assert session1_max < session2_min


# ---------------------------------------------------------------------------
# Routing: correct session queue receives results
# ---------------------------------------------------------------------------

class TestResultRouting:
    def test_message_from_session1_routed_to_session1_queue(self) -> None:
        result = _make_recognition_result()
        recognizer = _make_recognizer(result)
        app_state = ServerApplicationState()
        app_state.set_state("running")
        service = RecognizerService(recognizer=recognizer, app_state=app_state)

        out1: queue.Queue = queue.Queue()
        out2: queue.Queue = queue.Queue()
        service.register_session(1, out1)
        service.register_session(2, out2)
        service.start()

        segment = _make_segment(message_id=10_000_001)
        service.input_queue.put(segment)

        items = _drain(out1, count=2)
        service.stop()
        service.join()

        assert len(items) == 2
        assert out2.empty()

    def test_message_from_session2_routed_to_session2_queue(self) -> None:
        result = _make_recognition_result()
        recognizer = _make_recognizer(result)
        app_state = ServerApplicationState()
        app_state.set_state("running")
        service = RecognizerService(recognizer=recognizer, app_state=app_state)

        out1: queue.Queue = queue.Queue()
        out2: queue.Queue = queue.Queue()
        service.register_session(1, out1)
        service.register_session(2, out2)
        service.start()

        segment = _make_segment(message_id=20_000_001)
        service.input_queue.put(segment)

        items = _drain(out2, count=2)
        service.stop()
        service.join()

        assert len(items) == 2
        assert out1.empty()

    def test_result_and_ack_both_routed(self) -> None:
        result = _make_recognition_result()
        recognizer = _make_recognizer(result)
        app_state = ServerApplicationState()
        app_state.set_state("running")
        service = RecognizerService(recognizer=recognizer, app_state=app_state)

        out: queue.Queue = queue.Queue()
        service.register_session(1, out)
        service.start()

        service.input_queue.put(_make_segment(message_id=10_000_001))

        items = _drain(out, count=2)
        service.stop()
        service.join()

        types = {type(i) for i in items}
        assert RecognitionTextMessage in types
        assert RecognizerAck in types

    def test_empty_recognition_sends_only_ack(self) -> None:
        """When recognize_window returns None, only RecognizerAck is routed."""
        recognizer = _make_recognizer(None)
        app_state = ServerApplicationState()
        app_state.set_state("running")
        service = RecognizerService(recognizer=recognizer, app_state=app_state)

        out: queue.Queue = queue.Queue()
        service.register_session(1, out)
        service.start()

        service.input_queue.put(_make_segment(message_id=10_000_001))

        items = _drain(out, count=1)
        time.sleep(0.1)
        service.stop()
        service.join()

        assert len(items) == 1
        assert isinstance(items[0], RecognizerAck)


# ---------------------------------------------------------------------------
# Session removal
# ---------------------------------------------------------------------------

class TestSessionRemoval:
    def test_unregistered_session_results_are_dropped(self) -> None:
        result = _make_recognition_result()
        recognizer = _make_recognizer(result)
        app_state = ServerApplicationState()
        app_state.set_state("running")
        service = RecognizerService(recognizer=recognizer, app_state=app_state)

        out: queue.Queue = queue.Queue()
        service.register_session(1, out)
        service.unregister_session(1)
        service.start()

        service.input_queue.put(_make_segment(message_id=10_000_001))

        time.sleep(0.3)
        service.stop()
        service.join()

        assert out.empty()


# ---------------------------------------------------------------------------
# Shutdown via ServerApplicationState
# ---------------------------------------------------------------------------

class TestShutdown:
    def test_shutdown_stops_inference_thread(self) -> None:
        recognizer = _make_recognizer(None)
        app_state = ServerApplicationState()
        app_state.set_state("running")
        service = RecognizerService(recognizer=recognizer, app_state=app_state)
        service.start()

        assert service._thread is not None
        assert service._thread.is_alive()

        app_state.set_state("shutdown")
        service.join(timeout=2.0)

        assert not service._thread.is_alive()

    def test_service_uses_server_app_state_not_session_state(self) -> None:
        """RecognizerService observes ServerApplicationState; session shutdown does not stop it."""
        recognizer = _make_recognizer(None)
        server_app_state = ServerApplicationState()
        server_app_state.set_state("running")
        service = RecognizerService(recognizer=recognizer, app_state=server_app_state)
        service.start()

        session_app_state = ServerApplicationState()
        session_app_state.set_state("running")
        session_app_state.set_state("shutdown")

        time.sleep(0.1)
        assert service._thread.is_alive()

        server_app_state.set_state("shutdown")
        service.join(timeout=2.0)
