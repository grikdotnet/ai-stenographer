"""Shared ASR inference service for all active sessions.

A single inference thread calls Recognizer.recognize_window() sequentially.
Results are routed back to per-session output queues using session-prefixed
message IDs (session_index * 10_000_000).
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.types import AudioSegment, RecognitionTextMessage, RecognizerAck

if TYPE_CHECKING:
    from src.asr.Recognizer import Recognizer
    from src.ApplicationState import ApplicationState

logger = logging.getLogger(__name__)

_SESSION_PREFIX = 10_000_000


@dataclass
class _SessionDrainState:
    """Per-session drain bookkeeping for graceful close_session."""

    output_queue: queue.Queue
    pending_segments: int = 0
    drain_event: threading.Event | None = None


class _RecognizerInputQueue:
    """Queue-like wrapper that records pending per-session segments on submit."""

    def __init__(self, owner: "RecognizerService") -> None:
        self._owner = owner

    def put(self, item: AudioSegment, block: bool = True, timeout: float | None = None) -> None:
        self._owner._enqueue_segment(item, block=block, timeout=timeout)

    def put_nowait(self, item: AudioSegment) -> None:
        self._owner._enqueue_segment(item, block=False, timeout=None)


class RecognizerService:
    """Owns the shared Recognizer inference thread and session output routing.

    All active sessions submit AudioSegment items to the shared input_queue.
    Session affinity is encoded in message_id: ``session_index * 10_000_000 + local_id``.
    After each inference call the result (and always an ACK) are routed to
    the correct per-session output queue by the ``message_id // 10_000_000`` key.

    Observes ApplicationState for shutdown; stops inference thread on transition.
    A Recognizer may be attached after construction via attach_recognizer(); segments
    received before attachment are responded to with an error ACK.

    Args:
        app_state: Server lifecycle state; RecognizerService stops on shutdown.
    """

    def __init__(
        self,
        app_state: "ApplicationState",
    ) -> None:
        self._recognizer: "Recognizer | None" = None
        self._app_state = app_state
        self._input_queue: queue.Queue[AudioSegment] = queue.Queue()
        self.input_queue = _RecognizerInputQueue(self)
        self._session_output_queues: dict[int, queue.Queue] = {}
        self._session_states: dict[int, _SessionDrainState] = {}
        self._routing_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

        app_state.register_component_observer(self._on_state_change)

    def attach_recognizer(self, recognizer: "Recognizer") -> None:
        """Attach the recognizer after construction.

        Args:
            recognizer: Pure synchronous recognizer callable (recognize_window method used).
        """
        self._recognizer = recognizer

    def register_session(self, session_index: int, output_queue: queue.Queue) -> None:
        """Register a per-session output queue keyed by session_index.

        Args:
            session_index: 1-based session identifier.
            output_queue: Destination queue for RecognitionTextMessage / RecognizerAck.
        """
        with self._routing_lock:
            self._session_output_queues[session_index] = output_queue
            self._session_states[session_index] = _SessionDrainState(output_queue=output_queue)

    def unregister_session(self, session_index: int) -> None:
        """Remove a session's output queue; subsequent results for it are dropped.

        Args:
            session_index: Session to remove.
        """
        with self._routing_lock:
            self._session_output_queues.pop(session_index, None)
            self._session_states.pop(session_index, None)

    def flush_session(self, session_index: int, timeout: float) -> None:
        """Block until the session has no pending recognizer work.

        Args:
            session_index: Session to wait for.
            timeout: Maximum number of seconds to wait.
        """
        with self._routing_lock:
            state = self._session_states.get(session_index)
            if state is None:
                return
            if state.pending_segments == 0:
                return
            drain_event = state.drain_event
            if drain_event is None:
                drain_event = threading.Event()
                state.drain_event = drain_event

        if not drain_event.wait(timeout=timeout):
            logger.warning(
                "RecognizerService: flush_session timed out for session %s after %.2fs",
                session_index,
                timeout,
            )

    def start(self) -> None:
        """Start the inference thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="RecognizerService")
        self._thread.start()

    def stop(self) -> None:
        """Signal the inference thread to stop."""
        self._running = False

    def join(self, timeout: float = 5.0) -> None:
        """Wait for the inference thread to exit.

        Args:
            timeout: Seconds to wait before giving up.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _on_state_change(self, old_state: str, new_state: str) -> None:
        """Stop on server shutdown.

        Args:
            old_state: Previous server state.
            new_state: New server state.
        """
        if new_state == "shutdown":
            self.stop()

    def _run(self) -> None:
        """Inference thread loop.

        Algorithm:
            1. Blocking get from input_queue (0.1 s timeout to check _running flag).
            2. Call recognizer.recognize_window(segment).
            3. If result is not None, route RecognitionTextMessage to session queue.
            4. Always route RecognizerAck to session queue.
            5. If session is no longer registered, drop and log.
        """
        while self._running:
            try:
                segment: AudioSegment = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            message_id = segment.message_id
            session_index = message_id // _SESSION_PREFIX if message_id is not None else None

            if self._recognizer is None:
                self._route(session_index, RecognizerAck(message_id=message_id, ok=False, error="recognizer not ready"))
                continue

            try:
                result = self._recognizer.recognize_window(segment)
            except Exception:
                logger.exception(
                    "RecognizerService: recognize_window failed for message_id=%s", message_id
                )
                self._route(session_index, RecognizerAck(message_id=message_id, ok=False, error="inference error"))
                continue

            if result is not None:
                self._route(session_index, RecognitionTextMessage(result=result, message_id=message_id))

            self._route(session_index, RecognizerAck(message_id=message_id, ok=True))

    def _route(self, session_index: int | None, item) -> None:
        """Put item onto the registered session output queue.

        Drops and logs if session is not registered or index is None.

        Args:
            session_index: Routing key derived from message_id // 10_000_000.
            item: RecognitionTextMessage or RecognizerAck to deliver.
        """
        if session_index is None:
            logger.error("RecognizerService: cannot route item with None session_index")
            return

        with self._routing_lock:
            state = self._session_states.get(session_index)

        if state is None:
            logger.debug(
                "RecognizerService: session %s not registered, dropping result", session_index
            )
            return

        out_queue = state.output_queue
        try:
            out_queue.put_nowait(item)
        except queue.Full:
            logger.warning(
                "RecognizerService: output queue full for session %s, dropping", session_index
            )
        finally:
            if isinstance(item, RecognizerAck):
                self._mark_segment_complete(session_index)

    def _enqueue_segment(
        self,
        segment: AudioSegment,
        *,
        block: bool,
        timeout: float | None,
    ) -> None:
        """Track session work before placing a segment on the input queue."""
        message_id = segment.message_id
        session_index = message_id // _SESSION_PREFIX if message_id is not None else None
        if session_index is None:
            raise ValueError("RecognizerService: message_id is required for routing")

        with self._routing_lock:
            state = self._session_states.get(session_index)
            if state is not None:
                state.pending_segments += 1
                if state.drain_event is not None:
                    state.drain_event.clear()

        try:
            if block:
                self._input_queue.put(segment, timeout=timeout)
            else:
                self._input_queue.put_nowait(segment)
        except Exception:
            with self._routing_lock:
                state = self._session_states.get(session_index)
                if state is not None and state.pending_segments > 0:
                    state.pending_segments -= 1
                    if state.pending_segments == 0 and state.drain_event is not None:
                        state.drain_event.set()
            raise

    def _mark_segment_complete(self, session_index: int) -> None:
        """Decrease pending segment count after terminal ACK routing."""
        with self._routing_lock:
            state = self._session_states.get(session_index)
            if state is None:
                return
            if state.pending_segments > 0:
                state.pending_segments -= 1
            if state.pending_segments == 0 and state.drain_event is not None:
                state.drain_event.set()
