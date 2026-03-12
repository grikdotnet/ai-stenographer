"""Shared ASR inference service for all active sessions.

A single inference thread calls Recognizer.recognize_window() sequentially.
Results are routed back to per-session output queues using session-prefixed
message IDs (session_index * 10_000_000).
"""

import logging
import queue
import threading
from typing import TYPE_CHECKING

from src.types import AudioSegment, RecognitionTextMessage, RecognizerAck

if TYPE_CHECKING:
    from src.asr.Recognizer import Recognizer
    from src.ServerApplicationState import ServerApplicationState

logger = logging.getLogger(__name__)

_SESSION_PREFIX = 10_000_000


class RecognizerService:
    """Owns the shared Recognizer inference thread and session output routing.

    All active sessions submit AudioSegment items to the shared input_queue.
    Session affinity is encoded in message_id: ``session_index * 10_000_000 + local_id``.
    After each inference call the result (and always an ACK) are routed to
    the correct per-session output queue by the ``message_id // 10_000_000`` key.

    Observes ServerApplicationState for shutdown; stops inference thread on transition.

    Args:
        recognizer: Pure synchronous recognizer callable (recognize_window method used).
        app_state: Server lifecycle state; RecognizerService stops on shutdown.
    """

    def __init__(
        self,
        recognizer: "Recognizer",
        app_state: "ServerApplicationState",
    ) -> None:
        self._recognizer = recognizer
        self._app_state = app_state
        self.input_queue: queue.Queue[AudioSegment] = queue.Queue()
        self._session_output_queues: dict[int, queue.Queue] = {}
        self._routing_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

        app_state.register_component_observer(self._on_state_change)

    def register_session(self, session_index: int, output_queue: queue.Queue) -> None:
        """Register a per-session output queue keyed by session_index.

        Args:
            session_index: 1-based session identifier.
            output_queue: Destination queue for RecognitionTextMessage / RecognizerAck.
        """
        with self._routing_lock:
            self._session_output_queues[session_index] = output_queue

    def unregister_session(self, session_index: int) -> None:
        """Remove a session's output queue; subsequent results for it are dropped.

        Args:
            session_index: Session to remove.
        """
        with self._routing_lock:
            self._session_output_queues.pop(session_index, None)

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
                segment: AudioSegment = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            message_id = segment.message_id
            session_index = message_id // _SESSION_PREFIX if message_id is not None else None

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
            out_queue = self._session_output_queues.get(session_index)

        if out_queue is None:
            logger.debug(
                "RecognizerService: session %s not registered, dropping result", session_index
            )
            return

        try:
            out_queue.put_nowait(item)
        except queue.Full:
            logger.warning(
                "RecognizerService: output queue full for session %s, dropping", session_index
            )
