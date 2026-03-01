"""WebSocket result sender: sync-to-async bridge for recognition results.

Implements the RecognitionResultPublisher protocol so IncrementalTextMatcher
can call publish_partial_update / publish_finalization from its sync thread
and have results forwarded to the client WebSocket without blocking.
"""

import asyncio
import logging
from typing import Any, Protocol

from src.network.codec import encode_server_message
from src.network.types import WsRecognitionResult
from src.types import RecognitionResult

logger = logging.getLogger(__name__)

_SEND_QUEUE_MAXSIZE = 20


class _SupportsAsyncSend(Protocol):
    """Structural protocol for an object with an async send method."""

    async def send(self, message: str) -> None: ...


class WsResultSender:
    """Bridges the synchronous pipeline to an async WebSocket send channel.

    Implements publish_partial_update / publish_finalization so it can be
    passed directly as the publisher argument to IncrementalTextMatcher.

    A bounded asyncio.Queue (maxsize=20) decouples the sync callers from the
    async sender task.  When the queue is full the incoming message is dropped
    and logged — this provides backpressure without ever blocking the caller.

    Args:
        session_id: Session identifier included in every outbound message.
        websocket: Object with an ``async send(str)`` method.
        loop: The asyncio event loop running the sender task.
    """

    def __init__(
        self,
        session_id: str,
        websocket: _SupportsAsyncSend,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._session_id = session_id
        self._websocket = websocket
        self._loop = loop
        self._send_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=_SEND_QUEUE_MAXSIZE)
        self._sender_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background sender task.

        Must be called from within the asyncio event loop thread before any
        publish calls are made.
        """
        self._send_queue = asyncio.Queue(maxsize=_SEND_QUEUE_MAXSIZE)
        self._sender_task = asyncio.get_event_loop().create_task(self._drain_loop())

    async def stop(self) -> None:
        """Cancel the sender task, waiting briefly for a clean exit.

        Args: none.
        """
        if self._sender_task is None:
            return
        self._sender_task.cancel()
        try:
            await self._sender_task
        except asyncio.CancelledError:
            pass
        self._sender_task = None

    # ------------------------------------------------------------------
    # RecognitionResultPublisher protocol
    # ------------------------------------------------------------------

    def publish_partial_update(self, result: RecognitionResult) -> None:
        """Enqueue a partial recognition result for async delivery.

        Thread-safe: may be called from any thread.

        Args:
            result: Incremental RecognitionResult from IncrementalTextMatcher.
        """
        self._enqueue(_encode(self._session_id, result, status="partial"))

    def publish_finalization(self, result: RecognitionResult) -> None:
        """Enqueue a final recognition result for async delivery.

        Thread-safe: may be called from any thread.

        Args:
            result: Finalized RecognitionResult from IncrementalTextMatcher.
        """
        self._enqueue(_encode(self._session_id, result, status="final"))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enqueue(self, encoded: str) -> None:
        """Schedule a put on the send queue from a sync thread.

        Uses loop.call_soon_threadsafe to safely cross the thread boundary.
        Drops the message if the queue is full.
        """
        self._loop.call_soon_threadsafe(self._put_nowait, encoded)

    def _put_nowait(self, encoded: str) -> None:
        """Put message on the queue; drop and log if full (runs in loop thread)."""
        try:
            self._send_queue.put_nowait(encoded)
        except asyncio.QueueFull:
            logger.warning(
                "WsResultSender[%s]: send queue full, dropping message", self._session_id
            )

    async def _drain_loop(self) -> None:
        """Async task: drain the send queue and call websocket.send.

        Algorithm:
            1. await queue.get() — blocks until a message is available.
            2. await websocket.send(message) — delivers to client.
            3. On any send error, log and continue (connection errors stop naturally on next get).
        """
        while True:
            encoded = await self._send_queue.get()
            try:
                await self._websocket.send(encoded)
            except Exception:
                logger.exception(
                    "WsResultSender[%s]: error sending message", self._session_id
                )


def _encode(session_id: str, result: RecognitionResult, status: str) -> str:
    """Build a JSON-encoded WsRecognitionResult string.

    Args:
        session_id: Session to stamp on the message.
        result: Source RecognitionResult.
        status: Wire status string (``"partial"`` or ``"final"``).

    Returns:
        JSON string ready for WebSocket text frame transmission.
    """
    msg = WsRecognitionResult(
        session_id=session_id,
        status=status,  # type: ignore[arg-type]
        text=result.text,
        start_time=result.start_time,
        end_time=result.end_time,
        chunk_ids=result.chunk_ids,
        utterance_id=getattr(result, "utterance_id", 0),
        confidence=result.confidence,
        token_confidences=result.token_confidences,
        audio_rms=result.audio_rms,
        confidence_variance=result.confidence_variance,
    )
    return encode_server_message(msg)
