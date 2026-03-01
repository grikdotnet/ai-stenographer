"""WebSocket transport for the Tk client: async-to-sync audio send and sync-to-async result receive.

Runs an asyncio event loop on a daemon thread.  Outbound audio frames travel
from the sync AudioSource thread through a bounded asyncio.Queue to the
websocket send coroutine.  Inbound JSON text frames are dispatched directly to
RemoteRecognitionPublisher from the async receive coroutine.
"""

import asyncio
import itertools
import logging
from typing import Any

from src.ApplicationState import ApplicationState
from src.client.tk.RemoteRecognitionPublisher import RemoteRecognitionPublisher
from src.network.codec import encode_audio_frame
from src.network.types import WsAudioFrame

logger = logging.getLogger(__name__)

_SEND_QUEUE_MAXSIZE = 20
_SAMPLE_RATE = 16000


class _SupportsAsyncSend(Any.__class__):
    pass


class WsClientTransport:
    """Async WebSocket transport for the Tk client.

    Bridges the synchronous AudioSource to the async WebSocket connection.
    Outbound: sync send_audio_chunk() → bounded asyncio.Queue → websocket.send(bytes).
    Inbound: websocket text frames → RemoteRecognitionPublisher.dispatch(json_text).

    On ConnectionClosed the receive loop transitions app_state to "shutdown" so
    the client-side observer chain stops AudioSource and exits the Tk main loop.

    Args:
        server_url: WebSocket URL (ws://host:port).
        session_id: Session identifier stamped on every outbound audio frame.
        app_state: Application state; transitions to "shutdown" on connection loss.
        publisher: Decodes incoming JSON frames and calls the local subscriber.
        loop: The asyncio event loop running the send/receive tasks.
    """

    def __init__(
        self,
        server_url: str,
        session_id: str,
        app_state: ApplicationState,
        publisher: RemoteRecognitionPublisher,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._server_url = server_url
        self._session_id = session_id
        self._app_state = app_state
        self._publisher = publisher
        self._loop = loop
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_SEND_QUEUE_MAXSIZE)
        self._chunk_counter = itertools.count(1)
        self._drain_task: asyncio.Task | None = None
        self._receive_task: asyncio.Task | None = None
        self._websocket: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, websocket: Any) -> None:
        """Attach to an already-connected websocket and start send/receive tasks.

        Args:
            websocket: Object supporting async send(bytes/str) and async iteration.
        """
        self._websocket = websocket
        self._send_queue = asyncio.Queue(maxsize=_SEND_QUEUE_MAXSIZE)
        self._drain_task = asyncio.get_event_loop().create_task(self._drain_loop())
        self._receive_task = asyncio.get_event_loop().create_task(self._receive_loop())

    async def stop(self) -> None:
        """Cancel send/receive tasks and close the websocket."""
        for task in (self._drain_task, self._receive_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._drain_task = None
        self._receive_task = None

        if self._websocket is not None:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

    # ------------------------------------------------------------------
    # Sync-callable send interface
    # ------------------------------------------------------------------

    def send_audio_chunk(self, chunk: dict) -> None:
        """Thread-safe: encode a chunk dict as a binary WsAudioFrame and enqueue for send.

        Args:
            chunk: Dict with keys ``audio`` (numpy float32 array) and ``timestamp`` (float).
                   chunk_id is assigned from an internal monotonic counter.
        """
        audio = chunk["audio"]
        frame = WsAudioFrame(
            session_id=self._session_id,
            chunk_id=next(self._chunk_counter),
            timestamp=chunk["timestamp"],
            sample_rate=_SAMPLE_RATE,
            num_samples=len(audio),
            dtype="float32",
            channels=1,
            audio=audio,
        )
        encoded = encode_audio_frame(frame)
        self._loop.call_soon_threadsafe(self._put_nowait, encoded)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _put_nowait(self, data: bytes) -> None:
        """Put encoded frame on queue; drop and log if full (runs in loop thread)."""
        try:
            self._send_queue.put_nowait(data)
        except asyncio.QueueFull:
            logger.warning(
                "WsClientTransport[%s]: send queue full, dropping audio frame",
                self._session_id,
            )

    async def _drain_loop(self) -> None:
        """Async task: drain the send queue and call websocket.send(bytes).

        Algorithm:
            1. await queue.get() — blocks until an encoded frame is available.
            2. await websocket.send(frame) — delivers binary frame to server.
            3. On send error, log and continue.
        """
        while True:
            data = await self._send_queue.get()
            try:
                await self._websocket.send(data)
            except Exception:
                logger.exception(
                    "WsClientTransport[%s]: error sending audio frame", self._session_id
                )

    async def _receive_loop(self) -> None:
        """Async task: iterate websocket text frames and dispatch to publisher.

        Algorithm:
            1. async-iterate websocket frames.
            2. Call publisher.dispatch(text) for each text frame.
            3. On ConnectionClosed: log and transition app_state to "shutdown".
        """
        from websockets.exceptions import ConnectionClosed

        try:
            async for message in self._websocket:
                if isinstance(message, str):
                    self._publisher.dispatch(message)
                else:
                    logger.debug(
                        "WsClientTransport[%s]: unexpected binary frame in receive loop",
                        self._session_id,
                    )
        except ConnectionClosed:
            logger.warning(
                "WsClientTransport[%s]: connection closed by server", self._session_id
            )
            try:
                self._app_state.set_state("shutdown")
            except ValueError:
                pass
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "WsClientTransport[%s]: unexpected error in receive loop", self._session_id
            )
            try:
                self._app_state.set_state("shutdown")
            except ValueError:
                pass
