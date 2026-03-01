"""WebSocket server: accepts connections and dispatches to SessionManager.

Runs a websockets.serve() loop on a daemon asyncio event loop thread.
Each connected client gets a ClientSession via SessionManager.
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from websockets.exceptions import ConnectionClosed

from src.server.WsAudioReceiver import receive_audio
from src.server.SessionManager import SessionManager

if TYPE_CHECKING:
    from src.server.RecognizerService import RecognizerService
    from src.ServerApplicationState import ServerApplicationState

logger = logging.getLogger(__name__)


class WsServer:
    """WebSocket server that creates per-connection ClientSession instances.

    Runs on a dedicated daemon asyncio event loop thread.
    The server binds on the first call to start() and the bound port is
    available via the ``port`` property once the server is ready.

    Args:
        host: Hostname or IP to bind to (default ``"127.0.0.1"``).
        port: Port to listen on; 0 means OS assigns an available port.
        session_manager: Handles session creation and destruction.
        app_state: Server lifecycle state; stop() transitions to shutdown.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        app_state: "ServerApplicationState",
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._host = host
        self._port = port
        self._session_manager = session_manager
        self._app_state = app_state
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None
        self._ready = threading.Event()

    @property
    def port(self) -> int:
        """Return the bound port.

        Returns:
            The port number after start() completes; 0 if not started.
        """
        return self._port

    def start(self) -> None:
        """Start the asyncio event loop thread and begin accepting connections.

        Blocks until the server is bound and ready to accept connections.
        """
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="WsServer"
        )
        self._thread.start()
        self._ready.wait()

    def stop(self) -> None:
        """Stop accepting connections and shut down the event loop thread."""
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)

    def join(self, timeout: float = 5.0) -> None:
        """Wait for the event loop thread to exit.

        Args:
            timeout: Seconds to wait.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run_loop(self) -> None:
        """Run the asyncio event loop until stop() is called."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())
        self._loop.close()

    async def _serve(self) -> None:
        """Bind the WebSocket server and run the accept loop.

        Algorithm:
            1. Bind via websockets.serve() on the configured host:port.
            2. Record the actual bound port (important when port=0).
            3. Signal _ready so start() unblocks.
            4. Run until the event loop is stopped (via stop()).
            5. Close the server and all active sessions on exit.
        """
        import websockets

        async with websockets.serve(self._handle_connection, self._host, self._port) as server:
            self._server = server
            bound_port = server.sockets[0].getsockname()[1]
            self._port = bound_port
            logger.info("WsServer: listening on %s:%s", self._host, self._port)
            self._ready.set()
            await asyncio.get_event_loop().create_future()

    async def _handle_connection(self, websocket, path: str = "/") -> None:
        """Handle a single WebSocket connection for its full lifetime.

        Algorithm:
            1. Create a ClientSession via SessionManager.
            2. Run receive_audio coroutine until it returns (shutdown/disconnect).
            3. Destroy the session.

        Args:
            websocket: Connected WebSocket client.
            path: Request path (unused in v1).
        """
        loop = asyncio.get_event_loop()
        session = await self._session_manager.create_session(websocket, loop)
        session_id = session._session_id

        try:
            reason = await receive_audio(
                websocket=websocket,
                session_id=session_id,
                chunk_queue=session.chunk_queue,
            )
            logger.info(
                "WsServer: receive_audio returned reason=%s for session %s",
                reason,
                session_id,
            )
        except ConnectionClosed:
            logger.info("WsServer: connection closed unexpectedly for session %s", session_id)
        except Exception:
            logger.exception("WsServer: error in session %s", session_id)
        finally:
            await self._session_manager.destroy_session(session_id)
