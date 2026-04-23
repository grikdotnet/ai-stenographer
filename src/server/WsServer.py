"""WebSocket server: accepts connections and dispatches to SessionManager.

Runs a websockets.serve() loop on a daemon asyncio event loop thread.
Each connected client gets a ClientSession via SessionManager.
"""

import asyncio
import logging
import threading

from websockets.exceptions import ConnectionClosed

from src.ApplicationState import ApplicationState
from src.network.codec import encode_server_message
from src.server.CommandController import CommandController
from src.server.WsAudioReceiver import handle_audio_frame
from src.server.WsMessageRouter import (
    RoutedAudioMessage,
    RoutedCommandMessage,
    RoutedProtocolError,
    route_incoming_message,
)
from src.server.SessionManager import SessionManager

logger = logging.getLogger(__name__)

DEFAULT_SERVER_HOST = "127.0.0.1"


class WsServer:
    """WebSocket server that creates per-connection ClientSession instances.

    Runs on a dedicated daemon asyncio event loop thread.
    The server binds on the first call to start() and the bound port is
    available via the ``port`` property once the server is ready.

    Args:
        host: Hostname or IP to bind to (default ``"127.0.0.1"``).
        port: Port to listen on; 0 means OS assigns an available port.
        session_manager: Handles session creation and destruction.
        app_state: Server lifecycle state used to accept or reject audio frames.
        command_controller: Executes client control commands.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        app_state: ApplicationState,
        command_controller: CommandController | None = None,
        host: str = DEFAULT_SERVER_HOST,
        port: int = 0,
    ) -> None:
        self._host = host
        self._port = port
        self._session_manager = session_manager
        self._app_state = app_state
        self._command_controller = command_controller
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None
        self._ready = threading.Event()
        self._startup_error: Exception | None = None
        self._stop_event: asyncio.Event | None = None
        self._drain_task: asyncio.Task | None = None

    @property
    def port(self) -> int:
        """Return the bound port.

        Returns:
            The port number after start() completes; 0 if not started.
        """
        return self._port

    def set_command_controller(self, command_controller: CommandController) -> None:
        """Inject the command controller after construction."""
        self._command_controller = command_controller

    def start(self) -> None:
        """Start the asyncio event loop thread and begin accepting connections.

        Blocks until the server is bound and ready to accept connections.

        Raises:
            Exception: Any exception raised while binding the WebSocket server.
        """
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="WsServer"
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error

    def stop(self) -> None:
        """Stop accepting connections and shut down the event loop thread."""
        if self._loop is None or self._stop_event is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        except RuntimeError:
            pass

    def join(self, timeout: float | None = None) -> None:
        """Wait for the event loop thread to exit.

        Args:
            timeout: Seconds to wait; None blocks until the thread exits.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        """Return whether the event loop thread is currently alive."""
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        """Run the asyncio event loop until stop() is called."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            if not self._ready.is_set():
                self._startup_error = exc
        finally:
            self._ready.set()
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
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
            self._stop_event = asyncio.Event()
            self._ready.set()
            await self._stop_event.wait()
            await self._session_manager.close_all_sessions()

    async def _handle_connection(self, websocket, path: str = "/") -> None:
        """Handle a single WebSocket connection for its full lifetime.

        Algorithm:
            1. Create a ClientSession via SessionManager.
            2. Read websocket frames for the lifetime of the connection.
            3. Route audio frames to the audio receiver and text frames to CommandController.
            4. Destroy the session on disconnect or close_session.

        Args:
            websocket: Connected WebSocket client.
            path: Request path (unused in v1).
        """
        loop = asyncio.get_event_loop()
        session = await self._session_manager.create_session(websocket, loop)
        session_id = session.session_id
        gracefully_closed = False

        try:
            async for message in websocket:
                routed = route_incoming_message(message, session_id)

                if isinstance(routed, RoutedProtocolError):
                    await websocket.send(encode_server_message(routed.error))
                    continue

                if isinstance(routed, RoutedAudioMessage):
                    error = handle_audio_frame(
                        raw=routed.raw,
                        session_id=session_id,
                        chunk_queue=session.chunk_queue,
                        server_state=self._app_state.get_state(),
                    )
                    if error is not None:
                        await websocket.send(encode_server_message(error))
                    continue

                if isinstance(routed, RoutedCommandMessage):
                    if self._command_controller is None:
                        raise RuntimeError("WsServer command controller is not configured")
                    should_close, response = self._command_controller.handle(
                        routed.text,
                        session_id=session_id,
                    )
                    if response is not None:
                        await websocket.send(encode_server_message(response))
                    if should_close:
                        await self._session_manager.close_session(
                            session_id,
                            reason="close_session",
                        )
                        gracefully_closed = True
                        await self._close_websocket_if_possible(websocket)
                        return
        except ConnectionClosed:
            logger.info("WsServer: connection closed unexpectedly for session %s", session_id)
        except Exception:
            logger.exception("WsServer: error in session %s", session_id)
            try:
                await self._session_manager.close_session(
                    session_id,
                    reason="error",
                    message="internal server error",
                )
                gracefully_closed = True
                await self._close_websocket_if_possible(websocket)
            except Exception:
                logger.exception(
                    "WsServer: failed to close errored session %s gracefully",
                    session_id,
                )
        finally:
            if not gracefully_closed:
                await self._session_manager.destroy_session(session_id)

    async def _close_websocket_if_possible(self, websocket) -> None:
        """Close websocket when the object supports an async close method."""
        close = getattr(websocket, "close", None)
        if close is None:
            return
        await close()
