"""Session lifecycle manager: creates and destroys ClientSession instances.

Maintains an atomic session_index counter.  Each new WebSocket connection
gets a unique session (1-based) that partitions its message_id space.
"""

import logging
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.network.codec import encode_server_message
from src.network.types import WsSessionCreated
from src.server.ClientSession import ClientSession
from src.server.broadcast import _SHUTDOWN

if TYPE_CHECKING:
    from src.server.RecognizerService import RecognizerService
    from src.ApplicationState import ApplicationState

logger = logging.getLogger(__name__)


class SessionManager:
    """Creates and destroys ClientSession objects; tracks active sessions.

    Observes ApplicationState for server shutdown and closes all
    active sessions when the server transitions to shutdown.

    Args:
        recognizer_service: Shared inference service passed to each ClientSession.
        app_state: Server lifecycle state; triggers mass session close on shutdown.
        config: Application configuration dict (audio/vad/windowing sections).
        vad_model_path: Path to the Silero VAD ONNX model file.
        broadcast_queue: Shared queue owned by ServerApp; puts _SHUTDOWN on shutdown.
    """

    def __init__(
        self,
        recognizer_service: "RecognizerService",
        app_state: "ApplicationState",
        config: dict,
        vad_model_path: Path,
        broadcast_queue: queue.SimpleQueue,
    ) -> None:
        self._recognizer_service = recognizer_service
        self._app_state = app_state
        self._config = config
        self._vad_model_path = vad_model_path
        self._broadcast_sink = broadcast_queue

        self._session_index_counter = 0
        self._counter_lock = threading.Lock()

        self._sessions: dict[str, ClientSession] = {}
        self._sessions_lock = threading.Lock()

        app_state.register_component_observer(self._on_state_change)

    async def create_session(self, websocket: Any, loop: Any) -> ClientSession:
        """Create a new ClientSession for a connected WebSocket client.

        Algorithm:
            1. Allocate unique session_index (atomic increment under lock).
            2. Generate UUID session_id.
            3. Build and start a ClientSession (registers with RecognizerService).
            4. Send session_created JSON frame to client.

        Args:
            websocket: Active WebSocket connection.
            loop: asyncio event loop running the connection handler.

        Returns:
            Started ClientSession instance.
        """
        with self._counter_lock:
            self._session_index_counter += 1
            session_index = self._session_index_counter

        session_id = str(uuid.uuid4())

        session = ClientSession(
            session_id=session_id,
            session_index=session_index,
            websocket=websocket,
            loop=loop,
            recognizer_service=self._recognizer_service,
            app_state=self._app_state,
            config=self._config,
            vad_model_path=self._vad_model_path,
        )

        await session.start()

        with self._sessions_lock:
            self._sessions[session_id] = session

        created_msg = WsSessionCreated(
            session_id=session_id,
            protocol_version="v1",
            server_time=time.time(),
        )
        try:
            await websocket.send(encode_server_message(created_msg))
        except Exception:
            logger.warning(
                "SessionManager: send failed for session %s — destroying to prevent leak", session_id
            )
            await self.destroy_session(session_id)
            raise
        logger.info(
            "SessionManager: session created id=%s index=%s", session_id, session_index
        )
        return session

    async def destroy_session(self, session_id: str) -> None:
        """Close and remove a session by ID.

        Args:
            session_id: UUID of the session to destroy.
        """
        with self._sessions_lock:
            session = self._sessions.pop(session_id, None)

        if session is None:
            logger.warning("SessionManager: destroy_session called for unknown id=%s", session_id)
            return

        await session.close()
        logger.info("SessionManager: session destroyed id=%s", session_id)

    async def close_all_sessions(self) -> None:
        """Close all active sessions (called on server shutdown)."""
        with self._sessions_lock:
            session_items = list(self._sessions.items())
            self._sessions.clear()

        for session_id, session in session_items:
            try:
                await session.close()
            except Exception:
                logger.exception("SessionManager: error closing session %s", session_id)

        logger.info("SessionManager: all sessions closed")

    def _on_state_change(self, old_state: str, new_state: str) -> None:
        """Observe server shutdown and send a signal to WsServer via queue.
        Runs on whatever thread changes the state.

        Only the WsServer's async loop can call close_all_sessions(). 
        So we put _SHUTDOWN into the broadcast sink (queue), and WsServer calls close_all_sessions();

        Args:
            old_state: Previous server state.
            new_state: New server state.
        """
        if new_state == "shutdown":
            logger.info("SessionManager: server shutdown observed")
            if self._broadcast_sink is not None:
                self._broadcast_sink.put(_SHUTDOWN)
