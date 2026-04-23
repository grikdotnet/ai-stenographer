"""Session lifecycle manager: creates and destroys ClientSession instances.

Maintains an atomic session_index counter.  Each new WebSocket connection
gets a unique session (1-based) that partitions its message_id space.
"""

import logging
import threading
import time
import uuid
from typing import Any, TYPE_CHECKING

from src.network.codec import encode_server_message
from src.network.types import ServerMessage, WsServerState, WsSessionCreated
from src.asr.ModelDefinitions import SileroVadModel
from src.server.ClientSession import ClientSession
from src.server.protocols import IServerMessageBroadcaster

if TYPE_CHECKING:
    from src.server.RecognizerService import RecognizerService
    from src.ApplicationState import ApplicationState

logger = logging.getLogger(__name__)


class SessionManager(IServerMessageBroadcaster):
    """Creates and destroys ClientSession objects; tracks active sessions.

    Args:
        recognizer_service: Shared inference service passed to each ClientSession.
        app_state: Server lifecycle state observed for server_state broadcasts.
        config: Application configuration dict (audio/vad/windowing sections).
        vad_model: Shared Silero VAD model definition.
    """

    def __init__(
        self,
        recognizer_service: "RecognizerService",
        app_state: "ApplicationState",
        config: dict,
        vad_model: SileroVadModel,
    ) -> None:
        self._recognizer_service = recognizer_service
        self._app_state = app_state
        self._config = config
        self._vad_model = vad_model

        self._session_index_counter = 0
        self._counter_lock = threading.Lock()

        self._sessions: dict[str, ClientSession] = {}
        self._sessions_lock = threading.Lock()

        app_state.register_component_observer(self.on_state_change)

    async def create_session(self, websocket: Any, loop: Any) -> ClientSession:
        """Create a new ClientSession for a connected WebSocket client.

        Algorithm:
            1. Allocate unique session_index (atomic increment under lock).
            2. Generate UUID session_id.
            3. Build and start a ClientSession (registers with RecognizerService).
            4. Queue session_created + server_state through the session send queue.
            5. Flush the welcome pair before registering the session for broadcasts.

        Args:
            websocket: Active WebSocket connection.
            loop: asyncio event loop running the connection handler.

        Returns:
            Started ClientSession instance.
        """
        with self._counter_lock:
            self._session_index_counter += 1

        session_id = str(uuid.uuid4())

        session = ClientSession(
            session_id=session_id,
            session_index=self._session_index_counter,
            websocket=websocket,
            loop=loop,
            recognizer_service=self._recognizer_service,
            config=self._config,
            vad_model=self._vad_model,
        )

        await session.start()

        created_msg = WsSessionCreated(
            session_id=session_id,
            protocol_version="v1",
            server_time=time.time(),
        )
        try:
            session.send_encoded(encode_server_message(created_msg))
            session.send_encoded(
                encode_server_message(
                    WsServerState(state=self._app_state.get_state())  # type: ignore[arg-type]
                )
            )
            await session.wait_for_send_drain()
        except Exception:
            logger.warning(
                "SessionManager: send failed for session %s — destroying to prevent leak", session_id
            )
            await session.close()
            raise

        with self._sessions_lock:
            self._sessions[session_id] = session

        logger.info(
            "SessionManager: session created id=%s index=%s", session_id, self._session_index_counter
        )
        return session

    def iter_sessions(self) -> list[ClientSession]:
        """Return a stable snapshot of active sessions."""
        with self._sessions_lock:
            return list(self._sessions.values())

    def broadcast(self, message: ServerMessage) -> None:
        """Encode a server message once and fan it out to every active session."""
        encoded = encode_server_message(message)
        with self._sessions_lock:
            sessions = list(self._sessions.values())

        for session in sessions:
            session.send_encoded(encoded)

    def on_state_change(self, old_state: str, new_state: str) -> None:
        """Broadcast the new server state to every active session."""
        self.broadcast(WsServerState(state=new_state))  # type: ignore[arg-type]

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

    async def close_session(
        self,
        session_id: str,
        *,
        reason: str,
        message: str | None = None,
    ) -> None:
        """Gracefully close a session and remove it from the active map."""
        with self._sessions_lock:
            session = self._sessions.pop(session_id, None)

        if session is None:
            logger.warning("SessionManager: close_session called for unknown id=%s", session_id)
            return

        await session.close_gracefully(reason=reason, message=message)
        logger.info("SessionManager: session gracefully closed id=%s", session_id)

    async def close_all_sessions(self) -> None:
        """Close all active sessions (called on server shutdown)."""
        with self._sessions_lock:
            session_items = list(self._sessions.items())
            self._sessions.clear()

        for session_id, session in session_items:
            try:
                await session.close_gracefully(
                    reason="error",
                    message="server shutdown",
                )
            except Exception:
                logger.exception("SessionManager: error closing session %s", session_id)

        logger.info("SessionManager: all sessions closed")
