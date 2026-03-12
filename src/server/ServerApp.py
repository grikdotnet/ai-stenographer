"""Orchestrates WsServer and RecognizerService: the server-side entry point.

ServerApp owns ServerApplicationState, RecognizerService, SessionManager,
and WsServer. It starts and stops all components in the correct order.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.ServerApplicationState import ServerApplicationState
from src.server.RecognizerService import RecognizerService
from src.server.SessionManager import SessionManager
from src.server.WsServer import WsServer

if TYPE_CHECKING:
    from src.asr.Recognizer import Recognizer

logger = logging.getLogger(__name__)


class ServerApp:
    """Orchestrates all server-side components for one server lifecycle.

    Creates and wires together the ServerApplicationState, RecognizerService,
    SessionManager, and WsServer. Caller provides a pre-constructed Recognizer
    (pure callable, no thread).

    Args:
        recognizer: Pure synchronous recognizer (recognize_window callable).
        config: Application config dict with audio/vad/windowing sections.
        vad_model_path: Path to the Silero VAD ONNX model file.
        host: WebSocket host to bind on.
        port: WebSocket port to bind on; 0 for OS-assigned.
    """

    def __init__(
        self,
        recognizer: "Recognizer",
        config: dict,
        vad_model_path: Path,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._config = config
        self._vad_model_path = vad_model_path
        self._stopped = False

        self.app_state = ServerApplicationState()

        self._recognizer_service = RecognizerService(
            recognizer=recognizer,
            app_state=self.app_state,
        )

        self._session_manager = SessionManager(
            recognizer_service=self._recognizer_service,
            app_state=self.app_state,
            config=config,
            vad_model_path=vad_model_path,
        )

        self._ws_server = WsServer(
            session_manager=self._session_manager,
            app_state=self.app_state,
            host=host,
            port=port,
        )

    @property
    def port(self) -> int:
        """Bound WebSocket port (available after start()).

        Returns:
            Port number; 0 if start() has not been called.
        """
        return self._ws_server.port

    def start(self) -> None:
        """Start RecognizerService and WsServer; transition state to running.

        Algorithm:
            1. Start RecognizerService inference thread.
            2. Start WsServer (binds socket, blocks until ready).
            3. Transition ServerApplicationState to running.
        """
        self._recognizer_service.start()
        self._ws_server.start()
        self.app_state.set_state("running")
        logger.info("ServerApp: running on port %s", self.port)

    def stop(self) -> None:
        """Stop all server components gracefully.

        Algorithm:
            1. Guard against double-stop (idempotent).
            2. Transition state to shutdown — RecognizerService and SessionManager
               observers fire and begin teardown.
            3. Stop WsServer event loop; join to wait for thread exit.
            4. Join RecognizerService inference thread.
        """
        if self._stopped:
            return
        self._stopped = True

        try:
            self.app_state.set_state("shutdown")
        except ValueError:
            pass

        self._ws_server.stop()
        self._ws_server.join()
        self._recognizer_service.join()
        logger.info("ServerApp: stopped")
