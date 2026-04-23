"""Server-side application entry point wiring websockets and recognition."""

import logging
from typing import TYPE_CHECKING

from src.ApplicationState import ApplicationState
from src.asr.ModelRegistry import ModelRegistry
from src.asr.RecognizerFactory import IRecognizerFactory
from src.server.CommandController import CommandController
from src.server.DownloadProgressNotifier import DownloadProgressNotifier
from src.server.ModelCommandHandler import ModelCommandHandler
from src.server.RecognizerReadinessCoordinator import RecognizerReadinessCoordinator
from src.server.RecognizerService import RecognizerService
from src.server.SessionManager import SessionManager
from src.server.WsServer import WsServer


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.asr.Recognizer import Recognizer


class ServerApp:
    """Orchestrate all server-side components for one server lifecycle.

    Creates and wires together ``ApplicationState``, ``RecognizerService``,
    ``SessionManager``, and ``WsServer``. A recognizer is attached after
    construction via ``attach_recognizer()`` once the ASR model is loaded.
    """

    def __init__(
        self,
        config: dict,
        model_registry: ModelRegistry,
        app_state: ApplicationState,
        recognizer_factory: IRecognizerFactory | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._config = config
        self._model_registry = model_registry
        self._stopped = False

        self._app_state = app_state

        self._recognizer_service = RecognizerService(
            app_state=self._app_state,
        )
        vad_model = self._model_registry.get_vad_model()

        self._session_manager = SessionManager(
            recognizer_service=self._recognizer_service,
            app_state=self._app_state,
            config=config,
            vad_model=vad_model,
        )

        self._ws_server = WsServer(
            session_manager=self._session_manager,
            app_state=self._app_state,
            host=host,
            port=port,
        )
        self._download_notifier = DownloadProgressNotifier(
            broadcaster=self._session_manager,
        )
        self._recognizer_readiness = RecognizerReadinessCoordinator(
            app_state=self._app_state,
            recognizer_service=self._recognizer_service,
            recognizer_factory=recognizer_factory,
            download_events=self._download_notifier,
        )
        self._command_controller = CommandController(
            ModelCommandHandler(
                model_registry=self._model_registry,
                model_readiness=self._recognizer_readiness,
                download_events=self._download_notifier,
            )
        )
        self._ws_server.set_command_controller(self._command_controller)

    @property
    def app_state(self) -> ApplicationState:
        """Expose the shared application state for lifecycle observers."""
        return self._app_state

    def attach_recognizer(self, recognizer: "Recognizer") -> None:
        """Attach the synchronous recognizer to the recognizer service."""
        self._recognizer_readiness.attach_recognizer(recognizer)

    def join(self, timeout: float | None = None) -> None:
        """Wait for the WebSocket server thread to exit."""
        self._ws_server.join(timeout=timeout)

    def is_running(self) -> bool:
        """Return whether the WebSocket server thread is still running."""
        return self._ws_server.is_running()

    @property
    def port(self) -> int:
        """Return the bound WebSocket port after startup."""
        return self._ws_server.port

    def start(self) -> None:
        """Start ``RecognizerService`` and ``WsServer``."""
        self._recognizer_service.start()
        self._ws_server.start()
        logger.info("ServerApp: started infrastructure on port %s", self.port)

    def stop(self) -> None:
        """Stop all server components gracefully."""
        if self._stopped:
            return

        try:
            self._app_state.set_state("shutdown")
        except ValueError:
            pass
        self._stopped = True

        self._ws_server.stop()
        self._ws_server.join()
        self._recognizer_service.join()
        logger.info("ServerApp: stopped")
