"""Coordinate recognizer attachment when server models become available."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from src.ApplicationState import ApplicationState
from src.asr.RecognizerFactory import IRecognizerFactory
from src.server.RecognizerService import RecognizerService
from src.server.protocols import IDownloadProgressEvents

if TYPE_CHECKING:
    from src.asr.Recognizer import Recognizer

logger = logging.getLogger(__name__)


class RecognizerReadinessCoordinator:
    """Attach recognizers and publish download completion outcomes.

    Responsibilities:
    - Create and attach a recognizer when a model becomes available.
    - Track whether the shared recognizer service already has a recognizer.
    - Transition ``ApplicationState`` from ``waiting_for_model`` to ``running``.
    - Convert post-download attach failures into download error events.
    """

    def __init__(
        self,
        *,
        app_state: ApplicationState,
        recognizer_service: RecognizerService,
        recognizer_factory: IRecognizerFactory | None,
        download_events: IDownloadProgressEvents,
    ) -> None:
        self._app_state = app_state
        self._recognizer_service = recognizer_service
        self._recognizer_factory = recognizer_factory
        self._download_events = download_events
        self._recognizer_lock = threading.Lock()
        self._recognizer_attached = False

    def attach_recognizer(self, recognizer: "Recognizer") -> None:
        """Attach an already-created recognizer to the recognizer service.

        Args:
            recognizer: Recognizer instance created by startup orchestration.
        """
        with self._recognizer_lock:
            self._recognizer_service.attach_recognizer(recognizer)
            self._recognizer_attached = True

    def on_download_success(self, model_name: str) -> None:
        """Attach recognizer after download and publish the terminal outcome.

        Args:
            model_name: Downloaded model name.
        """
        try:
            self.ensure_model_ready(model_name)
        except Exception as exc:
            self._download_events.on_error(model_name, exc)
            logger.exception(
                "RecognizerReadinessCoordinator: failed to attach recognizer after downloading %s",
                model_name,
            )
            return
        self._download_events.on_complete(model_name)

    def ensure_model_ready(self, model_name: str) -> None:
        """Attach a recognizer on demand and transition to running if possible.

        Args:
            model_name: Model name that triggered readiness.

        Raises:
            RuntimeError: If no recognizer factory is configured.
        """
        with self._recognizer_lock:
            if self._recognizer_attached:
                self._transition_waiting_to_running()
                return
            if self._recognizer_factory is None:
                raise RuntimeError("Recognizer factory is not configured")

            recognizer = self._recognizer_factory.create_recognizer()
            self._recognizer_service.attach_recognizer(recognizer)
            self._recognizer_attached = True

        self._transition_waiting_to_running()

    def _transition_waiting_to_running(self) -> None:
        """Move the server to running when it was waiting for a model."""
        if self._app_state.get_state() == "waiting_for_model":
            self._app_state.set_state("running")
