"""Protocols used to decouple WebSocket command handling from model services."""

from typing import Protocol

from src.network.types import ServerMessage, WsModelInfo


class IModelCommandHandler(Protocol):
    """Protocol for server-side model command execution.

    Responsibilities:
    - Provide the current model list for ``list_models``.
    - Report whether a named model is already ready.
    - Start a background model download when requested.
    """

    def get_model_list(self) -> list[WsModelInfo]:
        """Return the current downloadable model list."""

    def is_model_ready(self, model_name: str) -> bool:
        """Return whether the given model is already available for use."""

    def ensure_model_ready(self, model_name: str) -> None:
        """Ensure a ready model is attached to the active recognizer pipeline."""

    def start_download(self, model_name: str) -> bool:
        """Start a model download.

        Returns:
            True if the download was accepted, False if another download is already running.

        Raises:
            ValueError: If the model name is invalid.
        """


class IDownloadProgressEvents(Protocol):
    """Protocol for publishing model download status events."""

    def on_progress(
        self,
        model_name: str,
        progress: float,
        downloaded_bytes: int,
        total_bytes: int,
    ) -> None:
        """Publish an in-progress model download update."""

    def on_complete(self, model_name: str) -> None:
        """Publish a completed model download update."""

    def on_error(self, model_name: str, exc: Exception) -> None:
        """Publish a failed model download update."""


class IModelReadinessCoordinator(Protocol):
    """Protocol for making a downloaded or existing model ready for inference."""

    def ensure_model_ready(self, model_name: str) -> None:
        """Ensure a ready model is attached to the active recognizer pipeline."""

    def on_download_success(self, model_name: str) -> None:
        """Handle a completed model download before publishing completion."""


class IServerMessageBroadcaster(Protocol):
    """Protocol for broadcasting a server message to every active session.

    Callers pass a ``ServerMessage`` dataclass. Implementers encode it with
    ``encode_server_message`` and deliver it to all connected sessions.
    """

    def broadcast(self, message: ServerMessage) -> None:
        """Encode ``message`` and deliver it to every active session."""
