"""Concrete model command handler used by the WebSocket command controller."""

from src.downloader.DownloadWorker import DownloadWorker
from src.asr.ModelRegistry import ModelRegistry
from src.network.types import WsModelInfo
from src.server.protocols import IDownloadProgressEvents, IModelReadinessCoordinator


class ModelCommandHandler:
    """Execute model list and download commands against the shared registry."""

    def __init__(
        self,
        *,
        model_registry: ModelRegistry,
        model_readiness: IModelReadinessCoordinator,
        download_events: IDownloadProgressEvents,
    ) -> None:
        self._registry = model_registry
        self._worker = DownloadWorker()
        self._model_readiness = model_readiness
        self._download_events = download_events

    def get_model_list(self) -> list[WsModelInfo]:
        """Return the current model list, marking the active download if present."""
        current_model_name = self._worker.current_model_name()
        return [
            model.to_ws_model_info(
                status_override=(
                    "downloading"
                    if current_model_name == model.name and not model.is_ready()
                    else None
                )
            )
            for model in self._registry.get_downloadable_models()
        ]

    def is_model_ready(self, model_name: str) -> bool:
        """Return whether the requested model is already present."""
        return self._registry.get_model(model_name).is_ready()

    def ensure_model_ready(self, model_name: str) -> None:
        """Ensure the requested ready model is attached to the live server pipeline."""
        if not self.is_model_ready(model_name):
            raise ValueError(f"Model {model_name!r} is not ready")
        self._model_readiness.ensure_model_ready(model_name)

    def start_download(self, model_name: str) -> bool:
        """Start downloading the requested model in a background worker."""
        model = self._registry.get_model(model_name)
        if not model.is_downloadable():
            raise ValueError(f"Model {model_name!r} is not downloadable")

        return self._worker.start(
            model=model,
            progress_callback=lambda progress, downloaded, total: self._download_events.on_progress(
                model_name, progress, downloaded, total
            ),
            on_success=self._model_readiness.on_download_success,
            on_error=self._download_events.on_error,
        )
