"""Shared model definitions for ASR and VAD assets.

These classes centralize model-relative paths, validation rules, download
support, and user-facing metadata so callers do not re-compose model
sub-paths by hand.
"""

from pathlib import Path
from typing import Callable, Protocol

from src.downloader.ModelDownloader import ModelDownloader, validate_parakeet
from src.network.types import WsModelInfo


ModelStatus = str


class IModelDefinition(Protocol):
    """Common interface for concrete shared model definitions."""

    name: str
    display_name: str
    size_description: str
    relative_path: Path

    def get_model_path(self) -> Path:
        """Return the absolute filesystem path for this model."""

    def is_ready(self) -> bool:
        """Return whether the model is ready for use."""

    def validate(self) -> bool:
        """Validate the model installation."""

    def is_downloadable(self) -> bool:
        """Return whether this model supports download via shared flows."""

    def download(
        self,
        progress_callback: Callable[[float, int, int], None] | None = None,
    ) -> None:
        """Download or provision the model."""

    def cleanup_partial_files(self) -> None:
        """Remove partial artifacts for this model's download flow."""

    def to_ws_model_info(self, status_override: ModelStatus | None = None) -> WsModelInfo:
        """Project the model into the wire metadata format."""


class BaseModelDefinition:
    """Base behavior shared by concrete model definitions.

    Responsibilities:
    - Hold the resolved models root.
    - Resolve the absolute model path from the relative path.
    - Provide default readiness and metadata projections.
    """

    name: str
    display_name: str
    size_description: str
    relative_path: Path

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir

    def get_model_path(self) -> Path:
        """Return the absolute filesystem path for this model."""
        return self._models_dir / self.relative_path

    def is_ready(self) -> bool:
        """Return whether the model validates successfully."""
        return self.validate()

    def is_downloadable(self) -> bool:
        """Return whether shared download flows support this model."""
        return False

    def download(
        self,
        progress_callback: Callable[[float, int, int], None] | None = None,
    ) -> None:
        """Download or provision the model.

        Raises:
            ValueError: If the model does not support the shared download flow.
        """
        raise ValueError(f"Model {self.name!r} is not downloadable")

    def cleanup_partial_files(self) -> None:
        """Remove any partial download artifacts for this model."""
        return

    def to_ws_model_info(self, status_override: ModelStatus | None = None) -> WsModelInfo:
        """Project the model into the server wire metadata format."""
        status = status_override or ("downloaded" if self.is_ready() else "missing")
        return WsModelInfo(
            name=self.name,
            display_name=self.display_name,
            size_description=self.size_description,
            status=status,
        )


class ParakeetAsrModel(BaseModelDefinition):
    """Concrete shared model definition for the Parakeet ASR package."""

    name = "parakeet"
    display_name = "Parakeet TDT 0.6B v3"
    size_description = "1.25 GB"
    relative_path = Path("parakeet")

    def validate(self) -> bool:
        """Validate the Parakeet manifest and file hashes."""
        return validate_parakeet(self._models_dir)

    def is_downloadable(self) -> bool:
        """Return whether this model supports download via the shared CDN flow."""
        return True

    def download(
        self,
        progress_callback: Callable[[float, int, int], None] | None = None,
    ) -> None:
        """Download the Parakeet model package into the models root."""
        ModelDownloader(self._models_dir).download_parakeet(progress_callback)

    def cleanup_partial_files(self) -> None:
        """Remove partial artifacts from a failed Parakeet download."""
        ModelDownloader.cleanup_partial_files(self._models_dir)


class SileroVadModel(BaseModelDefinition):
    """Concrete shared model definition for the Silero VAD ONNX file."""

    name = "silero_vad"
    display_name = "Silero VAD Model"
    size_description = "2 MB"
    relative_path = Path("silero_vad") / "silero_vad.onnx"

    def validate(self) -> bool:
        """Validate that the expected ONNX file exists and is non-empty."""
        model_path = self.get_model_path()
        return model_path.exists() and model_path.stat().st_size > 0
