"""Model validation registry for the STT pipeline.

Provides model presence checks and metadata. No download logic, no threading.
"""

from pathlib import Path
from typing import List, Optional

from src.downloader.ModelDownloader import ModelDownloader, validate_parakeet
from src.network.types import WsModelInfo


class ModelRegistry:
    """Validates model installations and provides model metadata.

    Pure query object — delegates hash validation to ModelDownloader.
    Used by DownloadManager for status queries and by main.py for startup checks.

    Args:
        models_dir: Root models directory.
    """

    _KNOWN_MODELS = {"parakeet"}

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir

    def is_known(self, model_name: str) -> bool:
        """Return True if model_name is a recognised model.

        Args:
            model_name: Model identifier to check.

        Returns:
            True if the model is known, False otherwise.
        """
        return model_name in self._KNOWN_MODELS

    def is_ready(self) -> bool:
        """Return True when Parakeet passes manifest validation.

        Returns:
            True if the model is present and valid.
        """
        return ModelDownloader(self._models_dir).validate_parakeet()

    def get_model_list(self) -> List[WsModelInfo]:
        """Return metadata for all known models with current availability status.

        Returns:
            List of WsModelInfo entries.
        """
        downloaded = validate_parakeet(self._models_dir)
        return [
            WsModelInfo(
                name="parakeet",
                display_name="Parakeet TDT 0.6B v3",
                size_description="1.25 GB",
                status="downloaded" if downloaded else "missing",
            )
        ]

    @staticmethod
    def get_missing_models(model_dir: Path) -> List[str]:
        """Return names of models that are not yet downloaded and validated.

        Args:
            model_dir: Root models directory.

        Returns:
            List of missing model names (e.g. ``["parakeet"]``), or empty list.
        """
        missing = []
        if not ModelRegistry.validate_model("parakeet", model_dir):
            missing.append("parakeet")
        return missing

    @staticmethod
    def validate_model(model_name: str, model_dir: Path) -> bool:
        """Validate a single model installation.

        For ``"parakeet"``: delegates to ``validate_parakeet()`` (manifest + hash check).
        For ``"silero_vad"``: checks that the primary ONNX file exists and is non-empty.

        Args:
            model_name: Name of the model to validate.
            model_dir: Root models directory.

        Returns:
            True if the model is fully valid, False otherwise.
        """
        if model_name == "parakeet":
            return validate_parakeet(model_dir)
        if model_name == "silero_vad":
            model_path = model_dir / "silero_vad" / "silero_vad.onnx"
            return model_path.exists() and model_path.stat().st_size > 0
        return False
