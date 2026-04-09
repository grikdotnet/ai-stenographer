"""
ModelManager handles model validation for STT pipeline.
"""
from pathlib import Path

from src.downloader.ModelRegistry import ModelRegistry


class ModelManager:
    """
    Manages model validation for STT pipeline.

    Supports both development mode (./models/) and distribution mode
    (_internal/models/) by accepting models_dir at construction time.

    Args:
        models_dir: Directory containing model subdirectories.
    """

    def __init__(self, models_dir: Path) -> None:
        """
        Args:
            models_dir: Directory containing model subdirectories.
        """
        self._models_dir = models_dir

    def model_exists(self) -> bool:
        """
        Returns whether the parakeet model is available.

        Returns:
            True if the parakeet installation validates successfully.
        """
        return self.validate_model('parakeet')

    def validate_model(self, model_name: str) -> bool:
        """
        Checks whether a model's primary file exists and is non-empty.

        Args:
            model_name: Name of the model ('parakeet' or 'silero_vad')

        Returns:
            True if model file exists and has non-zero size, False otherwise
        """
        if model_name == 'parakeet':
            return ModelRegistry.validate_model(model_name, self._models_dir)
        if model_name == 'silero_vad':
            model_path = self._models_dir / "silero_vad" / "silero_vad.onnx"
            return model_path.exists() and model_path.stat().st_size > 0
        return False
