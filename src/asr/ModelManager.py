"""
ModelManager handles model validation for STT pipeline.
"""
from pathlib import Path
from typing import List, Optional


MODEL_DIR = Path("./models")


class ModelManager:
    """
    Manages model validation for STT pipeline.

    Supports both development mode (./models/) and distribution mode
    (_internal/models/) by accepting model_dir parameter.
    """

    @staticmethod
    def get_missing_models(model_dir: Optional[Path] = None) -> List[str]:
        """
        Returns list of missing model names.

        Args:
            model_dir: Directory containing models (defaults to ./models)

        Returns:
            List of missing model names: ['parakeet'] or []
            Note: Silero VAD is bundled with distribution and never missing
        """
        if model_dir is None:
            model_dir = MODEL_DIR

        missing = []

        if not ModelManager.validate_model('parakeet', model_dir):
            missing.append('parakeet')

        return missing

    @staticmethod
    def validate_model(model_name: str, model_dir: Optional[Path] = None) -> bool:
        """
        Checks whether a model's primary file exists and is non-empty.

        Args:
            model_name: Name of the model ('parakeet' or 'silero_vad')
            model_dir: Directory containing models (defaults to ./models)

        Returns:
            True if model file exists and has non-zero size, False otherwise
        """
        if model_dir is None:
            model_dir = MODEL_DIR

        if model_name == 'parakeet':
            model_path = model_dir / "parakeet" / "encoder-model.fp16.onnx"
        elif model_name == 'silero_vad':
            model_path = model_dir / "silero_vad" / "silero_vad.onnx"
        else:
            return False

        if not model_path.exists():
            return False

        if model_path.stat().st_size == 0:
            return False

        return True
