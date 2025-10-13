"""
ModelManager handles model download and validation for STT pipeline.
"""
from pathlib import Path
from typing import List, Callable, Optional
from huggingface_hub import snapshot_download, hf_hub_download
import shutil


MODEL_DIR = Path("./models")
PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
SILERO_REPO = "onnx-community/silero-vad"
SILERO_FILE = "onnx/model.onnx"


class ModelManager:
    """
    Manages model downloads and validation for STT pipeline.

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
            List of missing model names: ['parakeet', 'silero_vad'] or []
        """
        if model_dir is None:
            model_dir = MODEL_DIR

        missing = []

        if not ModelManager.validate_model('parakeet', model_dir):
            missing.append('parakeet')

        if not ModelManager.validate_model('silero_vad', model_dir):
            missing.append('silero_vad')

        return missing

    @staticmethod
    def download_models(model_dir: Optional[Path] = None, progress_callback: Optional[Callable[[str, float, str], None]] = None) -> bool:
        """
        Downloads missing models with optional progress tracking.

        Args:
            model_dir: Directory to download models to (defaults to ./models)
            progress_callback: Callback function with signature:
                def callback(model_name: str, progress: float, status: str):
                    # progress: 0.0 to 1.0
                    # status: 'downloading' | 'complete' | 'error'

        Returns:
            True if all downloads succeeded, False otherwise
        """
        if model_dir is None:
            model_dir = MODEL_DIR

        missing = ModelManager.get_missing_models(model_dir)

        try:
            for model_name in missing:
                if progress_callback:
                    progress_callback(model_name, 0.0, 'downloading')

                if model_name == 'parakeet':
                    ModelManager._download_parakeet(model_dir)
                elif model_name == 'silero_vad':
                    ModelManager._download_silero(model_dir)

                if progress_callback:
                    progress_callback(model_name, 1.0, 'complete')

            return True

        except Exception as e:
            print(f"Model download error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

            if progress_callback:
                progress_callback(model_name if 'model_name' in locals() else 'unknown', 0.0, 'error')

            ModelManager._cleanup_partial_files(model_dir)
            return False

    @staticmethod
    def validate_model(model_name: str, model_dir: Optional[Path] = None) -> bool:
        """
        Validates that a model exists and has reasonable size.

        Args:
            model_name: Name of model to validate ('parakeet' or 'silero_vad')
            model_dir: Directory containing models (defaults to ./models)

        Returns:
            True if model file exists and is valid, False otherwise
        """
        if model_dir is None:
            model_dir = MODEL_DIR

        if model_name == 'parakeet':
            model_path = model_dir / "parakeet" / "encoder-model.onnx"
        elif model_name == 'silero_vad':
            model_path = model_dir / "silero_vad" / "silero_vad.onnx"
        else:
            return False

        if not model_path.exists():
            return False

        if model_path.stat().st_size == 0:
            return False

        return True

    @staticmethod
    def _download_parakeet(model_dir: Path):
        """
        Downloads Parakeet STT model from HuggingFace.

        Uses ignore_patterns to skip unnecessary files (quantized models,
        unused feature extractors, and documentation), saving ~800MB.

        Args:
            model_dir: Directory to download model to
        """
        parakeet_dir = model_dir / "parakeet"
        parakeet_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=PARAKEET_REPO,
            local_dir=str(parakeet_dir),
            ignore_patterns=["*.int8.onnx", "nemo128.onnx", "README.md"]
        )

    @staticmethod
    def _download_silero(model_dir: Path):
        """
        Downloads Silero VAD model from HuggingFace.

        HuggingFace downloads to onnx/model.onnx, but the application expects
        silero_vad.onnx at the root. This method copies the file and removes
        the onnx directory to save disk space.

        Args:
            model_dir: Directory to download model to
        """
        silero_dir = model_dir / "silero_vad"
        silero_dir.mkdir(parents=True, exist_ok=True)

        hf_hub_download(
            repo_id=SILERO_REPO,
            filename=SILERO_FILE,
            local_dir=str(silero_dir)
        )

        source_path = silero_dir / SILERO_FILE
        target_path = silero_dir / "silero_vad.onnx"

        if source_path.exists():
            shutil.copy2(source_path, target_path)
            # Remove the onnx directory to save disk space (~0.2GB)
            onnx_dir = silero_dir / "onnx"
            if onnx_dir.exists():
                shutil.rmtree(onnx_dir)

    @staticmethod
    def _cleanup_partial_files(model_dir: Path):
        """
        Removes partial download files after errors.

        Args:
            model_dir: Root models directory
        """
        for subdir_name in ["parakeet", "silero_vad"]:
            subdir = model_dir / subdir_name
            if subdir.exists():
                for partial_file in subdir.glob("*.partial"):
                    try:
                        partial_file.unlink()
                    except Exception:
                        pass  # Best effort cleanup
