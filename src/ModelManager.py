"""
ModelManager handles model download and validation for STT pipeline.
"""
from pathlib import Path
from typing import List, Callable, Optional
from huggingface_hub import snapshot_download, hf_hub_download
import shutil


# Model configuration
MODEL_DIR = Path("./models")
PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
SILERO_REPO = "onnx-community/silero-vad"
SILERO_FILE = "onnx/model.onnx"


class ModelManager:
    """Manages model downloads and validation for STT pipeline."""

    @staticmethod
    def get_missing_models() -> List[str]:
        """
        Returns list of missing model names.

        Returns:
            List of missing model names: ['parakeet', 'silero_vad'] or []
        """
        missing = []

        if not ModelManager.validate_model('parakeet'):
            missing.append('parakeet')

        if not ModelManager.validate_model('silero_vad'):
            missing.append('silero_vad')

        return missing

    @staticmethod
    def download_models(progress_callback: Optional[Callable[[str, float, str], None]] = None) -> bool:
        """
        Downloads missing models with optional progress tracking.

        Args:
            progress_callback: Callback function with signature:
                def callback(model_name: str, progress: float, status: str):
                    # progress: 0.0 to 1.0
                    # status: 'downloading' | 'complete' | 'error'

        Returns:
            True if all downloads succeeded, False otherwise
        """
        missing = ModelManager.get_missing_models()

        try:
            for model_name in missing:
                if progress_callback:
                    progress_callback(model_name, 0.0, 'downloading')

                if model_name == 'parakeet':
                    ModelManager._download_parakeet()
                elif model_name == 'silero_vad':
                    ModelManager._download_silero()

                if progress_callback:
                    progress_callback(model_name, 1.0, 'complete')

            return True

        except Exception as e:
            print(f"Model download error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

            if progress_callback:
                progress_callback(model_name if 'model_name' in locals() else 'unknown', 0.0, 'error')

            # Cleanup partial files
            ModelManager._cleanup_partial_files()
            return False

    @staticmethod
    def validate_model(model_name: str) -> bool:
        """
        Validates that a model exists and has reasonable size.

        Args:
            model_name: Name of model to validate ('parakeet' or 'silero_vad')

        Returns:
            True if model file exists and is valid, False otherwise
        """
        if model_name == 'parakeet':
            # Check for encoder-model.onnx which is required by onnx_asr
            model_path = MODEL_DIR / "parakeet" / "encoder-model.onnx"
        elif model_name == 'silero_vad':
            # Check for silero_vad.onnx at the path expected by config
            model_path = MODEL_DIR / "silero_vad" / "silero_vad.onnx"
        else:
            return False

        if not model_path.exists():
            return False

        # Check file size is reasonable (not 0 bytes)
        if model_path.stat().st_size == 0:
            return False

        return True

    @staticmethod
    def _download_parakeet():
        """Downloads Parakeet STT model from HuggingFace."""
        parakeet_dir = MODEL_DIR / "parakeet"
        parakeet_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=PARAKEET_REPO,
            local_dir=str(parakeet_dir)
        )

    @staticmethod
    def _download_silero():
        """
        Downloads Silero VAD model from HuggingFace.

        HuggingFace downloads to onnx/model.onnx, but the application expects
        silero_vad.onnx at the root. This method copies the file to both locations.
        """
        silero_dir = MODEL_DIR / "silero_vad"
        silero_dir.mkdir(parents=True, exist_ok=True)

        # Download from HuggingFace (creates onnx/model.onnx)
        hf_hub_download(
            repo_id=SILERO_REPO,
            filename=SILERO_FILE,
            local_dir=str(silero_dir)
        )

        # Copy to the location expected by config (silero_vad.onnx)
        source_path = silero_dir / SILERO_FILE
        target_path = silero_dir / "silero_vad.onnx"

        if source_path.exists():
            shutil.copy2(source_path, target_path)

    @staticmethod
    def _cleanup_partial_files():
        """Removes partial download files after errors."""
        for model_dir in [MODEL_DIR / "parakeet", MODEL_DIR / "silero_vad"]:
            if model_dir.exists():
                for partial_file in model_dir.glob("*.partial"):
                    try:
                        partial_file.unlink()
                    except Exception:
                        pass  # Best effort cleanup
