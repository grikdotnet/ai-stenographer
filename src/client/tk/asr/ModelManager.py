"""
ModelManager handles model download and validation for STT pipeline.
"""
import logging
import ssl
from pathlib import Path
from typing import Callable, List, Optional

import requests

from src.downloader.ModelDownloader import ModelDownloader
from src.asr.ModelRegistry import ModelRegistry


MODEL_DIR = Path("./models")


class ModelManager:
    """
    Manages model downloads and validation for STT pipeline.

    Supports both development mode (./models/) and distribution mode
    (_internal/models/) by accepting model_dir parameter.
    """

    _last_error: Optional[str] = None

    @staticmethod
    def get_last_error() -> Optional[str]:
        """Return the last error message from download_models()."""
        return ModelManager._last_error

    @staticmethod
    def get_missing_models(model_dir: Optional[Path] = None) -> List[str]:
        """
        Returns list of missing model names.

        Args:
            model_dir: Directory containing models (defaults to ./models)

        Returns:
            List of missing model names.
        """
        return ModelRegistry.get_missing_models_for_dir(model_dir or MODEL_DIR)

    @staticmethod
    def download_models(
        model_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float, str, int, int], None]] = None,
    ) -> bool:
        """
        Download missing shared models for the GUI flow.

        Args:
            model_dir: Directory to download models to.
            progress_callback: GUI callback receiving model status/progress updates.

        Returns:
            True if all required downloads succeeded, False otherwise.
        """
        model_dir_path = Path(model_dir or MODEL_DIR)
        model_dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Models directory: {model_dir_path.resolve()}")

        missing = ModelManager.get_missing_models(model_dir_path)
        downloader = ModelDownloader(model_dir_path)

        try:
            for model_name in missing:
                if progress_callback:
                    progress_callback(model_name, 0.0, 'downloading', 0, 0)

                if model_name == 'parakeet':
                    downloader.download_parakeet(
                        ModelManager._adapt_progress_callback(progress_callback)
                        if progress_callback else None,
                    )

                if progress_callback:
                    progress_callback(model_name, 1.0, 'complete', 1, 1)

            return True
        except Exception as e:
            import traceback

            error_type = type(e).__name__
            error_details = str(e)

            if requests and hasattr(e, '__class__'):
                if 'SSLError' in error_type or 'CertificateError' in error_type:
                    error_msg = f"SSL certificate validation failed: {error_details}"
                    ModelManager._last_error = error_msg
                    logging.error(error_msg)
                    logging.error(f"Certificate store: {ssl.get_default_verify_paths()}")
                    logging.error(
                        "Possible causes: Corporate proxy, custom CA, MSIX sandbox restrictions"
                    )
                elif 'ProxyError' in error_type:
                    error_msg = f"Proxy connection failed: {error_details}"
                    ModelManager._last_error = error_msg
                    logging.error(error_msg)
                elif 'ConnectionError' in error_type or 'ConnectTimeout' in error_type:
                    error_msg = f"Network connection failed: {error_details}"
                    ModelManager._last_error = error_msg
                    logging.error(error_msg)
                    logging.error(
                        "Check internetClient capability and network access in MSIX sandbox"
                    )
                elif 'Timeout' in error_type or 'ReadTimeout' in error_type:
                    error_msg = f"Download timeout (network too slow): {error_details}"
                    ModelManager._last_error = error_msg
                    logging.error(error_msg)
                else:
                    error_msg = f"Model download error: {error_type}: {error_details}"
                    ModelManager._last_error = error_msg
                    logging.error(error_msg)
            else:
                error_msg = f"Model download error: {error_type}: {error_details}"
                ModelManager._last_error = error_msg
                logging.error(error_msg)

            logging.error(traceback.format_exc())

            if progress_callback:
                progress_callback(
                    model_name if 'model_name' in locals() else 'unknown',
                    0.0,
                    'error',
                    0,
                    0,
                )

            ModelManager._cleanup_partial_files(model_dir_path)
            return False

    @staticmethod
    def validate_model(model_name: str, model_dir: Optional[Path] = None) -> bool:
        """Return True when the requested model is available and valid."""
        root = model_dir or MODEL_DIR
        return ModelRegistry.validate_model(model_name, root)

    @staticmethod
    def _adapt_progress_callback(
        progress_callback: Callable[[str, float, str, int, int], None],
    ) -> Callable[[float, int, int], None]:
        """Adapt shared downloader progress to the GUI callback signature.

        Args:
            progress_callback: GUI callback expecting model metadata.

        Returns:
            Callback accepted by the shared downloader.
        """
        def callback(progress: float, downloaded_bytes: int, total_bytes: int) -> None:
            progress_callback(
                'parakeet',
                progress,
                'downloading',
                downloaded_bytes,
                total_bytes,
            )

        return callback

    @staticmethod
    def _cleanup_partial_files(model_dir: Path) -> None:
        """
        Removes partial download files after errors.

        Args:
            model_dir: Root models directory
        """
        ModelDownloader.cleanup_partial_files(model_dir)
