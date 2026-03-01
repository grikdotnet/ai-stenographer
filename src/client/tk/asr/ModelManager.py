"""
ModelManager handles model download and validation for STT pipeline.
"""
import logging
import os
import ssl
from pathlib import Path
from typing import List, Callable, Optional
from urllib.request import getproxies
import requests
from src.client.tk.asr.DownloadProgressReporter import ProgressAggregator


MODEL_DIR = Path("./models")
PARAKEET_CDN_URL = "https://parakeet.grik.net/"
PARAKEET_FILES = [
    "config.json",
    "decoder_joint-model.fp16.onnx",
    "encoder-model.fp16.onnx",
    "nemo128.onnx",
    "vocab.txt"
]
# Silero VAD model is bundled with distribution (no download needed)


class ModelManager:
    """
    Manages model downloads and validation for STT pipeline.

    Supports both development mode (./models/) and distribution mode
    (_internal/models/) by accepting model_dir parameter.
    """

    _last_error: Optional[str] = None  # Store detailed error message for GUI display

    @staticmethod
    def _configure_network_environment():
        """
        Configure network settings for MSIX sandbox and corporate networks.

        Logs network configuration details and sets environment variables
        to handle proxies and SSL in restricted environments.
        """
        # Detect and log Windows proxy settings
        proxies = getproxies()
        if proxies:
            logging.info(f"System proxy detected: {proxies}")
            os.environ['HTTP_PROXY'] = proxies.get('http', '')
            os.environ['HTTPS_PROXY'] = proxies.get('https', '')
        else:
            logging.info("No system proxy detected")

    @staticmethod
    def get_last_error() -> Optional[str]:
        """Returns the last error message from download_models()"""
        return ModelManager._last_error

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

        # Silero VAD is bundled with distribution - never missing

        return missing

    @staticmethod
    def download_models(model_dir: Optional[Path] = None, progress_callback: Optional[Callable[[str, float, str, int, int], None]] = None) -> bool:
        """
        Downloads FP16 Parakeet models (1.25GB) from Cloudflare R2 CDN.

        Note: Silero VAD model is bundled with distribution and does not need downloading.

        Args:
            model_dir: Directory to download models to (defaults to ./models)
            progress_callback: Callback function with signature:
                def callback(model_name: str, progress: float, status: str,
                           downloaded_bytes: int, total_bytes: int):
                    # model_name: 'parakeet'
                    # progress: 0.0 to 1.0
                    # status: 'downloading' | 'complete' | 'error'
                    # downloaded_bytes: cumulative bytes downloaded
                    # total_bytes: total bytes to download

        Returns:
            True if all downloads succeeded, False otherwise
        """
        # Configure network environment for MSIX/corporate networks
        ModelManager._configure_network_environment()

        if model_dir is None:
            model_dir = MODEL_DIR

        # Ensure models directory exists and is writable
        model_dir_path = Path(model_dir)
        model_dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Models directory: {model_dir_path.resolve()}")
        logging.info(f"Directory writable: {os.access(model_dir_path, os.W_OK)}")

        missing = ModelManager.get_missing_models(model_dir)

        try:
            for model_name in missing:
                if progress_callback:
                    progress_callback(model_name, 0.0, 'downloading', 0, 0)

                if model_name == 'parakeet':
                    ModelManager._download_parakeet(model_dir, progress_callback)
                # Silero VAD is bundled - no download branch needed

                if progress_callback:
                    progress_callback(model_name, 1.0, 'complete', 0, 0)

            return True

        except Exception as e:
            # Detailed error logging with specific exception types
            import traceback

            error_type = type(e).__name__
            error_details = str(e)

            # Check if this is a requests-related error (from huggingface_hub internals)
            if requests and hasattr(e, '__class__'):
                if 'SSLError' in error_type or 'CertificateError' in error_type:
                    error_msg = f"SSL certificate validation failed: {error_details}"
                    ModelManager._last_error = error_msg
                    logging.error(error_msg)
                    logging.error(f"Certificate store: {ssl.get_default_verify_paths()}")
                    logging.error("Possible causes: Corporate proxy, custom CA, MSIX sandbox restrictions")
                elif 'ProxyError' in error_type:
                    error_msg = f"Proxy connection failed: {error_details}"
                    ModelManager._last_error = error_msg
                    logging.error(error_msg)
                    logging.error(f"System proxies: {getproxies()}")
                elif 'ConnectionError' in error_type or 'ConnectTimeout' in error_type:
                    error_msg = f"Network connection failed: {error_details}"
                    ModelManager._last_error = error_msg
                    logging.error(error_msg)
                    logging.error("Check internetClient capability and network access in MSIX sandbox")
                elif 'Timeout' in error_type or 'ReadTimeout' in error_type:
                    error_msg = f"Download timeout (network too slow): {error_details}"
                    ModelManager._last_error = error_msg
                    logging.error(error_msg)
                else:
                    # Generic error
                    error_msg = f"Model download error: {error_type}: {error_details}"
                    ModelManager._last_error = error_msg
                    logging.error(error_msg)
            else:
                # Generic error
                error_msg = f"Model download error: {error_type}: {error_details}"
                ModelManager._last_error = error_msg
                logging.error(error_msg)

            logging.error(traceback.format_exc())

            if progress_callback:
                progress_callback(model_name if 'model_name' in locals() else 'unknown', 0.0, 'error', 0, 0)

            ModelManager._cleanup_partial_files(model_dir)
            return False

    @staticmethod
    def validate_model(model_name: str, model_dir: Optional[Path] = None) -> bool:
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

    @staticmethod
    def _download_parakeet(
        model_dir: Path,
        progress_callback: Optional[Callable[[str, float, str, int, int], None]] = None
    ):
        """
        Downloads FP16 Parakeet STT model from Cloudflare R2 CDN.

        Args:
            model_dir: Directory to download model to
            progress_callback: Optional callback with signature:
                callback(model_name, progress, status, downloaded_bytes, total_bytes)
        """
        parakeet_dir = model_dir / "parakeet"
        parakeet_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Downloading FP16 Parakeet models from CDN (1.25GB)...")

        # Create progress tracking infrastructure if callback provided
        aggregator = ProgressAggregator() if progress_callback else None

        for filename in PARAKEET_FILES:
            url = PARAKEET_CDN_URL + filename
            target_path = parakeet_dir / filename
            file_id = f"parakeet_{filename}"

            logging.info(f"Downloading {filename} from {url}")

            try:
                # Use streaming download with progress tracking
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()

                # Get file size from Content-Length header
                file_size = int(response.headers.get('Content-Length', 0))

                # Download file in chunks
                chunk_size = 8192
                bytes_downloaded_for_file = 0

                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bytes_downloaded_for_file += len(chunk)

                            # Update progress callback
                            if progress_callback and aggregator:
                                # Update this file's progress
                                aggregator.update_file_progress(file_id, bytes_downloaded_for_file, file_size)

                                # Get overall progress across all files
                                overall_downloaded, overall_total, overall_percentage = aggregator.get_overall_progress()

                                # Call progress callback
                                progress_callback('parakeet', overall_percentage, 'downloading', overall_downloaded, overall_total)

                # Validate downloaded file
                if target_path.stat().st_size == 0:
                    raise RuntimeError(f"Downloaded file {filename} is empty")

                logging.info(f"Downloaded {filename} ({file_size} bytes)")

            except requests.exceptions.HTTPError as e:
                # Clean up partial download
                if target_path.exists():
                    target_path.unlink()
                raise RuntimeError(f"HTTP error downloading {filename}: {e.response.status_code} {e.response.reason}")
            except requests.exceptions.Timeout:
                if target_path.exists():
                    target_path.unlink()
                raise RuntimeError(f"Download timeout for {filename} (network too slow)")
            except requests.exceptions.ConnectionError as e:
                if target_path.exists():
                    target_path.unlink()
                raise RuntimeError(f"Connection error downloading {filename}: {str(e)}")
            except Exception as e:
                if target_path.exists():
                    target_path.unlink()
                raise RuntimeError(f"Error downloading {filename}: {str(e)}")

        logging.info("FP16 models downloaded successfully from CDN")


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
