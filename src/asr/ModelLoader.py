"""Loads the onnx_asr parakeet model with proper error handling."""

import json
import logging
from pathlib import Path
from typing import Any

import onnx_asr
import onnxruntime as rt
from onnx_asr.loader import (
    InvalidModelTypeInConfigError,
    ModelFileNotFoundError,
    ModelNotSupportedError,
    ModelPathNotDirectoryError,
    MoreThanOneModelFileFoundError,
)

logger = logging.getLogger(__name__)

_MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
_QUANTIZATION = "fp16"


class ModelLoadError(RuntimeError):
    """Raised when onnx_asr model loading fails with a user-facing message."""


def load_model(
    models_dir: Path,
    providers: list,
    sess_options: rt.SessionOptions,
) -> Any:
    """Load the parakeet ASR model with timestamps enabled.

    Args:
        models_dir: Root models directory; the parakeet model is expected at models_dir/parakeet.
        providers: ONNX Runtime execution provider list.
        sess_options: ONNX Runtime session options.

    Returns:
        Loaded model with timestamps enabled.

    Raises:
        ModelLoadError: On any onnx_asr or onnxruntime loading failure.
    """
    model_path = models_dir / "parakeet"
    try:
        base_model = onnx_asr.load_model(
            _MODEL_NAME,
            str(model_path),
            quantization=_QUANTIZATION,
            providers=providers,
            sess_options=sess_options,
        )
        model = base_model.with_timestamps()
    except ModelNotSupportedError as e:
        raise ModelLoadError(
            f"Model name '{_MODEL_NAME}' is not recognised by onnx_asr: {e}"
        ) from e
    except ModelPathNotDirectoryError as e:
        raise ModelLoadError(
            f"Model path is not a directory: {model_path}. Re-download the parakeet model. ({e})"
        ) from e
    except ModelFileNotFoundError as e:
        raise ModelLoadError(
            f"Expected model file not found in {model_path}. Re-download the parakeet model. ({e})"
        ) from e
    except MoreThanOneModelFileFoundError as e:
        raise ModelLoadError(
            f"Ambiguous model files — duplicate .onnx files found in {model_path}. ({e})"
        ) from e
    except InvalidModelTypeInConfigError as e:
        raise ModelLoadError(
            f"Model config.json has an unsupported model_type. Re-download the parakeet model. ({e})"
        ) from e
    except json.JSONDecodeError as e:
        raise ModelLoadError(
            f"Model config.json is corrupt or unreadable: {e}"
        ) from e
    except OSError as e:
        raise ModelLoadError(
            f"OS error loading model — check file permissions: {e}"
        ) from e
    except Exception as e:
        raise ModelLoadError(
            f"Unexpected error loading model ({type(e).__name__}): {e}"
        ) from e

    logger.info("Model loaded.")
    return model
