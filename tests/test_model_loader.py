"""Tests for src/asr/ModelLoader.py.

Tests cover:
- Successful load returns model with timestamps
- Each onnx_asr-specific exception is caught and re-raised as ModelLoadError
- json.JSONDecodeError, OSError, and unexpected exceptions are caught similarly
- Error messages contain identifying context (path, hint, exception type)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.asr.ModelLoader import ModelLoadError, load_model
from src.asr.ModelDefinitions import ParakeetAsrModel


_MODELS_DIR = Path("/fake/models")
_ASR_MODEL = ParakeetAsrModel(_MODELS_DIR)
_PROVIDERS = ["CPUExecutionProvider"]


def _make_sess_options() -> MagicMock:
    return MagicMock()


def _patch_load(side_effect=None, return_value=None):
    """Patch onnx_asr.load_model inside ModelLoader module."""
    if side_effect is not None:
        return patch("src.asr.ModelLoader.onnx_asr.load_model", side_effect=side_effect)
    return patch("src.asr.ModelLoader.onnx_asr.load_model", return_value=return_value)


class TestLoadModelSuccess:
    """load_model() returns model with timestamps on success."""

    def test_returns_model_with_timestamps(self) -> None:
        timestamped = MagicMock()
        base = MagicMock()
        base.with_timestamps.return_value = timestamped

        with _patch_load(return_value=base):
            result = load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())

        assert result is timestamped


class TestLoadModelNotSupportedError:
    """ModelNotSupportedError maps to ModelLoadError."""

    def _make_exc(self):
        from onnx_asr.loader import ModelNotSupportedError
        return ModelNotSupportedError("nemo-parakeet-tdt-0.6b-v3")

    def test_raises_model_load_error(self) -> None:
        with _patch_load(side_effect=self._make_exc()):
            with pytest.raises(ModelLoadError):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())

    def test_error_message_contains_model_name(self) -> None:
        with _patch_load(side_effect=self._make_exc()):
            with pytest.raises(ModelLoadError, match="nemo-parakeet"):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())


class TestLoadModelPathNotDirectoryError:
    """ModelPathNotDirectoryError maps to ModelLoadError."""

    def _make_exc(self):
        from onnx_asr.loader import ModelPathNotDirectoryError
        return ModelPathNotDirectoryError("/fake/models/parakeet")

    def test_raises_model_load_error(self) -> None:
        with _patch_load(side_effect=self._make_exc()):
            with pytest.raises(ModelLoadError):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())

    def test_error_message_contains_re_download_hint(self) -> None:
        with _patch_load(side_effect=self._make_exc()):
            with pytest.raises(ModelLoadError, match="(?i)re-download|download"):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())


class TestLoadModelFileNotFoundError:
    """ModelFileNotFoundError maps to ModelLoadError."""

    def _make_exc(self):
        from onnx_asr.loader import ModelFileNotFoundError
        return ModelFileNotFoundError("model_fp16.onnx", "/fake/models/parakeet")

    def test_raises_model_load_error(self) -> None:
        with _patch_load(side_effect=self._make_exc()):
            with pytest.raises(ModelLoadError):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())

    def test_error_message_contains_re_download_hint(self) -> None:
        with _patch_load(side_effect=self._make_exc()):
            with pytest.raises(ModelLoadError, match="(?i)re-download|download"):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())


class TestLoadModelMoreThanOneFileError:
    """MoreThanOneModelFileFoundError maps to ModelLoadError."""

    def _make_exc(self):
        from onnx_asr.loader import MoreThanOneModelFileFoundError
        return MoreThanOneModelFileFoundError("*.onnx", "/fake/models/parakeet")

    def test_raises_model_load_error(self) -> None:
        with _patch_load(side_effect=self._make_exc()):
            with pytest.raises(ModelLoadError):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())

    def test_error_message_contains_duplicate_hint(self) -> None:
        with _patch_load(side_effect=self._make_exc()):
            with pytest.raises(ModelLoadError, match="(?i)duplicate|ambiguous|more than one"):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())


class TestLoadModelInvalidModelTypeError:
    """InvalidModelTypeInConfigError maps to ModelLoadError."""

    def _make_exc(self):
        from onnx_asr.loader import InvalidModelTypeInConfigError
        return InvalidModelTypeInConfigError("unknown-type")

    def test_raises_model_load_error(self) -> None:
        with _patch_load(side_effect=self._make_exc()):
            with pytest.raises(ModelLoadError):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())


class TestLoadModelJsonDecodeError:
    """json.JSONDecodeError (corrupt config.json) maps to ModelLoadError."""

    def _make_exc(self):
        return json.JSONDecodeError("Expecting value", "doc", 0)

    def test_raises_model_load_error(self) -> None:
        with _patch_load(side_effect=self._make_exc()):
            with pytest.raises(ModelLoadError):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())

    def test_error_message_contains_config_hint(self) -> None:
        with _patch_load(side_effect=self._make_exc()):
            with pytest.raises(ModelLoadError, match="(?i)config"):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())


class TestLoadModelOsError:
    """OSError (disk/permissions) maps to ModelLoadError."""

    def test_raises_model_load_error(self) -> None:
        with _patch_load(side_effect=OSError("permission denied")):
            with pytest.raises(ModelLoadError):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())

    def test_error_message_contains_disk_hint(self) -> None:
        with _patch_load(side_effect=OSError("no space left")):
            with pytest.raises(ModelLoadError, match="(?i)disk|permission|OS error"):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())


class TestLoadModelUnexpectedException:
    """Any unexpected exception maps to ModelLoadError."""

    def test_raises_model_load_error(self) -> None:
        with _patch_load(side_effect=RuntimeError("onnxruntime internal")):
            with pytest.raises(ModelLoadError):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())

    def test_error_message_contains_exception_type(self) -> None:
        with _patch_load(side_effect=RuntimeError("onnxruntime internal")):
            with pytest.raises(ModelLoadError, match="RuntimeError"):
                load_model(_ASR_MODEL, _PROVIDERS, _make_sess_options())
