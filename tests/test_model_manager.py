"""Tests for ModelManager.model_exists()."""

from pathlib import Path
from unittest.mock import patch

from src.asr.ModelManager import ModelManager


_MODELS_DIR = Path("/fake/models")


class TestModelExists:
    def test_returns_true_when_registry_validates_parakeet(self) -> None:
        with patch("src.asr.ModelManager.ModelRegistry.validate_model", return_value=True):
            assert ModelManager(_MODELS_DIR).model_exists() is True

    def test_returns_false_when_registry_rejects_parakeet(self) -> None:
        with patch("src.asr.ModelManager.ModelRegistry.validate_model", return_value=False):
            assert ModelManager(_MODELS_DIR).model_exists() is False

    def test_model_exists_delegates_to_registry_for_parakeet(self) -> None:
        with patch("src.asr.ModelManager.ModelRegistry.validate_model", return_value=True) as mock_validate:
            ModelManager(_MODELS_DIR).model_exists()
        mock_validate.assert_called_once_with("parakeet", _MODELS_DIR)
