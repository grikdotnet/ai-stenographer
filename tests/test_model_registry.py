"""Tests for shared downloader registry and downloader facade."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.downloader.ModelDownloader import ModelDownloader
from src.downloader.ModelRegistry import ModelRegistry


_MODELS_DIR = Path("/fake/models")


class TestModelRegistry:
    def test_get_missing_models_returns_parakeet_when_validation_fails(self) -> None:
        with patch("src.downloader.ModelRegistry.validate_parakeet", return_value=False):
            assert ModelRegistry.get_missing_models(_MODELS_DIR) == ["parakeet"]

    def test_validate_model_returns_true_for_valid_silero(self) -> None:
        model_dir = MagicMock()
        silero_dir = MagicMock()
        model_path = MagicMock()
        model_dir.__truediv__.return_value = silero_dir
        silero_dir.__truediv__.return_value = model_path
        model_path.exists.return_value = True
        model_path.stat.return_value.st_size = 1

        assert ModelRegistry.validate_model("silero_vad", model_dir) is True


class TestModelDownloaderFacade:
    def test_validate_parakeet_delegates_to_module_function(self) -> None:
        downloader = ModelDownloader(_MODELS_DIR)
        with patch("src.downloader.ModelDownloader.validate_parakeet", return_value=True) as mock_validate:
            assert downloader.validate_parakeet() is True
        mock_validate.assert_called_once_with(_MODELS_DIR)

    def test_cleanup_partial_files_delegates_to_module_function(self) -> None:
        with patch("src.downloader.ModelDownloader.cleanup_partial_files") as mock_cleanup:
            ModelDownloader.cleanup_partial_files(_MODELS_DIR)
        mock_cleanup.assert_called_once_with(_MODELS_DIR)
