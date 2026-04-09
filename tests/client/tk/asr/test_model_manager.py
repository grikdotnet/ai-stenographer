"""
Tests for GUI ModelManager delegation to shared downloader components.
"""
from pathlib import Path
from unittest.mock import patch

from src.client.tk.asr.ModelManager import ModelManager


_MODELS_DIR = Path("/fake/models")


class TestModelManager:
    """Test suite for ModelManager functionality."""

    def test_get_missing_models_delegates_to_registry(self) -> None:
        with patch(
            'src.client.tk.asr.ModelManager.ModelRegistry.get_missing_models',
            return_value=['parakeet'],
        ) as mock_registry:
            assert ModelManager.get_missing_models(_MODELS_DIR) == ['parakeet']
        mock_registry.assert_called_once_with(_MODELS_DIR)

    def test_download_models_uses_shared_downloader(self) -> None:
        callback_calls = []

        def progress_callback(
            model_name: str,
            progress: float,
            status: str,
            downloaded_bytes: int,
            total_bytes: int,
        ) -> None:
            callback_calls.append((model_name, progress, status, downloaded_bytes, total_bytes))

        with (
            patch(
                'src.client.tk.asr.ModelManager.ModelRegistry.get_missing_models',
                return_value=['parakeet'],
            ),
            patch('src.client.tk.asr.ModelManager.ModelDownloader') as downloader_cls,
        ):
            downloader = downloader_cls.return_value

            def run_download(callback) -> None:
                callback(0.5, 50, 100)
                callback(1.0, 100, 100)

            downloader.download_parakeet.side_effect = run_download

            result = ModelManager.download_models(
                model_dir=_MODELS_DIR,
                progress_callback=progress_callback,
            )

        assert result is True
        downloader_cls.assert_called_once_with(_MODELS_DIR)
        downloader.download_parakeet.assert_called_once()
        assert ('parakeet', 0.0, 'downloading', 0, 0) in callback_calls
        assert ('parakeet', 0.5, 'downloading', 50, 100) in callback_calls
        assert ('parakeet', 1.0, 'complete', 1, 1) in callback_calls

    def test_download_failure_cleanup(self) -> None:
        with (
            patch(
                'src.client.tk.asr.ModelManager.ModelRegistry.get_missing_models',
                return_value=['parakeet'],
            ),
            patch('src.client.tk.asr.ModelManager.ModelDownloader') as downloader_cls,
        ):
            downloader = downloader_cls.return_value
            downloader.download_parakeet.side_effect = RuntimeError("Network error")

            result = ModelManager.download_models(model_dir=_MODELS_DIR)

        assert result is False
        downloader_cls.cleanup_partial_files.assert_called_once_with(_MODELS_DIR)

    def test_validate_model_delegates_to_registry(self) -> None:
        with patch(
            'src.client.tk.asr.ModelManager.ModelRegistry.validate_model',
            return_value=True,
        ) as mock_validate:
            assert ModelManager.validate_model('parakeet', _MODELS_DIR) is True
        mock_validate.assert_called_once_with('parakeet', _MODELS_DIR)
