"""
Tests for ModelManager - handles model download and validation.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from src.asr.ModelManager import ModelManager


class TestModelManager:
    """Test suite for ModelManager functionality."""

    def test_get_missing_models_all_present(self, tmp_path):
        """Returns empty list when all models are present."""
        # Create mock model directories
        parakeet_dir = tmp_path / "parakeet"
        parakeet_dir.mkdir()
        (parakeet_dir / "encoder-model.fp16.onnx").write_text("mock")

        silero_dir = tmp_path / "silero_vad"
        silero_dir.mkdir()
        (silero_dir / "silero_vad.onnx").write_text("mock")

        with patch('src.asr.ModelManager.MODEL_DIR', tmp_path):
            missing = ModelManager.get_missing_models()
            assert missing == []

    def test_parakeet_missing(self, tmp_path):
        with patch('src.asr.ModelManager.MODEL_DIR', tmp_path):
            missing = ModelManager.get_missing_models()
            # Only Parakeet should be missing (Silero is bundled, never missing)
            assert missing == ['parakeet']

    @patch('src.asr.ModelManager.requests.get')
    def test_download_parakeet_success(self, mock_get, tmp_path):
        """Successfully downloads Parakeet model from CDN."""
        # Setup mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Length': '1024'}
        mock_response.iter_content = lambda chunk_size: [b'x' * 1024]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch('src.asr.ModelManager.MODEL_DIR', tmp_path):
            result = ModelManager.download_models()

            assert result is True
            # Should download all 5 Parakeet files
            assert mock_get.call_count == 5
            # Verify files were created
            assert (tmp_path / "parakeet" / "config.json").exists()
            assert (tmp_path / "parakeet" / "encoder-model.fp16.onnx").exists()

    @patch('src.asr.ModelManager.requests.get')
    def test_download_with_progress_callback(self, mock_get, tmp_path):
        """Verifies progress callback receives 5-parameter updates during download."""
        callback_calls = []

        def progress_callback(model_name: str, progress: float, status: str,
                            downloaded_bytes: int, total_bytes: int):
            callback_calls.append((model_name, progress, status, downloaded_bytes, total_bytes))

        # Setup mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Length': '1024'}
        mock_response.iter_content = lambda chunk_size: [b'x' * 1024]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch('src.asr.ModelManager.MODEL_DIR', tmp_path):
            ModelManager.download_models(progress_callback=progress_callback)

            # Check callback was called with 5 parameters
            assert len(callback_calls) > 0
            for call in callback_calls:
                assert len(call) == 5  # 5 parameters
                model_name, progress, status, downloaded_bytes, total_bytes = call
                assert isinstance(model_name, str)
                assert model_name == 'parakeet'
                assert isinstance(progress, float)
                assert isinstance(status, str)
                assert isinstance(downloaded_bytes, int)
                assert isinstance(total_bytes, int)

            # Check for 'complete' status
            complete_calls = [c for c in callback_calls if c[2] == 'complete']
            assert len(complete_calls) > 0

    @patch('src.asr.ModelManager.requests.get')
    def test_download_http_404_error(self, mock_get, tmp_path):
        """Handles HTTP 404 errors correctly."""
        # Setup mock to raise HTTP 404 error
        import requests
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        with patch('src.asr.ModelManager.MODEL_DIR', tmp_path):
            result = ModelManager.download_models()

            assert result is False
            # Should have attempted download
            assert mock_get.call_count > 0

    @patch('src.asr.ModelManager.requests.get')
    def test_download_timeout_error(self, mock_get, tmp_path):
        """Handles timeout errors correctly."""
        # Setup mock to raise timeout error
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")

        with patch('src.asr.ModelManager.MODEL_DIR', tmp_path):
            result = ModelManager.download_models()

            assert result is False

    @patch('src.asr.ModelManager.requests.get')
    def test_download_connection_error(self, mock_get, tmp_path):
        """Handles connection errors correctly."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Network unreachable")

        with patch('src.asr.ModelManager.MODEL_DIR', tmp_path):
            result = ModelManager.download_models()

            assert result is False

    @patch('src.asr.ModelManager.requests.get')
    def test_download_failure_cleanup(self, mock_get, tmp_path):
        """Verifies partial files are removed on download error."""
        with patch('src.asr.ModelManager.MODEL_DIR', tmp_path):
            # Create partial file
            parakeet_dir = tmp_path / "parakeet"
            parakeet_dir.mkdir()
            partial_file = parakeet_dir / "model.onnx.partial"
            partial_file.write_text("partial")

            # Mock download failure
            import requests
            mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

            result = ModelManager.download_models()

            assert result is False
            # Partial file should be cleaned up
            assert not partial_file.exists()

    def test_validate_model_missing_file(self, tmp_path):
        """Returns False when model file doesn't exist."""
        with patch('src.asr.ModelManager.MODEL_DIR', tmp_path):
            assert ModelManager.validate_model('parakeet') is False
            assert ModelManager.validate_model('silero_vad') is False

    def test_validate_silero_bundled(self, tmp_path):
        """Validates bundled Silero VAD model correctly."""
        silero_dir = tmp_path / "silero_vad"
        silero_dir.mkdir()
        (silero_dir / "silero_vad.onnx").write_text("bundled model")

        with patch('src.asr.ModelManager.MODEL_DIR', tmp_path):
            assert ModelManager.validate_model('silero_vad') is True

    @patch('src.asr.ModelManager.requests.get')
    def test_cdn_url_format(self, mock_get, tmp_path):
        """Verifies CDN URLs are correctly formatted."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Length': '100'}
        mock_response.iter_content = lambda chunk_size: [b'x' * 100]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with patch('src.asr.ModelManager.MODEL_DIR', tmp_path):
            ModelManager._download_parakeet(tmp_path)
            # Check that all URLs start with the CDN base URL
            for call_args in mock_get.call_args_list:
                url = call_args[0][0]
                assert url.startswith("https://parakeet.grik.net/"), f"Invalid URL: {url}"
