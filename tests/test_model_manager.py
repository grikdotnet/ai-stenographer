"""
Tests for ModelManager - handles model download and validation.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from src.ModelManager import ModelManager


class TestModelManager:
    """Test suite for ModelManager functionality."""

    def test_get_missing_models_all_present(self, tmp_path):
        """Returns empty list when all models are present."""
        # Create mock model directories
        parakeet_dir = tmp_path / "parakeet"
        parakeet_dir.mkdir()
        (parakeet_dir / "encoder-model.onnx").write_text("mock")

        silero_dir = tmp_path / "silero_vad"
        silero_dir.mkdir()
        (silero_dir / "silero_vad.onnx").write_text("mock")

        with patch('src.ModelManager.MODEL_DIR', tmp_path):
            missing = ModelManager.get_missing_models()
            assert missing == []

    def test_get_missing_models_both_missing(self, tmp_path):
        """Returns both model names when both are missing."""
        with patch('src.ModelManager.MODEL_DIR', tmp_path):
            missing = ModelManager.get_missing_models()
            assert set(missing) == {'parakeet', 'silero_vad'}

    def test_get_missing_models_partial(self, tmp_path):
        """Returns only missing model when one is present."""
        # Create only Silero VAD
        silero_dir = tmp_path / "silero_vad"
        silero_dir.mkdir()
        (silero_dir / "silero_vad.onnx").write_text("mock")

        with patch('src.ModelManager.MODEL_DIR', tmp_path):
            missing = ModelManager.get_missing_models()
            assert missing == ['parakeet']

    @patch('src.ModelManager.hf_hub_download')
    @patch('src.ModelManager.snapshot_download')
    def test_download_parakeet_success(self, mock_snapshot, mock_hf, tmp_path):
        """Successfully downloads Parakeet model."""
        # Setup mock to create files when called
        def create_parakeet_files(*args, **kwargs):
            local_dir = Path(kwargs.get('local_dir', args[1] if len(args) > 1 else ''))
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "encoder-model.onnx").write_text("downloaded")
            return str(local_dir)

        def create_silero_files(*args, **kwargs):
            local_dir = Path(kwargs.get('local_dir', ''))
            onnx_dir = local_dir / "onnx"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            model_file = onnx_dir / "model.onnx"
            model_file.write_text("downloaded")
            # Copy to expected location
            silero_file = local_dir / "silero_vad.onnx"
            silero_file.write_text("downloaded")
            return str(model_file)

        mock_snapshot.side_effect = create_parakeet_files
        mock_hf.side_effect = create_silero_files

        with patch('src.ModelManager.MODEL_DIR', tmp_path):
            result = ModelManager.download_models()

            assert result is True
            mock_snapshot.assert_called_once()
            assert (tmp_path / "parakeet" / "encoder-model.onnx").exists()

    @patch('src.ModelManager.snapshot_download')
    @patch('src.ModelManager.hf_hub_download')
    def test_download_silero_success(self, mock_download, mock_snapshot, tmp_path):
        """Successfully downloads Silero VAD model."""
        # Setup mock to create files when called
        def create_parakeet_files(*args, **kwargs):
            local_dir = Path(kwargs.get('local_dir', ''))
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "encoder-model.onnx").write_text("downloaded")
            return str(local_dir)

        def create_silero_files(*args, **kwargs):
            local_dir = Path(kwargs.get('local_dir', ''))
            onnx_dir = local_dir / "onnx"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            model_file = onnx_dir / "model.onnx"
            model_file.write_text("downloaded")
            # Copy to expected location
            silero_file = local_dir / "silero_vad.onnx"
            silero_file.write_text("downloaded")
            return str(model_file)

        mock_snapshot.side_effect = create_parakeet_files
        mock_download.side_effect = create_silero_files

        with patch('src.ModelManager.MODEL_DIR', tmp_path):
            result = ModelManager.download_models()

            assert result is True
            mock_download.assert_called_once()
            assert (tmp_path / "silero_vad" / "silero_vad.onnx").exists()

    @patch('src.ModelManager.snapshot_download')
    @patch('src.ModelManager.hf_hub_download')
    def test_download_with_progress_callback(self, mock_hf, mock_snapshot, tmp_path):
        """Verifies progress callback receives updates during download."""
        callback_calls = []

        def progress_callback(model_name: str, progress: float, status: str):
            callback_calls.append((model_name, progress, status))

        # Setup mocks to create files
        def create_parakeet(*args, **kwargs):
            local_dir = Path(kwargs.get('local_dir', ''))
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "encoder-model.onnx").write_text("mock")
            return str(local_dir)

        def create_silero(*args, **kwargs):
            local_dir = Path(kwargs.get('local_dir', ''))
            onnx_dir = local_dir / "onnx"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            model_file = onnx_dir / "model.onnx"
            model_file.write_text("mock")
            # Copy to expected location
            silero_file = local_dir / "silero_vad.onnx"
            silero_file.write_text("mock")
            return str(model_file)

        mock_snapshot.side_effect = create_parakeet
        mock_hf.side_effect = create_silero

        with patch('src.ModelManager.MODEL_DIR', tmp_path):
            ModelManager.download_models(progress_callback=progress_callback)

            # Check callback was called with expected parameters
            assert len(callback_calls) > 0
            # Check for 'complete' status
            complete_calls = [c for c in callback_calls if c[2] == 'complete']
            assert len(complete_calls) > 0

    @patch('src.ModelManager.snapshot_download')
    def test_download_failure_cleanup(self, mock_snapshot, tmp_path):
        """Verifies partial files are removed on download error."""
        with patch('src.ModelManager.MODEL_DIR', tmp_path):
            # Create partial file
            parakeet_dir = tmp_path / "parakeet"
            parakeet_dir.mkdir()
            partial_file = parakeet_dir / "model.onnx.partial"
            partial_file.write_text("partial")

            # Mock download failure
            mock_snapshot.side_effect = Exception("Network error")

            result = ModelManager.download_models()

            assert result is False
            # Partial file should be cleaned up
            assert not partial_file.exists()

    def test_validate_model_missing_file(self, tmp_path):
        """Returns False when model file doesn't exist."""
        with patch('src.ModelManager.MODEL_DIR', tmp_path):
            assert ModelManager.validate_model('parakeet') is False
            assert ModelManager.validate_model('silero_vad') is False

    @patch('src.ModelManager.hf_hub_download')
    def test_silero_download_creates_correct_path(self, mock_download, tmp_path):
        """Verifies Silero VAD is downloaded to the path expected by config."""
        # Mock download to simulate HuggingFace behavior
        def create_silero(*args, **kwargs):
            local_dir = Path(kwargs.get('local_dir', ''))
            # HuggingFace creates subdirectory matching filename structure
            onnx_dir = local_dir / "onnx"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            model_file = onnx_dir / "model.onnx"
            model_file.write_text("mock silero model")
            return str(model_file)

        mock_download.side_effect = create_silero

        with patch('src.ModelManager.MODEL_DIR', tmp_path):
            ModelManager._download_silero()

            # After download, file should be at location expected by config
            expected_path = tmp_path / "silero_vad" / "silero_vad.onnx"
            assert expected_path.exists(), f"Expected model at {expected_path}"

            # Original HuggingFace download location should also exist
            hf_path = tmp_path / "silero_vad" / "onnx" / "model.onnx"
            assert hf_path.exists(), f"HuggingFace download at {hf_path}"
