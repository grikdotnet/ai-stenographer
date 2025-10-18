"""Tests for STTPipeline with execution providers."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import STTPipeline
from src.ExecutionProviderManager import ExecutionProviderManager


class TestPipelineExecutionProviders:
    """Tests for STTPipeline with execution providers."""

    @patch('src.pipeline.onnx_asr.load_model')
    @patch('src.pipeline.create_stt_window')
    @patch('src.pipeline.AudioSource')
    @patch('src.pipeline.VoiceActivityDetector')
    def test_pipeline_initializes_with_auto_provider(self, mock_vad, mock_audio, mock_gui, mock_load_model):
        """Pipeline should initialize successfully with auto provider mode."""
        # Mock GUI window creation
        mock_root = Mock()
        mock_text_widget = Mock()
        mock_gui.return_value = (mock_root, mock_text_widget)

        # Mock model loading
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Create pipeline with auto provider
        with patch.object(ExecutionProviderManager, 'detect_available_providers') as mock_detect:
            mock_detect.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

            pipeline = STTPipeline(verbose=False)

        # Verify ExecutionProviderManager was created
        assert pipeline.execution_provider_manager is not None
        assert isinstance(pipeline.execution_provider_manager, ExecutionProviderManager)

        # Verify model was loaded with providers
        assert mock_load_model.called
        call_args = mock_load_model.call_args
        assert 'providers' in call_args[1]

    @patch('src.pipeline.onnx_asr.load_model')
    @patch('src.pipeline.create_stt_window')
    @patch('src.pipeline.AudioSource')
    @patch('src.pipeline.VoiceActivityDetector')
    def test_pipeline_uses_explicit_directml(self, mock_vad, mock_audio, mock_gui, mock_load_model):
        """Pipeline should use DirectML when explicitly configured."""
        # Mock GUI window creation
        mock_root = Mock()
        mock_text_widget = Mock()
        mock_gui.return_value = (mock_root, mock_text_widget)

        # Mock model loading
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Create config file with DirectML
        import tempfile
        import json

        config = {
            "audio": {"sample_rate": 16000, "chunk_duration": 0.032},
            "vad": {"frame_duration_ms": 32, "threshold": 0.5},
            "windowing": {"window_duration": 3.0, "step_size": 1.0},
            "recognition": {"model_name": "parakeet", "inference": "directml"}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            json.dump(config, f)

        try:
            with patch.object(ExecutionProviderManager, 'detect_available_providers') as mock_detect:
                mock_detect.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

                pipeline = STTPipeline(config_path=config_path, verbose=False)

            # Verify DirectML was selected
            assert pipeline.execution_provider_manager.selected_provider == 'DirectML'

            # Verify model was loaded with DirectML providers
            call_args = mock_load_model.call_args
            providers = call_args[1]['providers']
            assert len(providers) == 2
            assert providers[0] == ('DmlExecutionProvider', {'device_id': 0})
        finally:
            Path(config_path).unlink()

    @patch('src.pipeline.onnx_asr.load_model')
    @patch('src.pipeline.create_stt_window')
    @patch('src.pipeline.AudioSource')
    @patch('src.pipeline.VoiceActivityDetector')
    def test_pipeline_config_override(self, mock_vad, mock_audio, mock_gui, mock_load_model):
        """Config should properly propagate to all components."""
        # Mock GUI window creation
        mock_root = Mock()
        mock_text_widget = Mock()
        mock_gui.return_value = (mock_root, mock_text_widget)

        # Mock model loading
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Create config file with CPU
        import tempfile
        import json

        config = {
            "audio": {"sample_rate": 16000, "chunk_duration": 0.032},
            "vad": {"frame_duration_ms": 32, "threshold": 0.5},
            "windowing": {"window_duration": 3.0, "step_size": 1.0},
            "recognition": {"model_name": "parakeet", "inference": "cpu"}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            json.dump(config, f)

        try:
            pipeline = STTPipeline(config_path=config_path, verbose=False)

            # Verify CPU was selected
            assert pipeline.execution_provider_manager.selected_provider == 'CPU'

            # Verify model was loaded with CPU providers
            call_args = mock_load_model.call_args
            providers = call_args[1]['providers']
            assert providers == ['CPUExecutionProvider']
        finally:
            Path(config_path).unlink()

    @patch('src.pipeline.onnx_asr.load_model')
    @patch('src.pipeline.create_stt_window')
    @patch('src.pipeline.AudioSource')
    @patch('src.pipeline.VoiceActivityDetector')
    def test_pipeline_provider_info_logging(self, mock_vad, mock_audio, mock_gui, mock_load_model, caplog):
        """Pipeline should log selected provider on startup."""
        # Mock GUI window creation
        mock_root = Mock()
        mock_text_widget = Mock()
        mock_gui.return_value = (mock_root, mock_text_widget)

        # Mock model loading
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Create pipeline with logging
        with caplog.at_level(logging.INFO):
            with patch.object(ExecutionProviderManager, 'detect_available_providers') as mock_detect:
                mock_detect.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

                pipeline = STTPipeline(verbose=False)

        # Verify log message
        assert any("Using execution provider" in record.message for record in caplog.records)
        # The message should contain provider info like "DirectML" or "CPU"
        provider_logs = [r.message for r in caplog.records if "Using execution provider" in r.message]
        assert len(provider_logs) > 0
        assert any("DirectML" in log or "CPU" in log for log in provider_logs)
