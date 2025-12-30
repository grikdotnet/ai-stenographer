"""Tests for STTPipeline integration with SessionOptionsStrategy pattern."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import STTPipeline
from src.ExecutionProviderManager import ExecutionProviderManager
from src.SessionOptionsStrategy import IntegratedGPUStrategy, DiscreteGPUStrategy, CPUStrategy


class TestPipelineSessionOptionsStrategy:
    """Tests for pipeline integration with SessionOptionsStrategy pattern."""

    @patch('src.pipeline.onnx_asr.load_model')
    @patch('src.gui.ApplicationWindow.ApplicationWindow')
    @patch('src.pipeline.AudioSource')
    @patch('src.pipeline.VoiceActivityDetector')
    @patch('src.ExecutionProviderManager.ExecutionProviderManager._enumerate_adapters_dxgi')
    def test_pipeline_uses_integrated_gpu_strategy(self, mock_dxgi, mock_vad, mock_audio, mock_app_window, mock_load_model):
        """Pipeline should use IntegratedGPUStrategy for integrated GPU."""
        # Mock ApplicationWindow
        mock_root = Mock()
        mock_gui_window = Mock()
        mock_app_window_instance = Mock()
        mock_app_window_instance.get_root.return_value = mock_root
        mock_app_window_instance.get_gui_window.return_value = mock_gui_window
        mock_app_window.return_value = mock_app_window_instance

        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Mock DXGI to return Intel integrated GPU
        mock_dxgi.return_value = [
            {'index': 0, 'name': 'Intel(R) Iris(R) Xe Graphics', 'type': 'integrated', 'vram_gb': 1.0}
        ]

        with patch('onnxruntime.get_available_providers') as mock_providers:
            mock_providers.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

            pipeline = STTPipeline(verbose=False)

        assert pipeline.execution_provider_manager.detect_gpu_type() == 'integrated'

        # The strategy should have configured sess_options with enable_cpu_mem_arena=False
        call_args = mock_load_model.call_args
        sess_options = call_args[1].get('sess_options')
        if sess_options:
            assert sess_options.enable_cpu_mem_arena is False, \
                "IntegratedGPUStrategy should disable CPU memory arena"

    @patch('src.pipeline.onnx_asr.load_model')
    @patch('src.gui.ApplicationWindow.ApplicationWindow')
    @patch('src.pipeline.AudioSource')
    @patch('src.pipeline.VoiceActivityDetector')
    @patch('src.ExecutionProviderManager.ExecutionProviderManager._enumerate_adapters_dxgi')
    def test_pipeline_uses_discrete_gpu_strategy(self, mock_dxgi, mock_vad, mock_audio, mock_app_window, mock_load_model):
        """Pipeline should use DiscreteGPUStrategy for discrete GPU."""
        # Mock ApplicationWindow
        mock_root = Mock()
        mock_gui_window = Mock()
        mock_app_window_instance = Mock()
        mock_app_window_instance.get_root.return_value = mock_root
        mock_app_window_instance.get_gui_window.return_value = mock_gui_window
        mock_app_window.return_value = mock_app_window_instance

        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Mock DXGI to return NVIDIA discrete GPU
        mock_dxgi.return_value = [
            {'index': 0, 'name': 'NVIDIA GeForce RTX 3060', 'type': 'discrete', 'vram_gb': 8.0}
        ]

        with patch('onnxruntime.get_available_providers') as mock_providers:
            mock_providers.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

            pipeline = STTPipeline(verbose=False)

        assert pipeline.execution_provider_manager.detect_gpu_type() == 'discrete'

        # Verify session options were configured for discrete GPU
        call_args = mock_load_model.call_args
        sess_options = call_args[1].get('sess_options')
        if sess_options:
            assert sess_options.enable_cpu_mem_arena is True, \
                "DiscreteGPUStrategy should enable CPU memory arena"

    @patch('src.pipeline.onnx_asr.load_model')
    @patch('src.gui.ApplicationWindow.ApplicationWindow')
    @patch('src.pipeline.AudioSource')
    @patch('src.pipeline.VoiceActivityDetector')
    def test_pipeline_uses_cpu_strategy(self, mock_vad, mock_audio, mock_app_window, mock_load_model):
        """Pipeline should use CPUStrategy for CPU mode."""
        # Mock ApplicationWindow
        mock_root = Mock()
        mock_gui_window = Mock()
        mock_app_window_instance = Mock()
        mock_app_window_instance.get_root.return_value = mock_root
        mock_app_window_instance.get_gui_window.return_value = mock_gui_window
        mock_app_window.return_value = mock_app_window_instance

        mock_model = Mock()
        mock_load_model.return_value = mock_model

        import tempfile
        import json

        config = {
            "audio": {
                "sample_rate": 16000,
                "chunk_duration": 0.032,
                "silence_energy_threshold": 1.2,
                "rms_normalization": {
                    "target_rms": 0.05,
                    "silence_threshold": 0.0001,
                    "gain_smoothing": 0.85
                }
            },
            "vad": {"frame_duration_ms": 32, "threshold": 0.5},
            "windowing": {"window_duration": 3.0, "max_speech_duration_ms": 1500, "silence_timeout": 0.5},
            "recognition": {"model_name": "parakeet", "inference": "cpu"}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            json.dump(config, f)

        try:
            pipeline = STTPipeline(config_path=config_path, verbose=False)

            # Verify CPU mode was selected
            assert pipeline.execution_provider_manager.selected_provider == 'CPU'
            assert pipeline.execution_provider_manager.detect_gpu_type() == 'cpu'

            # Verify session options were configured for CPU
            call_args = mock_load_model.call_args
            sess_options = call_args[1].get('sess_options')
            if sess_options:
                assert sess_options.enable_cpu_mem_arena is True, \
                    "CPUStrategy should enable CPU memory arena"
                assert sess_options.intra_op_num_threads >= 1, \
                    "CPUStrategy should set thread counts"
        finally:
            Path(config_path).unlink()

    @patch('src.pipeline.onnx_asr.load_model')
    @patch('src.gui.ApplicationWindow.ApplicationWindow')
    @patch('src.pipeline.AudioSource')
    @patch('src.pipeline.VoiceActivityDetector')
    @patch('src.ExecutionProviderManager.ExecutionProviderManager._enumerate_adapters_dxgi')
    def test_session_options_correctly_applied_by_strategy(self, mock_dxgi, mock_vad, mock_audio, mock_app_window, mock_load_model):
        """Session options from strategy should be passed to model loading."""
        # Mock ApplicationWindow
        mock_root = Mock()
        mock_gui_window = Mock()
        mock_app_window_instance = Mock()
        mock_app_window_instance.get_root.return_value = mock_root
        mock_app_window_instance.get_gui_window.return_value = mock_gui_window
        mock_app_window.return_value = mock_app_window_instance

        # Mock model loading
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Mock DXGI to return NVIDIA discrete GPU
        mock_dxgi.return_value = [
            {'index': 0, 'name': 'NVIDIA GeForce RTX 3060', 'type': 'discrete', 'vram_gb': 8.0}
        ]

        with patch('onnxruntime.get_available_providers') as mock_providers:
            mock_providers.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

            pipeline = STTPipeline(verbose=False)

        # Verify model was called with sess_options
        call_args = mock_load_model.call_args
        assert 'sess_options' in call_args[1], \
            "sess_options should be passed to load_model()"

        # Verify providers were passed
        assert 'providers' in call_args[1], \
            "providers should be passed to load_model()"

        providers = call_args[1]['providers']
        assert isinstance(providers, list), \
            "providers should be list format"
        assert len(providers) > 0, \
            "providers should contain at least one execution provider"
        # First provider should be DmlExecutionProvider tuple for discrete GPU
        assert isinstance(providers[0], tuple), \
            "DmlExecutionProvider should be tuple with options"
        assert providers[0][0] == 'DmlExecutionProvider', \
            "First provider should be DmlExecutionProvider"
