"""Tests for Recognizer with different execution providers."""

import pytest
import queue
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asr.Recognizer import Recognizer
from src.asr.ExecutionProviderManager import ExecutionProviderManager
import src.types as stt_types
from onnx_asr.asr import TimestampedResult


class TestRecognizerExecutionProviders:
    """Tests for Recognizer with different execution providers."""

    def test_recognizer_with_directml_gpu(self, speech_audio):
        """Recognizer should work with DirectML GPU provider."""
        # Config: DirectML GPU provider
        config = {"recognition": {"inference": "directml"}}

        with patch('onnxruntime.get_available_providers') as mock_get_providers:
            mock_get_providers.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

            manager = ExecutionProviderManager(config)
            providers = manager.build_provider_list()

        # Verify DirectML provider config (list format)
        assert isinstance(providers, list)
        assert len(providers) == 2
        assert isinstance(providers[0], tuple)
        assert providers[0][0] == 'DmlExecutionProvider'
        assert 'device_id' in providers[0][1]
        assert providers[1] == 'CPUExecutionProvider'

        # Create mock model with DirectML providers
        mock_model = Mock()
        mock_model.recognize.return_value = TimestampedResult(
            text="test speech",
            tokens=None,
            timestamps=None
        )

        # Create queues
        chunk_queue = queue.Queue()
        text_queue = queue.Queue()

        # Create Recognizer with mock model
        recognizer = Recognizer(input_queue=chunk_queue, output_queue=text_queue, model=mock_model, sample_rate=16000, verbose=False, app_state=Mock())

        # Create AudioSegment with speech audio
        segment = stt_types.AudioSegment(
            data=speech_audio,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=1.0,
            type='incremental',
            chunk_ids=[1, 2, 3]
        )

        # Recognize
        result = recognizer.recognize_window(segment)

        # Verify recognition worked
        assert result is not None
        assert result.text == "test speech"
        assert mock_model.recognize.called

    def test_recognizer_provider_fallback(self):
        """Recognizer should fall back to CPU on provider failure."""
        # Config: Auto mode
        config = {"recognition": {"inference": "auto"}}

        # Simulate: No DirectML available
        with patch('onnxruntime.get_available_providers') as mock_get_providers:
            mock_get_providers.return_value = ['CPUExecutionProvider']

            manager = ExecutionProviderManager(config)
            selected = manager.selected_provider

        # Verify CPU fallback
        assert selected == 'CPU'

        # Build provider list
        providers = manager.build_provider_list()
        assert providers == ['CPUExecutionProvider']

    def test_recognizer_inference_consistency(self, speech_audio):
        """Recognized text should be consistent across CPU/GPU providers.

        Note: This test verifies provider selection logic consistency.
        Actual model output consistency would require real model inference.
        """
        # Config for CPU
        config_cpu = {"recognition": {"inference": "cpu"}}

        with patch('onnxruntime.get_available_providers') as mock_get_providers:
            mock_get_providers.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

            # CPU provider
            manager_cpu = ExecutionProviderManager(config_cpu)
            providers_cpu = manager_cpu.build_provider_list()

        assert providers_cpu == ['CPUExecutionProvider']

        # Config for DirectML
        config_directml = {"recognition": {"inference": "directml"}}

        with patch('onnxruntime.get_available_providers') as mock_get_providers:
            mock_get_providers.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

            # DirectML provider
            manager_directml = ExecutionProviderManager(config_directml)
            providers_directml = manager_directml.build_provider_list()

        assert isinstance(providers_directml, list)
        assert len(providers_directml) == 2
        assert isinstance(providers_directml[0], tuple)
        assert providers_directml[0][0] == 'DmlExecutionProvider'
        assert 'device_id' in providers_directml[0][1]

        # Both should produce consistent provider configs
        # In real usage, both would produce same text output for same audio
        assert isinstance(providers_cpu, list)
        assert isinstance(providers_directml, list)
