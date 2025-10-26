"""Tests for ExecutionProviderManager - DirectML/CPU provider selection and GPU detection."""

import pytest
from unittest.mock import Mock, patch
import onnxruntime as ort
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ExecutionProviderManager import ExecutionProviderManager


class TestExecutionProviderManager:
    """Tests for execution provider selection and device detection."""

    def test_select_directml_in_auto_mode(self):
        """Auto mode should select DirectML when available."""
        with patch('onnxruntime.get_available_providers') as mock_get_providers:
            mock_get_providers.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']
            manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})
        assert manager.selected_provider == 'DirectML'

    def test_explicit_directml_selection(self):
        """Config 'directml' should force DirectML."""
        with patch('onnxruntime.get_available_providers') as mock_get_providers:
            mock_get_providers.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']
            manager = ExecutionProviderManager({"recognition": {"inference": "directml"}})
        assert manager.selected_provider == 'DirectML'

    def test_select_provider_fallback_to_cpu(self):
        """Should fall back to CPU when GPU unavailable."""
        with patch('onnxruntime.get_available_providers') as mock_get_providers:
            mock_get_providers.return_value = ['CPUExecutionProvider']
            manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})
        assert manager.selected_provider == 'CPU'

    # Provider list building
    def test_build_provider_list_for_directml(self):
        """Should return DirectML provider config list with dynamically selected device_id."""
        with patch.object(ExecutionProviderManager, 'select_provider') as mock_select:
            mock_select.return_value = 'DirectML'
            manager = ExecutionProviderManager({"recognition": {"inference": "directml"}})
            providers = manager.build_provider_list()

        # Expected: [('DmlExecutionProvider', {'device_id': <N>}), 'CPUExecutionProvider']
        assert isinstance(providers, list)
        assert len(providers) == 2
        assert isinstance(providers[0], tuple)
        assert providers[0][0] == 'DmlExecutionProvider'
        assert isinstance(providers[0][1], dict)
        assert 'device_id' in providers[0][1]
        assert isinstance(providers[0][1]['device_id'], int)
        assert providers[1] == 'CPUExecutionProvider'

    def test_build_provider_list_for_cpu(self):
        """CPU mode should return CPU provider config list."""
        with patch.object(ExecutionProviderManager, 'select_provider') as mock_select:
            mock_select.return_value = 'CPU'

            manager = ExecutionProviderManager({"recognition": {"inference": "cpu"}})
            providers = manager.build_provider_list()

        assert providers == ['CPUExecutionProvider']


class TestGPUDetectionSingleGPU:
    """Tests for single GPU scenarios (reduced set - 3 tests)."""

    def test_intel_integrated_gpu(self, monkeypatch):
        """Intel integrated GPU should be detected correctly."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        # Mock DXGI enumeration for Intel Iris Xe
        def mock_dxgi_enumerate(self):
            return [
                {
                    'index': 0,
                    'name': 'Intel(R) Iris(R) Xe Graphics',
                    'type': 'integrated',
                    'vram_gb': 1.0
                }
            ]

        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr(
            ExecutionProviderManager,
            '_enumerate_adapters_dxgi',
            mock_dxgi_enumerate
        )

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'integrated', "Intel GPU should be detected as integrated"

    def test_nvidia_discrete_gpu(self, monkeypatch):
        """NVIDIA discrete GPU should be detected correctly."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        # Mock DXGI enumeration for NVIDIA RTX
        def mock_dxgi_enumerate(self):
            return [
                {
                    'index': 0,
                    'name': 'NVIDIA GeForce RTX 3060',
                    'type': 'discrete',
                    'vram_gb': 4.0
                }
            ]

        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr(
            ExecutionProviderManager,
            '_enumerate_adapters_dxgi',
            mock_dxgi_enumerate
        )

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'discrete', "NVIDIA RTX should be detected as discrete"

    def test_amd_discrete_gpu(self, monkeypatch):
        """AMD discrete GPU should be detected correctly."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        # Mock DXGI enumeration for AMD Radeon
        def mock_dxgi_enumerate(self):
            return [
                {
                    'index': 0,
                    'name': 'AMD Radeon RX 6700 XT',
                    'type': 'discrete',
                    'vram_gb': 12.0
                }
            ]

        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr(
            ExecutionProviderManager,
            '_enumerate_adapters_dxgi',
            mock_dxgi_enumerate
        )

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'discrete', "AMD Radeon RX should be detected as discrete"


class TestGPUDetectionMultiGPU:
    """Tests for multi-GPU scenarios (2 tests)."""

    def test_intel_plus_nvidia_prefer_discrete(self, monkeypatch):
        """Multi-GPU: Intel integrated + NVIDIA discrete → prefer discrete."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        # Mock DXGI enumeration with both GPUs in DXGI order
        def mock_dxgi_enumerate(self):
            return [
                {
                    'index': 0,
                    'name': 'Intel(R) UHD Graphics 630',
                    'type': 'integrated',
                    'vram_gb': 1.0
                },
                {
                    'index': 1,
                    'name': 'NVIDIA GeForce RTX 3060',
                    'type': 'discrete',
                    'vram_gb': 4.0
                }
            ]

        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr(
            ExecutionProviderManager,
            '_enumerate_adapters_dxgi',
            mock_dxgi_enumerate
        )

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'discrete', "Should prioritize discrete GPU (NVIDIA RTX)"

class TestGPUDetectionEdgeCases:
    """Tests for edge cases and error handling (3 tests)."""

    def test_cpu_mode_returns_cpu(self):
        """CPU mode should return 'cpu' without detection."""
        config = {'recognition': {'inference': 'cpu'}}
        manager = ExecutionProviderManager(config)

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'cpu', "CPU mode should return 'cpu'"

    def test_unknown_gpu_defaults_to_discrete(self, monkeypatch):
        """Unknown GPU should default to 'discrete' (conservative)."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        # Mock DXGI enumeration with unknown GPU name
        def mock_dxgi_enumerate(self):
            return [
                {
                    'index': 0,
                    'name': 'Quantum Processing Unit 9000',
                    'type': 'discrete',  # _classify_gpu_name defaults to discrete
                    'vram_gb': 8.0
                }
            ]

        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr(
            ExecutionProviderManager,
            '_enumerate_adapters_dxgi',
            mock_dxgi_enumerate
        )

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'discrete', "Unknown GPU should default to discrete"

    def test_dxgi_failure_defaults_to_discrete(self, monkeypatch):
        """DXGI enumeration failure should default to 'discrete' (safe fallback)."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        # Mock DXGI enumeration failure
        def mock_dxgi_enumerate_failure(self):
            raise Exception("DXGI CreateDXGIFactory failed")

        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr(
            ExecutionProviderManager,
            '_enumerate_adapters_dxgi',
            mock_dxgi_enumerate_failure
        )

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'discrete', "DXGI failure should default to discrete"
