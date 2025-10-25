"""Tests for ExecutionProviderManager - DirectML/CPU provider selection and GPU detection."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import onnxruntime as ort
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ExecutionProviderManager import ExecutionProviderManager


class TestExecutionProviderManager:
    """Tests for execution provider selection and device detection."""

    # Provider detection
    def test_get_available_providers_includes_cpu(self):
        """ONNX Runtime should always include CPU provider."""
        providers = ort.get_available_providers()

        # Should always include CPU
        assert 'CPUExecutionProvider' in providers
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_directml_provider_available_on_windows(self):
        """DirectML provider should be available on Windows."""
        if sys.platform != 'win32':
            pytest.skip("DirectML only available on Windows")

        providers = ort.get_available_providers()

        # DirectML should be available on Windows
        assert 'DmlExecutionProvider' in providers

    # Provider selection logic
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
        """Should return DirectML provider config dict with dynamically selected device_id."""
        with patch.object(ExecutionProviderManager, 'select_provider') as mock_select:
            mock_select.return_value = 'DirectML'

            manager = ExecutionProviderManager({"recognition": {"inference": "directml"}})
            providers = manager.build_provider_list()

        # Expected: {'DmlExecutionProvider': {'device_id': <N>}, 'CPUExecutionProvider': {}}
        assert isinstance(providers, dict)
        assert len(providers) == 2
        assert 'DmlExecutionProvider' in providers
        assert 'CPUExecutionProvider' in providers
        assert isinstance(providers['DmlExecutionProvider'], dict)
        assert 'device_id' in providers['DmlExecutionProvider']
        assert isinstance(providers['DmlExecutionProvider']['device_id'], int)
        assert providers['CPUExecutionProvider'] == {}

    def test_build_provider_list_for_cpu(self):
        """CPU mode should return CPU provider config dict."""
        with patch.object(ExecutionProviderManager, 'select_provider') as mock_select:
            mock_select.return_value = 'CPU'

            manager = ExecutionProviderManager({"recognition": {"inference": "cpu"}})
            providers = manager.build_provider_list()

        assert providers == {'CPUExecutionProvider': {}}

    # Session creation tests - validates actual ONNX Runtime integration
    def test_create_session_with_directml_provider(self):
        """Should create ONNX Runtime session with DirectML provider."""
        # Create a minimal dummy ONNX model for testing
        try:
            import onnx
            from onnx import helper, TensorProto
        except ImportError:
            pytest.skip("onnx package not installed")

        # Simple model: input -> identity -> output
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])
        node = helper.make_node('Identity', ['input'], ['output'])
        graph = helper.make_graph([node], 'test_graph', [input_tensor], [output_tensor])
        model = helper.make_model(graph)

        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            model_path = f.name
            onnx.save(model, model_path)

        try:
            # Build DirectML provider list
            with patch.object(ExecutionProviderManager, 'select_provider') as mock_select:
                mock_select.return_value = 'DirectML'

                manager = ExecutionProviderManager({"recognition": {"inference": "directml"}})
                provider_list = manager.build_provider_list()

            # Create session
            session = ort.InferenceSession(model_path, providers=provider_list)

            # Check active providers
            active_providers = session.get_providers()

            # DirectML should be in active providers OR silent fallback to CPU (both valid)
            assert 'CPUExecutionProvider' in active_providers
            # DirectML may or may not be active depending on hardware
            assert len(active_providers) > 0
        finally:
            Path(model_path).unlink()


class TestGPUDetectionSingleGPU:
    """Tests for single GPU scenarios (reduced set - 3 tests)."""

    def test_intel_integrated_gpu(self, monkeypatch):
        """Intel integrated GPU should be detected correctly."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        # Mock Windows wmic output for Intel Iris Xe
        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = "Name\nIntel(R) Iris(R) Xe Graphics\n"
            return result

        monkeypatch.setattr('subprocess.run', mock_subprocess_run)
        monkeypatch.setattr('sys.platform', 'win32')

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'integrated', "Intel GPU should be detected as integrated"

    def test_nvidia_discrete_gpu(self, monkeypatch):
        """NVIDIA discrete GPU should be detected correctly."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = "Name\nNVIDIA GeForce RTX 3060\n"
            return result

        monkeypatch.setattr('subprocess.run', mock_subprocess_run)
        monkeypatch.setattr('sys.platform', 'win32')

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'discrete', "NVIDIA RTX should be detected as discrete"

    def test_amd_discrete_gpu(self, monkeypatch):
        """AMD discrete GPU should be detected correctly."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = "Name\nAMD Radeon RX 6700 XT\n"
            return result

        monkeypatch.setattr('subprocess.run', mock_subprocess_run)
        monkeypatch.setattr('sys.platform', 'win32')

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'discrete', "AMD Radeon RX should be detected as discrete"


class TestGPUDetectionMultiGPU:
    """Tests for multi-GPU scenarios (2 tests)."""

    def test_intel_plus_nvidia_prefer_discrete(self, monkeypatch):
        """Multi-GPU: Intel integrated + NVIDIA discrete → prefer discrete."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = "Name\nIntel(R) UHD Graphics 630\nNVIDIA GeForce RTX 3060\n"
            return result

        monkeypatch.setattr('subprocess.run', mock_subprocess_run)
        monkeypatch.setattr('sys.platform', 'win32')

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'discrete', "Should prioritize discrete GPU (NVIDIA RTX)"

    def test_both_intel_stay_integrated(self, monkeypatch):
        """Multi-GPU: Both Intel (UHD + Iris) → integrated."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = "Name\nIntel(R) UHD Graphics 630\nIntel(R) Iris(R) Xe Graphics\n"
            return result

        monkeypatch.setattr('subprocess.run', mock_subprocess_run)
        monkeypatch.setattr('sys.platform', 'win32')

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'integrated', "Both Intel GPUs are integrated"


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

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = "Name\nQuantum Processing Unit 9000\n"
            return result

        monkeypatch.setattr('subprocess.run', mock_subprocess_run)
        monkeypatch.setattr('sys.platform', 'win32')

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'discrete', "Unknown GPU should default to discrete"

    def test_subprocess_failure_defaults_to_discrete(self, monkeypatch):
        """Subprocess failure should default to 'discrete' (safe fallback)."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_subprocess_run(cmd, **kwargs):
            raise Exception("wmic command failed")

        monkeypatch.setattr('subprocess.run', mock_subprocess_run)
        monkeypatch.setattr('sys.platform', 'win32')

        gpu_type = manager.detect_gpu_type()

        assert gpu_type == 'discrete', "Subprocess failure should default to discrete"


class TestAdapterEnumeration:
    """Tests for adapter enumeration and device_id selection."""

    def test_enumerate_adapters_single_gpu(self, monkeypatch):
        """Should enumerate single GPU adapter with correct index."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        # Mock wmic output with DeviceID column
        # wmic format: DeviceID is first column (index 0)
        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            # wmic output format: DeviceID    Name
            result.stdout = (
                "DeviceID    Name\n"
                "0           NVIDIA GeForce RTX 3060\n"
            )
            return result

        monkeypatch.setattr('subprocess.run', mock_subprocess_run)
        monkeypatch.setattr('sys.platform', 'win32')

        adapters = manager.enumerate_adapters()

        assert len(adapters) == 1
        assert adapters[0]['index'] == 0
        assert adapters[0]['name'] == 'NVIDIA GeForce RTX 3060'
        assert adapters[0]['type'] == 'discrete'

    def test_enumerate_adapters_multi_gpu(self, monkeypatch):
        """Should enumerate multiple GPU adapters with correct indices."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = (
                "DeviceID    Name\n"
                "0           Intel(R) UHD Graphics 630\n"
                "1           NVIDIA GeForce RTX 3060\n"
            )
            return result

        monkeypatch.setattr('subprocess.run', mock_subprocess_run)
        monkeypatch.setattr('sys.platform', 'win32')

        adapters = manager.enumerate_adapters()

        assert len(adapters) == 2
        assert adapters[0]['index'] == 0
        assert adapters[0]['name'] == 'Intel(R) UHD Graphics 630'
        assert adapters[0]['type'] == 'integrated'
        assert adapters[1]['index'] == 1
        assert adapters[1]['name'] == 'NVIDIA GeForce RTX 3060'
        assert adapters[1]['type'] == 'discrete'

    def test_enumerate_adapters_cpu_mode_returns_empty(self):
        """CPU mode should return empty adapter list."""
        config = {'recognition': {'inference': 'cpu'}}
        manager = ExecutionProviderManager(config)

        adapters = manager.enumerate_adapters()

        assert adapters == []

    def test_select_device_id_single_discrete_gpu(self, monkeypatch):
        """Should select adapter index for single discrete GPU."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = (
                "DeviceID    Name\n"
                "0           NVIDIA GeForce RTX 3060\n"
            )
            return result

        monkeypatch.setattr('subprocess.run', mock_subprocess_run)
        monkeypatch.setattr('sys.platform', 'win32')

        device_id, gpu_type = manager.select_device_id()

        assert device_id == 0
        assert gpu_type == 'discrete'

    def test_select_device_id_prefer_discrete_in_multi_gpu(self, monkeypatch):
        """Should select discrete GPU adapter when both integrated and discrete present."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = (
                "DeviceID    Name\n"
                "0           Intel(R) UHD Graphics 630\n"
                "1           NVIDIA GeForce RTX 3060\n"
            )
            return result

        monkeypatch.setattr('subprocess.run', mock_subprocess_run)
        monkeypatch.setattr('sys.platform', 'win32')

        device_id, gpu_type = manager.select_device_id()

        assert device_id == 1, "Should select NVIDIA at adapter 1"
        assert gpu_type == 'discrete'

    def test_select_device_id_cpu_mode_returns_default(self):
        """CPU mode should return default device_id and 'cpu' type."""
        config = {'recognition': {'inference': 'cpu'}}
        manager = ExecutionProviderManager(config)

        device_id, gpu_type = manager.select_device_id()

        assert device_id == -1, "CPU mode should return -1"
        assert gpu_type == 'cpu'

    def test_build_provider_list_uses_selected_device_id(self, monkeypatch):
        """Should use selected device_id when building provider list."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_subprocess_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = (
                "DeviceID    Name\n"
                "0           Intel(R) UHD Graphics 630\n"
                "1           NVIDIA GeForce RTX 3060\n"
            )
            return result

        monkeypatch.setattr('subprocess.run', mock_subprocess_run)
        monkeypatch.setattr('sys.platform', 'win32')

        with patch.object(ExecutionProviderManager, 'select_provider') as mock_select:
            mock_select.return_value = 'DirectML'
            provider_list = manager.build_provider_list()

        # Should use device_id=1 (discrete GPU)
        assert isinstance(provider_list, dict)
        assert len(provider_list) == 2
        assert provider_list == {
            'DmlExecutionProvider': {'device_id': 1},
            'CPUExecutionProvider': {}
        }
