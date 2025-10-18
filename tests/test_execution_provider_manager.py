"""Tests for ExecutionProviderManager - Windows ML/DirectML/CPU provider selection."""

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

    # Windows version detection (NEW)
    def test_detect_windows_version(self):
        """Should detect Windows version (major, build)."""
        manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})

        version = manager.detect_windows_version()

        # On Windows: (major, build) tuple like (10, 19045) or (11, 26100)
        # On non-Windows: None
        if sys.platform == 'win32':
            assert version is not None
            assert isinstance(version, tuple)
            assert len(version) == 2
            major, build = version
            assert major in [10, 11]
            assert isinstance(build, int)
            assert build > 0
        else:
            assert version is None

    @patch('platform.version')
    def test_windows_11_24h2_detection(self, mock_version):
        """Should detect Windows 11 24H2+ (build ≥ 26100) for WindowsML."""
        mock_version.return_value = "10.0.26100"

        with patch('sys.platform', 'win32'):
            manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})
            version = manager.detect_windows_version()

        assert version == (11, 26100)

    # Provider detection (UPDATED)
    def test_detect_available_providers(self):
        """Should detect ONNX Runtime available providers."""
        manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})

        providers = manager.detect_available_providers()

        # Should always include CPU
        assert 'CPUExecutionProvider' in providers
        # May include DirectML on Windows
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_detect_directml_provider(self):
        """Should detect DirectML provider on Windows."""
        manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})

        providers = manager.detect_available_providers()

        # DirectML should be available on Windows
        if sys.platform == 'win32':
            assert 'DmlExecutionProvider' in providers

    # Provider selection logic (UPDATED)
    @patch('platform.version')
    def test_select_windowsml_on_win11_24h2(self, mock_version):
        """Auto mode should select WindowsML on Win11 24H2+."""
        mock_version.return_value = "10.0.26100"

        with patch('sys.platform', 'win32'):
            with patch.object(ExecutionProviderManager, 'detect_available_providers') as mock_detect:
                mock_detect.return_value = ['WindowsMLExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']

                manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})
                selected = manager.select_provider()

        assert selected == 'WindowsML'

    @patch('platform.version')
    def test_select_directml_on_win10(self, mock_version):
        """Auto mode should select DirectML on Win10."""
        mock_version.return_value = "10.0.19045"

        with patch('sys.platform', 'win32'):
            with patch.object(ExecutionProviderManager, 'detect_available_providers') as mock_detect:
                mock_detect.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

                manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})
                selected = manager.select_provider()

        assert selected == 'DirectML'

    @patch('platform.version')
    def test_select_directml_on_older_win11(self, mock_version):
        """Auto mode should select DirectML on Win11 < 24H2."""
        mock_version.return_value = "10.0.22000"  # Win11 21H2

        with patch('sys.platform', 'win32'):
            with patch.object(ExecutionProviderManager, 'detect_available_providers') as mock_detect:
                mock_detect.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

                manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})
                selected = manager.select_provider()

        assert selected == 'DirectML'

    def test_explicit_directml_selection(self):
        """Config 'directml' should force DirectML."""
        with patch.object(ExecutionProviderManager, 'detect_available_providers') as mock_detect:
            mock_detect.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

            manager = ExecutionProviderManager({"recognition": {"inference": "directml"}})
            selected = manager.select_provider()

        assert selected == 'DirectML'

    def test_select_provider_fallback_to_cpu(self):
        """Should fall back to CPU when GPU unavailable."""
        with patch.object(ExecutionProviderManager, 'detect_available_providers') as mock_detect:
            mock_detect.return_value = ['CPUExecutionProvider']

            manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})
            selected = manager.select_provider()

        assert selected == 'CPU'

    # Provider list building (UPDATED)
    def test_build_provider_list_for_directml(self):
        """Should return DirectML provider config with device_id property."""
        with patch.object(ExecutionProviderManager, 'select_provider') as mock_select:
            mock_select.return_value = 'DirectML'

            manager = ExecutionProviderManager({"recognition": {"inference": "directml"}})
            providers = manager.build_provider_list()

        # Expected: [('DmlExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
        assert len(providers) == 2
        assert providers[0] == ('DmlExecutionProvider', {'device_id': 0})
        assert providers[1] == 'CPUExecutionProvider'

    def test_build_provider_list_for_cpu(self):
        """CPU mode should return simple provider list."""
        with patch.object(ExecutionProviderManager, 'select_provider') as mock_select:
            mock_select.return_value = 'CPU'

            manager = ExecutionProviderManager({"recognition": {"inference": "cpu"}})
            providers = manager.build_provider_list()

        assert providers == ['CPUExecutionProvider']

    def test_build_provider_list_for_windowsml(self):
        """Should return WindowsML provider config."""
        with patch.object(ExecutionProviderManager, 'select_provider') as mock_select:
            mock_select.return_value = 'WindowsML'

            manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})
            providers = manager.build_provider_list()

        # Expected: ['WindowsMLExecutionProvider', 'CPUExecutionProvider']
        assert len(providers) == 2
        assert providers[0] == 'WindowsMLExecutionProvider'
        assert providers[1] == 'CPUExecutionProvider'

    def test_get_device_info_for_gui_display(self):
        """Should return device info dict for GUI display."""
        with patch.object(ExecutionProviderManager, 'select_provider') as mock_select:
            with patch.object(ExecutionProviderManager, 'detect_available_providers') as mock_detect:
                mock_select.return_value = 'DirectML'
                mock_detect.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']

                manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})
                device_info = manager.get_device_info()

        assert 'selected' in device_info
        assert 'available' in device_info
        assert 'provider' in device_info
        assert device_info['selected'] == 'DirectML'
        assert device_info['provider'] == 'DirectML'
        assert 'DirectML' in device_info['available']
        assert 'CPU' in device_info['available']

    # Session creation tests (NEW - validates actual ONNX Runtime integration)
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

    def test_create_session_with_windowsml_provider(self):
        """Should create ONNX Runtime session with WindowsML provider."""
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
            # Check if WindowsML is available
            available_providers = ort.get_available_providers()

            if 'WindowsMLExecutionProvider' not in available_providers:
                pytest.skip("WindowsML provider not available (requires onnxruntime-winml)")

            # Build WindowsML provider list
            with patch.object(ExecutionProviderManager, 'select_provider') as mock_select:
                mock_select.return_value = 'WindowsML'

                manager = ExecutionProviderManager({"recognition": {"inference": "auto"}})
                provider_list = manager.build_provider_list()

            # Create session
            session = ort.InferenceSession(model_path, providers=provider_list)

            # Check active providers
            active_providers = session.get_providers()

            # WindowsML or CPU should be active (both valid)
            assert len(active_providers) > 0
            assert 'CPUExecutionProvider' in active_providers
        finally:
            Path(model_path).unlink()
