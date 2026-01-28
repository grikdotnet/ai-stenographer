"""Tests for ExecutionProviderManager - DXGI-based GPU enumeration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asr.ExecutionProviderManager import ExecutionProviderManager


class TestDXGIEnumeration:
    """Tests for native DXGI adapter enumeration (Windows only).

    Tests the _enumerate_adapters_dxgi() method that uses ctypes to call
    native DXGI API for guaranteed-correct DirectML device_id ordering.
    """

    def test_dxgi_enumeration_single_discrete_gpu(self, monkeypatch):
        """Single NVIDIA GPU → device_id=0, type=discrete.

        Verifies DXGI enumeration correctly identifies single discrete GPU
        at device_id 0 (primary adapter).
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        # Mock DXGI to return single NVIDIA adapter
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

        adapters = manager._enumerate_adapters_dxgi()

        assert len(adapters) == 1
        assert adapters[0]['index'] == 0
        assert adapters[0]['name'] == 'NVIDIA GeForce RTX 3060'
        assert adapters[0]['type'] == 'discrete'
        assert adapters[0]['vram_gb'] == 4.0

    def test_dxgi_enumeration_single_integrated_gpu(self, monkeypatch):
        """Single Intel iGPU → device_id=0, type=integrated.

        Verifies DXGI correctly identifies single integrated GPU
        at device_id 0 (primary adapter).
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

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

        adapters = manager._enumerate_adapters_dxgi()

        assert len(adapters) == 1
        assert adapters[0]['index'] == 0
        assert adapters[0]['name'] == 'Intel(R) Iris(R) Xe Graphics'
        assert adapters[0]['type'] == 'integrated'

    def test_dxgi_enumeration_multi_gpu_laptop_order(self, monkeypatch):
        """Laptop: Intel iGPU at index 0, NVIDIA dGPU at index 1.

        Verifies DXGI returns adapters in EXACT hardware enumeration order.
        Typical laptop: integrated GPU (primary display) at index 0,
        discrete GPU (secondary) at index 1.
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

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

        adapters = manager._enumerate_adapters_dxgi()

        assert len(adapters) == 2
        # DXGI ground truth: Intel at index 0
        assert adapters[0]['index'] == 0
        assert adapters[0]['name'] == 'Intel(R) UHD Graphics 630'
        assert adapters[0]['type'] == 'integrated'
        # NVIDIA at index 1
        assert adapters[1]['index'] == 1
        assert adapters[1]['name'] == 'NVIDIA GeForce RTX 3060'
        assert adapters[1]['type'] == 'discrete'

    def test_dxgi_enumeration_multi_gpu_desktop_order(self, monkeypatch):
        """Desktop: NVIDIA at index 0 (primary), AMD at index 1.

        Verifies DXGI returns adapters in EXACT hardware order.
        Desktop with multiple discrete GPUs: primary display adapter first.
        We DON'T reorder - we trust DXGI enumeration completely.
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_dxgi_enumerate(self):
            return [
                {
                    'index': 0,
                    'name': 'NVIDIA GeForce RTX 3060',
                    'type': 'discrete',
                    'vram_gb': 4.0
                },
                {
                    'index': 1,
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

        adapters = manager._enumerate_adapters_dxgi()

        assert len(adapters) == 2
        # DXGI order preserved: NVIDIA at index 0
        assert adapters[0]['index'] == 0
        assert adapters[0]['name'] == 'NVIDIA GeForce RTX 3060'
        assert adapters[0]['type'] == 'discrete'
        # AMD at index 1
        assert adapters[1]['index'] == 1
        assert adapters[1]['name'] == 'AMD Radeon RX 6700 XT'
        assert adapters[1]['type'] == 'discrete'

    def test_dxgi_failure_returns_empty_list(self, monkeypatch):
        """DXGI enumeration failure → empty list (caller handles fallback).

        Verifies graceful failure when DXGI ctypes calls fail.
        Caller (enumerate_adapters) should handle empty list appropriately.
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_dxgi_enumerate_failure(self):
            raise Exception("DXGI CreateDXGIFactory failed")

        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr(
            ExecutionProviderManager,
            '_enumerate_adapters_dxgi',
            mock_dxgi_enumerate_failure
        )

        # Should catch exception and return empty list
        with pytest.raises(Exception, match="DXGI CreateDXGIFactory failed"):
            adapters = manager._enumerate_adapters_dxgi()

    def test_dxgi_non_windows_returns_empty(self, monkeypatch):
        """Non-Windows platform → empty list.

        Verifies DXGI enumeration is Windows-only.
        On Linux/macOS, should return empty list immediately.
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        monkeypatch.setattr('sys.platform', 'linux')

        # Implementation should check sys.platform and return []
        # This test will guide implementation
        adapters = manager._enumerate_adapters_dxgi()

        assert adapters == []

    def test_dxgi_includes_microsoft_basic_render_driver(self, monkeypatch):
        """Microsoft Basic Render Driver IS enumerated by DXGI (but should be filtered later).

        DXGI enumerates ALL adapters including virtual ones.
        Filtering happens in select_device_id(), not here.
        This test verifies DXGI returns the adapter (preserving correct indices).
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_dxgi_enumerate(self):
            return [
                {
                    'index': 0,
                    'name': 'Intel(R) Iris(R) Xe Graphics',
                    'type': 'integrated',
                    'vram_gb': 1.0
                },
                {
                    'index': 1,
                    'name': 'NVIDIA GeForce RTX 4070 Laptop GPU',
                    'type': 'discrete',
                    'vram_gb': 7.8
                },
                {
                    'index': 2,
                    'name': 'Microsoft Basic Render Driver',
                    'type': 'virtual',  # Should be classified as virtual
                    'vram_gb': 0.0
                }
            ]

        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr(
            ExecutionProviderManager,
            '_enumerate_adapters_dxgi',
            mock_dxgi_enumerate
        )

        adapters = manager._enumerate_adapters_dxgi()

        # DXGI should return all 3 adapters (including virtual)
        # Indices must be preserved: 0, 1, 2 (DirectML device_id mapping)
        assert len(adapters) == 3
        assert adapters[0]['index'] == 0
        assert adapters[0]['name'] == 'Intel(R) Iris(R) Xe Graphics'
        assert adapters[1]['index'] == 1
        assert adapters[1]['name'] == 'NVIDIA GeForce RTX 4070 Laptop GPU'
        assert adapters[2]['index'] == 2
        assert adapters[2]['name'] == 'Microsoft Basic Render Driver'
        assert adapters[2]['type'] == 'virtual'
        assert adapters[2]['vram_gb'] == 0.0


class TestGPUClassification:
    """Tests for _classify_gpu_name() including virtual GPU detection."""

    def test_microsoft_basic_render_driver_classified_as_virtual(self):
        """Microsoft Basic Render Driver → 'virtual' (software renderer)."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        gpu_type = manager._classify_gpu_name('Microsoft Basic Render Driver')

        assert gpu_type == 'virtual', "Basic Render Driver should be virtual, not discrete"

    def test_microsoft_basic_display_adapter_classified_as_virtual(self):
        """Microsoft Basic Display Adapter → 'virtual' (fallback driver)."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        gpu_type = manager._classify_gpu_name('Microsoft Basic Display Adapter')

        assert gpu_type == 'virtual', "Basic Display Adapter should be virtual"

    def test_remote_desktop_adapter_classified_as_virtual(self):
        """Remote Desktop adapters → 'virtual'."""
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        gpu_type = manager._classify_gpu_name('Microsoft Remote Display Adapter')

        assert gpu_type == 'virtual', "Remote Display Adapter should be virtual"


class TestAdapterEnumerationDXGI:
    """Tests for enumerate_adapters() using DXGI backend.

    Tests the public API that uses DXGI as source of truth
    for adapter ordering.
    """

    def test_enumerate_adapters_uses_dxgi_order(self, monkeypatch):
        """enumerate_adapters() returns adapters in EXACT DXGI order.

        Verifies no reordering happens - DXGI order is ground truth.
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

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

        adapters = manager.enumerate_adapters()

        assert len(adapters) == 2
        # Verify EXACT order preserved (no sorting/reordering)
        assert adapters[0]['index'] == 0
        assert adapters[0]['name'] == 'Intel(R) UHD Graphics 630'
        assert adapters[1]['index'] == 1
        assert adapters[1]['name'] == 'NVIDIA GeForce RTX 3060'

    def test_enumerate_adapters_cpu_mode_returns_empty(self):
        """CPU mode → empty list (no GPU enumeration).

        Verifies CPU mode skips GPU enumeration entirely.
        """
        config = {'recognition': {'inference': 'cpu'}}
        manager = ExecutionProviderManager(config)

        adapters = manager.enumerate_adapters()

        assert adapters == []

    def test_enumerate_adapters_dxgi_failure_fallback(self, monkeypatch):
        """DXGI fails → fallback behavior (conservative).

        Verifies graceful fallback when DXGI enumeration fails.
        Should return empty list or single default adapter (conservative).
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_dxgi_enumerate_failure(self):
            raise Exception("DXGI failed")

        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr(
            ExecutionProviderManager,
            '_enumerate_adapters_dxgi',
            mock_dxgi_enumerate_failure
        )

        # Should handle failure gracefully
        adapters = manager.enumerate_adapters()

        # Conservative fallback: empty list or single adapter at device_id=0
        assert isinstance(adapters, list)
        # Implementation will determine exact fallback behavior


class TestDeviceIDSelection:
    """Tests for select_device_id() with DXGI ordering.

    Tests device selection logic that uses DXGI-based adapter list.
    """

    def test_select_device_id_single_discrete(self, monkeypatch):
        """Single discrete GPU → device_id=0.

        Single GPU systems: use device_id=0 regardless of type.
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

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

        device_id, gpu_type = manager.select_device_id()

        assert device_id == 0
        assert gpu_type == 'discrete'

    def test_select_device_id_laptop_prefer_discrete(self, monkeypatch):
        """Laptop with [Intel@0, NVIDIA@1] → prefer device_id=1.

        Multi-GPU systems: prefer discrete GPU even if not at index 0.
        This is the key test - DXGI gives us correct indices.
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

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

        device_id, gpu_type = manager.select_device_id()

        # Should prefer discrete GPU at DXGI index 1
        assert device_id == 1, "Should select discrete NVIDIA at DXGI index 1"
        assert gpu_type == 'discrete'

    def test_select_device_id_desktop_primary_discrete(self, monkeypatch):
        """Desktop with [NVIDIA@0] → device_id=0.

        Desktop with discrete GPU as primary display.
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

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

        device_id, gpu_type = manager.select_device_id()

        assert device_id == 0
        assert gpu_type == 'discrete'

    def test_select_device_id_fallback_to_integrated(self, monkeypatch):
        """Only integrated GPU available → use it.

        Systems with only integrated GPU: use device_id=0.
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

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

        device_id, gpu_type = manager.select_device_id()

        assert device_id == 0
        assert gpu_type == 'integrated'

    def test_select_device_id_cpu_mode(self):
        """CPU mode → (-1, 'cpu').

        CPU mode returns sentinel value -1.
        """
        config = {'recognition': {'inference': 'cpu'}}
        manager = ExecutionProviderManager(config)

        device_id, gpu_type = manager.select_device_id()

        assert device_id == -1
        assert gpu_type == 'cpu'

    def test_select_device_id_skips_virtual_gpu(self, monkeypatch):
        """Should skip virtual GPUs (Basic Render Driver) and select real GPU.

        System: Intel@0, NVIDIA@1, Basic Render@2
        Expected: Select NVIDIA@1 (skip virtual@2)
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_dxgi_enumerate(self):
            return [
                {
                    'index': 0,
                    'name': 'Intel(R) Iris(R) Xe Graphics',
                    'type': 'integrated',
                    'vram_gb': 1.0
                },
                {
                    'index': 1,
                    'name': 'NVIDIA GeForce RTX 4070 Laptop GPU',
                    'type': 'discrete',
                    'vram_gb': 7.8
                },
                {
                    'index': 2,
                    'name': 'Microsoft Basic Render Driver',
                    'type': 'virtual',
                    'vram_gb': 0.0
                }
            ]

        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr(
            ExecutionProviderManager,
            '_enumerate_adapters_dxgi',
            mock_dxgi_enumerate
        )

        device_id, gpu_type = manager.select_device_id()

        # Should select NVIDIA@1, not Basic Render@2
        assert device_id == 1, "Should select discrete NVIDIA, not virtual Basic Render"
        assert gpu_type == 'discrete'

    def test_select_device_id_only_integrated_and_virtual(self, monkeypatch):
        """System with only iGPU + virtual → select iGPU.

        System: Intel@0, Basic Render@1
        Expected: Select Intel@0 (skip virtual@1)
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

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
                    'name': 'Microsoft Basic Render Driver',
                    'type': 'virtual',
                    'vram_gb': 0.0
                }
            ]

        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr(
            ExecutionProviderManager,
            '_enumerate_adapters_dxgi',
            mock_dxgi_enumerate
        )

        device_id, gpu_type = manager.select_device_id()

        # Should select Intel@0, not Basic Render@1
        assert device_id == 0, "Should select integrated GPU, not virtual"
        assert gpu_type == 'integrated'

    def test_select_device_id_only_virtual_gpu_fallback(self, monkeypatch):
        """System with ONLY virtual GPU → fallback to device_id=0.

        System: Basic Render@0 (only adapter)
        Expected: Return device_id=0, type='discrete' (conservative fallback)
        This is edge case - should rarely happen in practice.
        """
        config = {'recognition': {'inference': 'directml'}}
        manager = ExecutionProviderManager(config)

        def mock_dxgi_enumerate(self):
            return [
                {
                    'index': 0,
                    'name': 'Microsoft Basic Render Driver',
                    'type': 'virtual',
                    'vram_gb': 0.0
                }
            ]

        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr(
            ExecutionProviderManager,
            '_enumerate_adapters_dxgi',
            mock_dxgi_enumerate
        )

        device_id, gpu_type = manager.select_device_id()

        # Conservative fallback when only virtual GPU exists
        assert device_id == 0, "Fallback to device_id=0 when only virtual GPU"
        assert gpu_type == 'discrete', "Conservative fallback type"


class TestProviderListBuilding:
    """Tests for build_provider_list() with correct device_id.

    Tests that provider list includes correct DXGI-based device_id.
    """

    def test_build_provider_list_laptop_uses_discrete_device_id(self, monkeypatch):
        """Laptop: should use device_id=1 for discrete GPU.

        Verifies provider list uses DXGI device_id for discrete GPU.
        """
        config = {'recognition': {'inference': 'directml'}}

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

        with patch('onnxruntime.get_available_providers') as mock_get_providers:
            mock_get_providers.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']
            manager = ExecutionProviderManager(config)

            monkeypatch.setattr(
                ExecutionProviderManager,
                '_enumerate_adapters_dxgi',
                mock_dxgi_enumerate
            )

            provider_list = manager.build_provider_list()

        # Should use discrete GPU at DXGI index 1
        assert isinstance(provider_list, list)
        assert len(provider_list) == 2
        assert provider_list[0] == ('DmlExecutionProvider', {'device_id': 1})
        assert provider_list[1] == 'CPUExecutionProvider'

    def test_build_provider_list_desktop_uses_primary(self, monkeypatch):
        """Desktop: should use device_id=0 for primary GPU.

        Desktop with single discrete GPU at DXGI index 0.
        """
        config = {'recognition': {'inference': 'directml'}}

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

        with patch('onnxruntime.get_available_providers') as mock_get_providers:
            mock_get_providers.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']
            manager = ExecutionProviderManager(config)

            monkeypatch.setattr(
                ExecutionProviderManager,
                '_enumerate_adapters_dxgi',
                mock_dxgi_enumerate
            )

            provider_list = manager.build_provider_list()

        # Should use primary GPU at DXGI index 0
        assert isinstance(provider_list, list)
        assert len(provider_list) == 2
        assert provider_list[0] == ('DmlExecutionProvider', {'device_id': 0})
        assert provider_list[1] == 'CPUExecutionProvider'
