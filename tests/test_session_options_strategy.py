import pytest
from unittest.mock import Mock, MagicMock
from src.asr.SessionOptionsStrategy import (
    SessionOptionsStrategy,
    IntegratedGPUStrategy,
    DiscreteGPUStrategy,
    CPUStrategy
)
from src.asr.SessionOptionsFactory import SessionOptionsFactory


class TestIntegratedGPUStrategy:
    """Tests for integrated GPU session options (Intel/AMD APU)."""

    def test_session_options_memory_arena_disabled(self):
        """Integrated GPU should disable CPU memory arena (zero-copy shared memory)."""
        config = {}
        strategy = IntegratedGPUStrategy(config)
        sess_options = Mock()

        strategy.configure_session_options(sess_options)

        assert sess_options.enable_cpu_mem_arena is False, \
            "Integrated GPU must disable CPU arena for zero-copy"

    def test_session_options_threading(self):
        """Integrated GPU should use minimal threading (GPU handles parallelism)."""
        config = {}
        strategy = IntegratedGPUStrategy(config)
        sess_options = Mock()

        strategy.configure_session_options(sess_options)

        assert sess_options.intra_op_num_threads == 1
        assert sess_options.inter_op_num_threads == 1
        assert sess_options.enable_mem_pattern is True

    def test_no_directml_config_entries(self):
        """Integrated GPU should not force DirectML config (auto-enabled when possible)."""
        config = {}
        strategy = IntegratedGPUStrategy(config)
        sess_options = Mock()

        strategy.configure_session_options(sess_options)

        # Verify no DirectML config entries are forced (allows mixed CPU/GPU models)
        sess_options.add_session_config_entry.assert_not_called()

    def test_hardware_type(self):
        """Integrated GPU strategy should return 'integrated' type."""
        config = {}
        strategy = IntegratedGPUStrategy(config)

        assert strategy.get_hardware_type() == 'integrated'


class TestDiscreteGPUStrategy:
    """Tests for discrete GPU session options (NVIDIA/AMD dGPU)."""

    def test_session_options_memory_arena_enabled(self):
        """Discrete GPU should enable CPU memory arena (staging buffer for PCIe)."""
        config = {}
        strategy = DiscreteGPUStrategy(config)
        sess_options = Mock()

        strategy.configure_session_options(sess_options)

        assert sess_options.enable_cpu_mem_arena is True, \
            "Discrete GPU must enable CPU arena for PCIe staging"

    def test_session_options_threading(self):
        """Discrete GPU should use minimal threading (GPU handles parallelism)."""
        config = {}
        strategy = DiscreteGPUStrategy(config)
        sess_options = Mock()

        strategy.configure_session_options(sess_options)

        assert sess_options.intra_op_num_threads == 1
        assert sess_options.inter_op_num_threads == 1
        assert sess_options.enable_mem_pattern is True

    def test_no_directml_config_entries(self):
        """Discrete GPU should not force DirectML config (auto-enabled when possible)."""
        config = {}
        strategy = DiscreteGPUStrategy(config)
        sess_options = Mock()

        strategy.configure_session_options(sess_options)

        # Verify no DirectML config entries are forced (allows mixed CPU/GPU models)
        sess_options.add_session_config_entry.assert_not_called()

    def test_hardware_type(self):
        """Discrete GPU strategy should return 'discrete' type."""
        config = {}
        strategy = DiscreteGPUStrategy(config)

        assert strategy.get_hardware_type() == 'discrete'


class TestCPUStrategy:
    """Tests for CPU-only session options."""

    def test_session_options_memory_arena_enabled(self):
        """CPU should enable CPU memory arena."""
        config = {}
        strategy = CPUStrategy(config)
        sess_options = Mock()

        strategy.configure_session_options(sess_options)

        assert sess_options.enable_cpu_mem_arena is True

    def test_session_options_threading_auto_detect(self):
        """CPU should auto-detect thread counts with proper capping."""
        config = {}
        strategy = CPUStrategy(config)
        sess_options = Mock()

        strategy.configure_session_options(sess_options)

        # Thread counts should be capped
        assert sess_options.intra_op_num_threads <= 8
        assert sess_options.inter_op_num_threads <= 4
        assert sess_options.intra_op_num_threads >= 1
        assert sess_options.inter_op_num_threads >= 1
        assert sess_options.enable_mem_pattern is True

    def test_session_options_threading_high_core_system(self, monkeypatch):
        """CPU with 32 cores should cap at intra=8, inter=4."""
        import os
        monkeypatch.setattr(os, 'cpu_count', lambda: 32)

        config = {}
        strategy = CPUStrategy(config)
        sess_options = Mock()

        strategy.configure_session_options(sess_options)

        assert sess_options.intra_op_num_threads == 8, "Should cap at 8 threads"
        assert sess_options.inter_op_num_threads == 4, "Should cap at 4 threads"

    def test_hardware_type(self):
        """CPU strategy should return 'cpu' type."""
        config = {}
        strategy = CPUStrategy(config)

        assert strategy.get_hardware_type() == 'cpu'


class TestSessionOptionsFactory:
    """Tests for SessionOptionsFactory."""

    def test_factory_returns_integrated_strategy(self):
        """Factory should return IntegratedGPUStrategy for 'integrated' type."""
        config = {}
        factory = SessionOptionsFactory(config)

        strategy = factory.get_strategy('integrated')

        assert isinstance(strategy, IntegratedGPUStrategy)
        assert strategy.get_hardware_type() == 'integrated'

    def test_factory_returns_discrete_strategy(self):
        """Factory should return DiscreteGPUStrategy for 'discrete' type."""
        config = {}
        factory = SessionOptionsFactory(config)

        strategy = factory.get_strategy('discrete')

        assert isinstance(strategy, DiscreteGPUStrategy)
        assert strategy.get_hardware_type() == 'discrete'

    def test_factory_returns_cpu_strategy(self):
        """Factory should return CPUStrategy for 'cpu' type."""
        config = {}
        factory = SessionOptionsFactory(config)

        strategy = factory.get_strategy('cpu')

        assert isinstance(strategy, CPUStrategy)
        assert strategy.get_hardware_type() == 'cpu'

    def test_factory_defaults_to_cpu_for_unknown_type(self):
        """Factory should default to CPUStrategy for unknown hardware type."""
        config = {}
        factory = SessionOptionsFactory(config)

        strategy = factory.get_strategy('quantum_processor')

        assert isinstance(strategy, CPUStrategy)
        assert strategy.get_hardware_type() == 'cpu'
