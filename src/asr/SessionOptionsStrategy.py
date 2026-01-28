"""Hardware-specific ONNX Runtime session configuration strategies.

This module implements the Strategy pattern for configuring ONNX Runtime session options
based on hardware type (integrated GPU, discrete GPU, or CPU). Each strategy encapsulates
the optimal configuration for its hardware architecture.

Architecture rationale:
- Integrated GPU: Shared system memory (UMA) → zero-copy, disable CPU arena
- Discrete GPU: Dedicated VRAM (PCIe) → staging buffer, enable CPU arena
- CPU: Multi-threaded execution → optimize thread counts
"""
from abc import ABC, abstractmethod
import onnxruntime as rt
from typing import Dict, Any
import logging


class SessionOptionsStrategy(ABC):
    """Base strategy for hardware-specific ONNX Runtime session configuration.

    Each concrete strategy implements optimal session options for a specific
    hardware type, following the Strategy pattern for clean separation of concerns.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy with configuration.

        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def configure_session_options(self, sess_options: rt.SessionOptions) -> None:
        """Configure session options for specific hardware type.

        Args:
            sess_options: ONNX Runtime SessionOptions to configure
        """
        pass

    @abstractmethod
    def get_hardware_type(self) -> str:
        """Return hardware type identifier.

        Returns:
            One of: 'integrated', 'discrete', 'cpu'
        """
        pass


class IntegratedGPUStrategy(SessionOptionsStrategy):
    """Strategy for integrated GPUs (Intel Iris/UHD, AMD Vega/Radeon Graphics).

    Integrated GPUs share system RAM with the CPU (Unified Memory Architecture).
    Key optimization: Zero-copy memory access by disabling CPU memory arena.

    Hardware examples:
    - Intel Iris Xe, Intel UHD Graphics
    - AMD Radeon Graphics (Ryzen APU), AMD Vega Graphics
    """

    def configure_session_options(self, sess_options: rt.SessionOptions) -> None:
        """Configure for integrated GPU: zero-copy shared memory optimizations.

        Threading: Minimal (1,1) - GPU handles parallelism internally
        Memory: Disable CPU arena - prevents double-buffering in shared memory
        DirectML: Use default settings (graph capture auto-enabled when possible)
        """
        # No CPU threading
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.enable_mem_pattern = True

        # Disable CPU arena for zero-copy shared memory
        sess_options.enable_cpu_mem_arena = False

        self.logger.info("Integrated GPU: Zero-copy shared memory optimizations")

    def get_hardware_type(self) -> str:
        """Return hardware type identifier."""
        return 'integrated'


class DiscreteGPUStrategy(SessionOptionsStrategy):
    """Strategy for discrete GPUs (NVIDIA GeForce/Quadro, AMD Radeon RX).

    Discrete GPUs have dedicated VRAM connected via PCIe bus.
    Key optimization: CPU memory arena as staging buffer for PCIe transfers.

    Hardware examples:
    - NVIDIA GeForce RTX/GTX, Quadro, Tesla
    - AMD Radeon RX series, Radeon Pro
    """

    def configure_session_options(self, sess_options: rt.SessionOptions) -> None:
        """Configure for discrete GPU: PCIe memory transfer optimizations.

        Threading: Minimal (1,1) - GPU handles parallelism internally
        Memory: Enable CPU arena - staging buffer for System RAM → PCIe → VRAM
        DirectML: Use default settings (graph capture auto-enabled when possible)
        """
        # No CPU threading
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.enable_mem_pattern = True

        # Enable CPU arena for PCIe staging buffer
        sess_options.enable_cpu_mem_arena = True

        self.logger.info("Discrete GPU: PCIe memory transfer optimizations")

    def get_hardware_type(self) -> str:
        """Return hardware type identifier."""
        return 'discrete'


class CPUStrategy(SessionOptionsStrategy):
    """Strategy for CPU-only execution.

    CPU execution relies on multi-threading for parallelism.
    Key optimization: Balance thread counts to maximize CPU utilization.

    Thread counts are capped to avoid oversubscription on high-core systems:
    - intra_op_num_threads: Max 8 (parallelizes operations like matrix multiply)
    - inter_op_num_threads: Max 4 (parallelizes independent operations)
    """

    def configure_session_options(self, sess_options: rt.SessionOptions) -> None:
        """Configure for CPU: multi-threaded execution with optimal thread counts.

        Threading: Auto-detect with caps (intra≤8, inter≤4) to avoid oversubscription
        Memory: Enable CPU arena for efficient memory management
        """
        import os

        # Auto-detect CPU count
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 4
            self.logger.warning("Could not detect CPU count, using default: 4")

        # Cap thread counts to avoid oversubscription
        sess_options.intra_op_num_threads = min(cpu_count, 8)
        sess_options.inter_op_num_threads = min(cpu_count // 2, 4)
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        self.logger.info(f"CPU: Multi-threaded execution "
                        f"(intra={sess_options.intra_op_num_threads}, "
                        f"inter={sess_options.inter_op_num_threads})")

    def get_hardware_type(self) -> str:
        """Return hardware type identifier."""
        return 'cpu'
