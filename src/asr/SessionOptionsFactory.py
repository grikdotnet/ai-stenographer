"""Factory for creating hardware-specific session option strategies.

This module implements the Factory pattern for creating SessionOptionsStrategy instances
based on detected hardware type. Provides centralized strategy instantiation and lookup.
"""
from typing import Dict, Any
from .SessionOptionsStrategy import (
    SessionOptionsStrategy,
    IntegratedGPUStrategy,
    DiscreteGPUStrategy,
    CPUStrategy
)


class SessionOptionsFactory:
    """Factory for creating hardware-specific session option strategies.

    Maintains a registry of strategy instances and provides lookup by hardware type.
    Defaults to CPU strategy for unknown hardware types (safe fallback).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize factory with configuration.

        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self._strategies = {
            'integrated': IntegratedGPUStrategy(config),
            'discrete': DiscreteGPUStrategy(config),
            'cpu': CPUStrategy(config)
        }

    def get_strategy(self, gpu_type: str) -> SessionOptionsStrategy:
        """Get strategy for given GPU type.

        Args:
            gpu_type: One of 'integrated', 'discrete', 'cpu', or unknown type

        Returns:
            Appropriate SessionOptionsStrategy instance (defaults to CPU for unknown types)
        """
        return self._strategies.get(gpu_type, self._strategies['cpu'])
