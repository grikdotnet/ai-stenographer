"""ASR subsystem - speech recognition and neural model handling."""
from src.asr.Recognizer import Recognizer
from src.asr.VoiceActivityDetector import VoiceActivityDetector
from src.asr.SessionOptionsFactory import SessionOptionsFactory
from src.asr.SessionOptionsStrategy import (
    SessionOptionsStrategy,
    IntegratedGPUStrategy,
    DiscreteGPUStrategy,
    CPUStrategy
)
from src.asr.ExecutionProviderManager import ExecutionProviderManager

__all__ = [
    'Recognizer',
    'VoiceActivityDetector',
    'SessionOptionsFactory',
    'SessionOptionsStrategy',
    'IntegratedGPUStrategy',
    'DiscreteGPUStrategy',
    'CPUStrategy',
    'ExecutionProviderManager',
]
