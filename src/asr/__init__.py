"""ASR subsystem - speech recognition and neural model handling."""
from src.asr.Recognizer import Recognizer
from src.asr.AdaptiveWindower import AdaptiveWindower
from src.asr.VoiceActivityDetector import VoiceActivityDetector
from src.asr.SessionOptionsFactory import SessionOptionsFactory
from src.asr.SessionOptionsStrategy import (
    SessionOptionsStrategy,
    IntegratedGPUStrategy,
    DiscreteGPUStrategy,
    CPUStrategy
)
from src.asr.ModelManager import ModelManager
from src.asr.ExecutionProviderManager import ExecutionProviderManager
from src.asr.DownloadProgressReporter import (
    DownloadProgressReporter,
    ProgressAggregator,
    create_bound_progress_reporter
)

__all__ = [
    'Recognizer',
    'AdaptiveWindower',
    'VoiceActivityDetector',
    'SessionOptionsFactory',
    'SessionOptionsStrategy',
    'IntegratedGPUStrategy',
    'DiscreteGPUStrategy',
    'CPUStrategy',
    'ModelManager',
    'ExecutionProviderManager',
    'DownloadProgressReporter',
    'ProgressAggregator',
    'create_bound_progress_reporter'
]
