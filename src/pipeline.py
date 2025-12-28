import queue
import signal
import sys
import logging
import onnx_asr
import onnxruntime as rt
import json
from pathlib import Path
from typing import List, Any, Dict, TYPE_CHECKING
import tkinter as tk

from .AudioSource import AudioSource
from .FileAudioSource import FileAudioSource
from .SoundPreProcessor import SoundPreProcessor
from .AdaptiveWindower import AdaptiveWindower
from .Recognizer import Recognizer
from .TextMatcher import TextMatcher
from .GuiWindow import GuiWindow, create_stt_window, run_gui_loop
from .VoiceActivityDetector import VoiceActivityDetector
from .ExecutionProviderManager import ExecutionProviderManager
from .SessionOptionsFactory import SessionOptionsFactory
from .ApplicationState import ApplicationState

if TYPE_CHECKING:
    from onnx_asr.adapters import TimestampedResultsAsrAdapter

class STTPipeline:
    def __init__(self, model_path: str = "./models/parakeet", models_dir: Path = None, verbose: bool = False, window_duration: float = 2.0, config_path: str = "./config/stt_config.json", input_file: str = None) -> None:
        """Initialize STT pipeline with hardware-accelerated recognition.

        Strategy pattern: GPU type detection selects optimal session configuration
        (integrated GPU, discrete GPU, or CPU) with hardware-specific optimizations.

        Args:
            model_path: Path to Parakeet model directory
            models_dir: Path to models directory (for VAD)
            verbose: Enable verbose logging
            window_duration: Window duration for recognition (seconds)
            config_path: Path to configuration JSON file
            input_file: Optional path to WAV file for testing (instead of microphone)
        """
        self.config: Dict = self._load_config(config_path)
        self._is_stopped: bool = False

        # Create queues
        self.chunk_queue: queue.Queue = queue.Queue(maxsize=200)      # Raw audio chunks
        self.speech_queue: queue.Queue = queue.Queue(maxsize=200)     # AudioSegments (prelim + final)
        self.text_queue: queue.Queue = queue.Queue(maxsize=50)        # RecognitionResults

        self.execution_provider_manager: ExecutionProviderManager = ExecutionProviderManager(self.config)
        providers = self.execution_provider_manager.build_provider_list()

        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

        gpu_type = self.execution_provider_manager.detect_gpu_type()
        factory = SessionOptionsFactory(self.config)
        strategy = factory.get_strategy(gpu_type)
        strategy.configure_session_options(sess_options)

        base_model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3",
            model_path,
            quantization='fp16',
            providers=providers,
            sess_options=sess_options,
        )
        self.model: TimestampedResultsAsrAdapter = base_model.with_timestamps()
        logging.info("FP16 models loaded successfully (timestamped mode)")

        # Create ApplicationState first (needed by both pipeline and GUI)
        self.app_state: ApplicationState = ApplicationState(config=self.config)

        # Create complete GUI setup (window + gui_window + pause controller + control panel)
        self.root: tk.Tk
        self.gui_window: GuiWindow
        self.root, self.gui_window = create_stt_window(self.config, self.app_state)

        if models_dir is None:
            models_dir = Path("./models")
        vad_model_path = models_dir / "silero_vad" / "silero_vad.onnx"

        self.vad: VoiceActivityDetector = VoiceActivityDetector(
            config=self.config,
            model_path=vad_model_path,
            verbose=verbose
        )

        self.adaptive_windower: AdaptiveWindower = AdaptiveWindower(
            speech_queue=self.speech_queue,
            config=self.config,
            verbose=verbose
        )

        self.sound_preprocessor: SoundPreProcessor = SoundPreProcessor(
            chunk_queue=self.chunk_queue,
            speech_queue=self.speech_queue,
            vad=self.vad,
            windower=self.adaptive_windower,
            config=self.config,
            app_state=self.app_state,
            verbose=verbose
        )

        # Create AudioSource - microphone or file
        if input_file:
            logging.info(f"Using file input: {input_file}")
            self.audio_source: FileAudioSource = FileAudioSource(
                chunk_queue=self.chunk_queue,
                config=self.config,
                file_path=input_file,
                verbose=verbose
            )
        else:
            logging.info("Using microphone input")
            self.audio_source: AudioSource = AudioSource(
                chunk_queue=self.chunk_queue,
                config=self.config,
                app_state=self.app_state,
                verbose=verbose
            )

        sample_rate = self.config['audio']['sample_rate']
        self.recognizer: Recognizer = Recognizer(
            speech_queue=self.speech_queue,
            text_queue=self.text_queue,
            model=self.model,
            sample_rate=sample_rate,
            app_state=self.app_state,
            verbose=verbose
        )

        self.text_matcher: TextMatcher = TextMatcher(
            self.text_queue,
            self.gui_window,
            app_state=self.app_state,
            verbose=verbose
        )

        # Update components list
        self.components: List[Any] = [
            self.audio_source,
            self.sound_preprocessor,
            self.recognizer,
            self.text_matcher
        ]

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file.

        Args:
            config_path: Path to stt_config.json

        Returns:
            Configuration dictionary
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, 'r') as f:
            return json.load(f)

    def _warmup_model(self) -> None:
        """Warm up model to trigger shader compilation.

        DirectML compiles GPU shaders on first inference.
        This method runs a dummy inference during startup to pre-compile shaders.
        Skips for CPU providers.
        """
        if self.execution_provider_manager.selected_provider == 'CPU':
            return

        import numpy as np
        import time

        dummy_audio = np.zeros(48000, dtype=np.float32)

        start = time.perf_counter()

        try:
            _ = self.model.recognize(dummy_audio)
            elapsed = (time.perf_counter() - start) * 1000
        except Exception as e:
            logging.warning(f"Warm-up failed: {e}")

    def start(self) -> None:
        logging.info("Starting STT Pipeline...")
        self._warmup_model()

        for component in self.components:
            component.start()

        # Set state to running after all components started
        self.app_state.set_state('running')
        logging.info("Pipeline running. Press Ctrl+C to stop.")

    def stop(self) -> None:
        """Stop all pipeline components via observer pattern.

        Sets ApplicationState to 'shutdown', which triggers all component
        observers to stop themselves
        """
        if self._is_stopped:
            return

        self._is_stopped = True

        logging.info("Stopping pipeline...")
        self.app_state.set_state('shutdown')  # Triggers all observers
        logging.info("Pipeline stopped.")

    def run(self) -> None:
        self.start()

        def signal_handler(sig: int, frame: Any) -> None:
            self.stop()
            sys.exit(0)

        def on_window_close() -> None:
            """Handle window close event - stop pipeline gracefully."""
            self.stop()
            try:
                self.root.destroy()
            except tk.TclError:
                pass

        signal.signal(signal.SIGINT, signal_handler)
        self.root.protocol("WM_DELETE_WINDOW", on_window_close)

        try:
            run_gui_loop(self.root)
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.stop()