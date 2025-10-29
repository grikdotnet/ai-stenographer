import queue
import signal
import sys
import logging
import onnx_asr
import onnxruntime as rt
import json
from pathlib import Path
from typing import List, Any, Dict
import tkinter as tk
from tkinter import scrolledtext

from .AudioSource import AudioSource
from .SoundPreProcessor import SoundPreProcessor
from .AdaptiveWindower import AdaptiveWindower
from .Recognizer import Recognizer
from .TextMatcher import TextMatcher
from .GuiWindow import GuiWindow, create_stt_window, run_gui_loop
from .VoiceActivityDetector import VoiceActivityDetector
from .ExecutionProviderManager import ExecutionProviderManager
from .SessionOptionsFactory import SessionOptionsFactory

class STTPipeline:
    def __init__(self, model_path: str = "./models/parakeet", models_dir: Path = None, verbose: bool = False, window_duration: float = 2.0, step_duration: float = 1.0, config_path: str = "./config/stt_config.json") -> None:
        """Initialize STT pipeline with hardware-accelerated recognition.

        Strategy pattern: GPU type detection selects optimal session configuration
        (integrated GPU, discrete GPU, or CPU) with hardware-specific optimizations.
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

        logging.info("Loading FP16 Parakeet models...")
        self.model: Any = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3",
            model_path,
            quantization='fp16',
            providers=providers,
            sess_options=sess_options,
            cpu_preprocessing=False
        )
        logging.info("FP16 models loaded successfully")

        self.root: tk.Tk
        self.text_widget: scrolledtext.ScrolledText
        self.root, self.text_widget = create_stt_window()

        self.gui_window: GuiWindow = GuiWindow(self.text_widget, self.root)

        if models_dir is None:
            models_dir = Path("./models")
        vad_model_path = models_dir / "silero_vad" / "silero_vad.onnx"

        # Create VAD (unchanged)
        self.vad: VoiceActivityDetector = VoiceActivityDetector(
            config=self.config,
            model_path=vad_model_path,
            verbose=verbose
        )

        # Create AdaptiveWindower (renamed queue)
        self.adaptive_windower: AdaptiveWindower = AdaptiveWindower(
            speech_queue=self.speech_queue,
            config=self.config,
            verbose=verbose
        )

        # Create SoundPreProcessor (NEW)
        self.sound_preprocessor: SoundPreProcessor = SoundPreProcessor(
            chunk_queue=self.chunk_queue,
            speech_queue=self.speech_queue,
            vad=self.vad,
            windower=self.adaptive_windower,
            config=self.config,
            verbose=verbose
        )

        # Create simplified AudioSource
        self.audio_source: AudioSource = AudioSource(
            chunk_queue=self.chunk_queue,
            config=self.config,
            verbose=verbose
        )

        # Create Recognizer (renamed queue)
        self.recognizer: Recognizer = Recognizer(
            speech_queue=self.speech_queue,
            text_queue=self.text_queue,
            model=self.model,
            verbose=verbose
        )

        self.text_matcher: TextMatcher = TextMatcher(self.text_queue, self.gui_window, verbose=verbose)

        # Update components list
        self.components: List[Any] = [
            self.audio_source,
            self.sound_preprocessor,  # NEW
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
        """Warm up model to trigger shader compilation (DirectML/CUDA).

        DirectML compiles GPU shaders on first inference, causing ~500ms delay.
        This method runs a dummy inference during startup to pre-compile shaders.
        Skips warm-up for CPU providers (no shader compilation needed).
        """
        if self.execution_provider_manager.selected_provider == 'CPU':
            return

        import numpy as np
        import time

        dummy_audio = np.zeros(48000, dtype=np.float32)

        logging.info("Warming up GPU model (shader compilation)...")
        start = time.perf_counter()

        try:
            _ = self.model.recognize(dummy_audio)
            elapsed = (time.perf_counter() - start) * 1000
            logging.info(f"Warm-up complete: {elapsed:.0f}ms")
        except Exception as e:
            logging.warning(f"Warm-up failed: {e}")

    def start(self) -> None:
        logging.info("Starting STT Pipeline...")
        self._warmup_model()

        for component in self.components:
            component.start()
        logging.info("Pipeline running. Press Ctrl+C to stop.")

    def stop(self) -> None:
        """Stop all pipeline components.

        Stops components in proper order:
        1. AudioSource (stop capturing audio)
        2. SoundPreProcessor (stop processing, flushes pending segments)
        3. Recognizer (stop processing audio)
        4. TextMatcher finalization (flush pending text)
        5. TextMatcher (stop processing)

        This method is idempotent - safe to call multiple times.
        """
        if self._is_stopped:
            return

        self._is_stopped = True

        logging.info("Stopping pipeline...")

        # 1. Stop audio capture
        self.audio_source.stop()

        # 2. Stop preprocessing (flushes pending segments)
        self.sound_preprocessor.stop()

        # 3. Stop recognition
        self.recognizer.stop()

        # 4. Finalize text
        logging.info("Finalizing pending text...")
        self.text_matcher.finalize_pending()

        # 5. Stop text processing
        self.text_matcher.stop()

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