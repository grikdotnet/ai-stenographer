import queue
import signal
import sys
import logging
import onnx_asr
import json
from pathlib import Path
from typing import List, Any, Dict
import tkinter as tk
from tkinter import scrolledtext

from .AudioSource import AudioSource
from .AdaptiveWindower import AdaptiveWindower
from .Recognizer import Recognizer
from .TextMatcher import TextMatcher
from .GuiWindow import GuiWindow, create_stt_window, run_gui_loop
from .VoiceActivityDetector import VoiceActivityDetector
from .ExecutionProviderManager import ExecutionProviderManager

class STTPipeline:
    def __init__(self, model_path: str = "./models/parakeet", models_dir: Path = None, verbose: bool = False, window_duration: float = 2.0, step_duration: float = 1.0, config_path: str = "./config/stt_config.json") -> None:
        # Load configuration
        self.config: Dict = self._load_config(config_path)

        # Track if pipeline is stopped (for idempotent stop())
        self._is_stopped: bool = False

        # Create queues - single chunk_queue for AudioSegments (preliminary and finalized)
        self.chunk_queue: queue.Queue = queue.Queue(maxsize=100)
        self.text_queue: queue.Queue = queue.Queue(maxsize=50)

        # Create execution provider manager for hardware acceleration
        self.execution_provider_manager: ExecutionProviderManager = ExecutionProviderManager(self.config)
        providers = self.execution_provider_manager.build_provider_list()

        # Log selected provider
        device_info = self.execution_provider_manager.get_device_info()
        logging.info(f"Using execution provider: {device_info['selected']} ({device_info['provider']})")

        # Load recognition model with selected providers
        self.model: Any = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3",
            model_path,
            providers=providers
        )

        # Create GUI window
        self.root: tk.Tk
        self.text_widget: scrolledtext.ScrolledText
        self.root, self.text_widget = create_stt_window()

        # Create GuiWindow instance for TextMatcher
        self.gui_window: GuiWindow = GuiWindow(self.text_widget, self.root)

        # Determine absolute path for VAD model
        if models_dir is None:
            models_dir = Path("./models")
        vad_model_path = models_dir / "silero_vad" / "silero_vad.onnx"

        # Create VAD instance
        self.vad: VoiceActivityDetector = VoiceActivityDetector(
            config=self.config,
            model_path=vad_model_path,
            verbose=verbose
        )

        # Create components with unified architecture
        # AdaptiveWindower aggregates preliminary segments into finalized windows
        self.adaptive_windower: AdaptiveWindower = AdaptiveWindower(
            chunk_queue=self.chunk_queue,
            config=self.config,
            verbose=verbose
        )

        # AudioSource emits preliminary segments and sends to windower
        self.audio_source: AudioSource = AudioSource(
            chunk_queue=self.chunk_queue,
            vad=self.vad,
            windower=self.adaptive_windower,
            config=self.config,
            verbose=verbose
        )

        # Recognizer processes both preliminary and finalized segments
        self.recognizer: Recognizer = Recognizer(
            chunk_queue=self.chunk_queue,
            text_queue=self.text_queue,
            model=self.model,
            verbose=verbose
        )

        # TextMatcher now handles display directly via gui_window
        self.text_matcher: TextMatcher = TextMatcher(self.text_queue, self.gui_window, verbose=verbose)

        # All components (AdaptiveWindower has no thread - called directly by AudioSource)
        # TwoStageDisplayHandler removed - functionality merged into TextMatcher + GuiWindow
        self.components: List[Any] = [
            self.audio_source, self.recognizer,
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

    def start(self) -> None:
        logging.info("Starting STT Pipeline...")
        for component in self.components:
            component.start()
        logging.info("Pipeline running. Press Ctrl+C to stop.")

    def stop(self) -> None:
        """Stop all pipeline components.

        Stops components in proper order:
        1. AudioSource (stop capturing audio)
        2. Recognizer (stop processing audio)
        3. TextMatcher finalization (flush pending text)
        4. TextMatcher (stop processing)

        This method is idempotent - safe to call multiple times.
        """
        # Check if already stopped
        if self._is_stopped:
            return

        self._is_stopped = True

        logging.info("Stopping pipeline...")

        # Stop audio capture and recognition
        self.audio_source.stop()
        self.recognizer.stop()

        # Finalize any pending partial text before stopping TextMatcher
        logging.info("Finalizing pending text...")
        self.text_matcher.finalize_pending()

        # Stop text processing
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

        # Register window close handler
        self.root.protocol("WM_DELETE_WINDOW", on_window_close)

        try:
            # Run GUI main loop
            run_gui_loop(self.root)
        except KeyboardInterrupt:
            self.stop()
        finally:
            # Ensure pipeline is stopped even if GUI exits unexpectedly
            self.stop()