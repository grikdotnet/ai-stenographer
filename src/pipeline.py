import queue
import signal
import sys
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

class STTPipeline:
    def __init__(self, model_path: str = "./models/parakeet", verbose: bool = False, window_duration: float = 2.0, step_duration: float = 1.0, config_path: str = "./config/stt_config.json") -> None:
        # Load configuration
        self.config: Dict = self._load_config(config_path)

        # Create queues - single chunk_queue for AudioSegments (preliminary and finalized)
        self.chunk_queue: queue.Queue = queue.Queue(maxsize=100)
        self.text_queue: queue.Queue = queue.Queue(maxsize=50)

        # Load recognition model
        self.model: Any = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", model_path)

        # Create GUI window
        self.root: tk.Tk
        self.text_widget: scrolledtext.ScrolledText
        self.root, self.text_widget = create_stt_window()

        # Create GuiWindow instance for TextMatcher
        self.gui_window: GuiWindow = GuiWindow(self.text_widget, self.root)

        # Create VAD instance
        self.vad: VoiceActivityDetector = VoiceActivityDetector(
            config=self.config,
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
        print("Starting STT Pipeline...")
        for component in self.components:
            component.start()
        print("Pipeline running. Press Ctrl+C to stop.")

    def stop(self) -> None:
        """Stop all pipeline components.

        Stops components in proper order:
        1. AudioSource (stop capturing audio)
        2. Recognizer (stop processing audio)
        3. TextMatcher finalization (flush pending text)
        4. TextMatcher (stop processing)
        """
        print("\nStopping pipeline...")

        # Stop audio capture and recognition
        self.audio_source.stop()
        self.recognizer.stop()

        # Finalize any pending partial text before stopping TextMatcher
        print("Finalizing pending text...")
        self.text_matcher.finalize_pending()

        # Stop text processing
        self.text_matcher.stop()

        print("Pipeline stopped.")

    def run(self) -> None:
        self.start()

        def signal_handler(sig: int, frame: Any) -> None:
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Run GUI main loop
            run_gui_loop(self.root)
        except KeyboardInterrupt:
            self.stop()