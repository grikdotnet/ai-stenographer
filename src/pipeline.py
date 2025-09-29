import queue
import signal
import sys
import onnx_asr
from typing import List, Any
import tkinter as tk
from tkinter import scrolledtext

from .AudioSource import AudioSource
from .Windower import Windower
from .Recognizer import Recognizer
from .TextMatcher import TextMatcher
from .TwoStageDisplayHandler import TwoStageDisplayHandler
from .GuiWindow import create_stt_window, run_gui_loop

class STTPipeline:
    def __init__(self, model_path: str = "./models/parakeet", verbose: bool = False, window_duration: float = 2.0, step_duration: float = 1.0) -> None:
        # Create queues
        self.chunk_queue: queue.Queue = queue.Queue(maxsize=100)
        self.window_queue: queue.Queue = queue.Queue(maxsize=50)
        self.text_queue: queue.Queue = queue.Queue(maxsize=50)
        self.final_queue: queue.Queue = queue.Queue(maxsize=50)
        self.partial_queue: queue.Queue = queue.Queue(maxsize=50)

        # Load recognition model
        self.model: Any = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", model_path)

        # Create GUI window
        self.root: tk.Tk
        self.text_widget: scrolledtext.ScrolledText
        self.root, self.text_widget = create_stt_window()

        # Create components with consistent parameters
        self.audio_source: AudioSource = AudioSource(self.chunk_queue)
        self.windower: Windower = Windower(self.chunk_queue, self.window_queue, window_duration=window_duration, step_duration=step_duration, verbose=verbose)
        self.recognizer: Recognizer = Recognizer(self.window_queue, self.text_queue, self.model, verbose=verbose)
        self.text_matcher: TextMatcher = TextMatcher(self.text_queue, self.final_queue, self.partial_queue, verbose=verbose)
        self.display_handler: TwoStageDisplayHandler = TwoStageDisplayHandler(self.final_queue, self.partial_queue, self.text_widget, self.root)

        # All components
        self.components: List[Any] = [
            self.audio_source, self.windower, self.recognizer,
            self.text_matcher, self.display_handler
        ]
    
    def start(self) -> None:
        print("Starting STT Pipeline...")
        for component in self.components:
            component.start()
        print("Pipeline running. Press Ctrl+C to stop.")

    def stop(self) -> None:
        print("\nStopping pipeline...")
        for component in self.components:
            component.stop()
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