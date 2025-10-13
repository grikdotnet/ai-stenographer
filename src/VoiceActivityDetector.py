# src/VoiceActivityDetector.py
import onnxruntime
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


class VoiceActivityDetector:
    """Voice Activity Detection using Silero VAD ONNX model.

    Detects speech vs. silence in audio streams, providing segment boundaries
    and speech probability scores. Uses the Silero VAD ONNX model loaded with
    ONNX Runtime for accurate real-time detection.

    Args:
        config: Configuration dictionary containing VAD and audio parameters
        model_path: Absolute path to the Silero VAD ONNX model file
        verbose: Enable detailed logging for debugging (default: False)
    """

    def __init__(self, config: Dict[str, Any], model_path: Path, verbose: bool = False):
        self.config = config
        self.vad_config = config['vad']
        self.audio_config = config['audio']
        self.model_path = model_path
        self.verbose = verbose

        self.sample_rate: int = self.audio_config['sample_rate']
        self.threshold: float = self.vad_config['threshold']
        self.frame_duration_ms: int = self.vad_config['frame_duration_ms']

        # Calculate frame size in samples
        self.frame_size: int = int(self.sample_rate * self.frame_duration_ms / 1000)

        # Load Silero VAD ONNX model
        self.model: Optional[onnxruntime.InferenceSession] = None
        self.model_state: Optional[np.ndarray] = None
        self._load_model()

        # State tracking for segment detection
        self.reset_state()


    def _load_model(self) -> None:
        """Load Silero VAD ONNX model from local path.

        Loads the model from the absolute path provided during initialization.
        Initializes the model state tensor for stateful inference.
        Applies optimizations for reduced CPU usage in streaming mode.
        """
        model_path = self.model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {model_path}. "
                f"Run 'python download_model.py' to download it."
            )

        try:
            # Configure session options for optimal performance
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 1  # Single thread for small model
            sess_options.inter_op_num_threads = 1

            self.model = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )

            # Initialize model state (2, batch_size, 128)
            # batch_size = 1 for single audio stream
            self.model_state = np.zeros((2, 1, 128), dtype=np.float32)

            # Cache sample rate input to avoid repeated allocations
            self.sr_input = np.array([self.sample_rate], dtype=np.int64)

        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD ONNX model: {e}")


    def reset_state(self) -> None:
        """Reset ONNX model internal state.

        Clears the model's temporal context. Call this between processing
        different audio files or streams to ensure clean state.
        """
        if self.model_state is not None:
            self.model_state = np.zeros((2, 1, 128), dtype=np.float32)


    def process_frame(self, audio_frame: np.ndarray) -> Dict[str, Any]:
        """Process a single audio frame and return speech detection result.

        Analyzes a frame of audio (typically 32ms) and returns whether it
        contains speech along with confidence probability. Maintains ONNX model
        state across frames for temporal context.

        Args:
            audio_frame: Audio data as float32 numpy array

        Returns:
            Dictionary containing:
                - is_speech (bool): True if speech detected
                - speech_probability (float): Confidence score 0.0-1.0
        """
        # Ensure correct size
        if len(audio_frame) != self.frame_size:
            # Pad or truncate to frame size
            if len(audio_frame) < self.frame_size:
                audio_frame = np.pad(audio_frame, (0, self.frame_size - len(audio_frame)))
            else:
                audio_frame = audio_frame[:self.frame_size]

        # Prepare inputs for ONNX model
        # Input shape: [batch_size, audio_length]
        # Avoid unnecessary copy if already float32
        if audio_frame.dtype != np.float32:
            audio_frame = audio_frame.astype(np.float32)
        audio_input = audio_frame.reshape(1, -1)

        # Run inference with cached sr_input
        ort_outputs = self.model.run(
            None,
            {
                'input': audio_input,
                'state': self.model_state,
                'sr': self.sr_input
            }
        )

        # Extract outputs
        speech_prob = float(ort_outputs[0][0][0])  # Shape: [1, 1]
        self.model_state = ort_outputs[1]  # Update state for next frame

        # Determine if speech based on threshold
        is_speech = speech_prob > self.threshold

        return {
            'is_speech': is_speech,
            'speech_probability': speech_prob
        }
