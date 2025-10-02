# src/VoiceActivityDetector.py
import onnxruntime
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


class VoiceActivityDetector:
    """Voice Activity Detection using Silero VAD ONNX model.

    Detects speech vs. silence in audio streams, providing segment boundaries
    and speech probability scores. Uses the Silero VAD ONNX model loaded with
    ONNX Runtime for accurate real-time detection.

    Args:
        config: Configuration dictionary containing VAD and audio parameters
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vad_config = config['vad']
        self.audio_config = config['audio']

        self.sample_rate: int = self.audio_config['sample_rate']
        self.threshold: float = self.vad_config['threshold']
        self.frame_duration_ms: int = self.vad_config['frame_duration_ms']
        self.min_speech_duration_ms: int = self.vad_config['min_speech_duration_ms']
        self.silence_timeout_ms: int = self.vad_config.get('silence_timeout_ms', 300)
        self.max_speech_duration_ms: int = self.vad_config.get('max_speech_duration_ms', 30000)

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

        Loads the model from the path specified in config using ONNX Runtime.
        Initializes the model state tensor for stateful inference.
        """
        model_path = Path(self.vad_config['model_path'])

        if not model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {model_path}. "
                f"Run 'python download_model.py' to download it."
            )

        try:
            self.model = onnxruntime.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )

            # Initialize model state (2, batch_size, 128)
            # batch_size = 1 for single audio stream
            self.model_state = np.zeros((2, 1, 128), dtype=np.float32)

        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD ONNX model: {e}")


    def reset_state(self) -> None:
        """Reset internal state for processing new audio stream.

        Clears accumulated speech buffer, resets timing information, and
        resets the ONNX model's internal state.
        Call this between processing different audio files or streams.
        """
        self.speech_buffer: List[np.ndarray] = []
        self.silence_frames: int = 0
        self.is_speech_active: bool = False
        self.speech_start_time: float = 0.0
        self.current_time: float = 0.0

        # Reset ONNX model state
        if self.model_state is not None:
            self.model_state = np.zeros((2, 1, 128), dtype=np.float32)


    def process_frame(self, audio_frame: np.ndarray) -> Dict[str, Any]:
        """Process a single audio frame and return speech detection result.

        Analyzes a frame of audio (typically 32ms) and returns whether it
        contains speech along with confidence probability. Maintains internal
        state across frames for improved accuracy.

        Args:
            audio_frame: Audio data as float32 numpy array

        Returns:
            Dictionary containing:
                - is_speech (bool): True if speech detected
                - speech_probability (float): Confidence score 0.0-1.0
                - timestamp (float): Current time in seconds
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
        audio_input = audio_frame.astype(np.float32).reshape(1, -1)
        sr_input = np.array([self.sample_rate], dtype=np.int64)

        # Run inference
        ort_outputs = self.model.run(
            None,
            {
                'input': audio_input,
                'state': self.model_state,
                'sr': sr_input
            }
        )

        # Extract outputs
        speech_prob = float(ort_outputs[0][0][0])  # Shape: [1, 1]
        self.model_state = ort_outputs[1]  # Update state for next frame

        # Determine if speech based on threshold
        is_speech = speech_prob > self.threshold

        result = {
            'is_speech': is_speech,
            'speech_probability': speech_prob,
            'timestamp': self.current_time
        }

        return result


    def process_audio(self, audio: np.ndarray, reset_state: bool = True) -> List[Dict[str, Any]]:
        """Process audio and return detected speech segments.

        Processes audio frame-by-frame, detecting speech boundaries and
        returning segments with timing information and audio data.

        Args:
            audio: Audio data as float32 numpy array (can be single frame or longer)
            reset_state: Whether to reset VAD state before processing (default True)
                        Set to False for streaming mode where state persists across calls

        Returns:
            List of segment dictionaries, each containing:
                - is_speech (bool): True for speech, False for silence
                - start_time (float): Segment start in seconds (relative to stream start)
                - end_time (float): Segment end in seconds
                - duration (float): Segment duration in seconds
                - data (np.ndarray): Audio data for this segment
        """
        if reset_state:
            self.reset_state()

        segments: List[Dict[str, Any]] = []

        # Process audio in frames
        for i in range(0, len(audio), self.frame_size):
            chunk = audio[i:i + self.frame_size]

            # Skip incomplete frames at the end
            if len(chunk) < self.frame_size:
                break

            result = self.process_frame(chunk)
            frame_time = i / self.sample_rate  # Relative time within this chunk

            if result['is_speech']:
                self._handle_speech_frame(chunk, segments, frame_time)
            else:
                self._handle_silence_frame(chunk, segments, frame_time)

        # Update current time for next call
        self.current_time += len(audio) / self.sample_rate

        # Finalize any remaining speech segment if in batch mode
        if reset_state and self.is_speech_active and len(self.speech_buffer) > 0:
            self._finalize_chunk(segments)

        return segments


    def _handle_speech_frame(self, frame: np.ndarray, segments: List[Dict[str, Any]], frame_time: float) -> None:
        """Handle a frame detected as speech.

        Accumulates speech frames into buffer, starts new segment if needed,
        and enforces maximum duration limit by splitting long segments.

        Args:
            frame: Audio frame data
            segments: List to append completed segments to
            frame_time: Relative time of this frame within current chunk
        """
        self.silence_frames = 0

        if not self.is_speech_active:
            # Start new speech segment
            self.is_speech_active = True
            self.speech_start_time = self.current_time + frame_time
            self.speech_buffer = [frame]
        else:
            # Continue accumulating speech
            self.speech_buffer.append(frame)

            # Check if exceeded max duration
            current_duration_ms = len(self.speech_buffer) * self.frame_duration_ms
            if current_duration_ms >= self.max_speech_duration_ms:
                self._finalize_chunk(segments)
                # Start new segment immediately
                self.is_speech_active = True
                self.speech_start_time = self.current_time + frame_time
                self.speech_buffer = [frame]


    def _handle_silence_frame(self, frame: np.ndarray, segments: List[Dict[str, Any]], frame_time: float) -> None:
        """Handle a frame detected as silence.

        Tracks consecutive silence frames and finalizes speech segment
        when silence timeout is exceeded.

        Args:
            frame: Audio frame data
            segments: List to append completed segments to
            frame_time: Relative time of this frame within current chunk
        """
        if self.is_speech_active:
            self.silence_frames += 1
            silence_duration_ms = self.silence_frames * self.frame_duration_ms

            if silence_duration_ms >= self.silence_timeout_ms:
                # Silence timeout reached - finalize segment
                self._finalize_chunk(segments)


    def _finalize_chunk(self, segments: List[Dict[str, Any]]) -> None:
        """Finalize accumulated speech segment and add to segments list.

        Applies minimum duration filter and creates segment dictionary
        with timing and audio data.

        Args:
            segments: List to append completed segment to
        """
        if not self.speech_buffer:
            return

        # Check minimum duration
        duration_ms = len(self.speech_buffer) * self.frame_duration_ms
        if duration_ms < self.min_speech_duration_ms:
            # Too short, discard
            self.is_speech_active = False
            self.speech_buffer = []
            self.silence_frames = 0
            return

        # Create segment
        audio_data = np.concatenate(self.speech_buffer)
        end_time = self.speech_start_time + (len(audio_data) / self.sample_rate)

        segment = {
            'is_speech': True,
            'start_time': self.speech_start_time,
            'end_time': end_time,
            'duration': end_time - self.speech_start_time,
            'data': audio_data
        }

        segments.append(segment)

        # Reset state
        self.is_speech_active = False
        self.speech_buffer = []
        self.silence_frames = 0


    def flush(self) -> List[Dict[str, Any]]:
        """Force finalization of any pending speech segment.

        Useful for end-of-stream scenarios where no more audio will be provided.
        In streaming mode, call this when the audio stream ends to emit the final segment.

        Returns:
            List of finalized segments (0 or 1 segment)
        """
        segments: List[Dict[str, Any]] = []
        if self.is_speech_active and len(self.speech_buffer) > 0:
            self._finalize_chunk(segments)
        return segments
