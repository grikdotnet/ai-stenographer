# src/AudioSource.py
import sounddevice as sd
import queue
import numpy as np
import time
from typing import Optional, Dict, Any, List
from src.VoiceActivityDetector import VoiceActivityDetector

class AudioSource:
    """Captures audio from microphone and processes through VAD.

    AudioSource uses sounddevice to capture real-time audio from the microphone,
    processes it through Voice Activity Detection, and emits variable-length
    speech segments to chunk_queue.

    Args:
        chunk_queue: Queue to receive VAD segments
        config: Configuration dictionary loaded from stt_config.json (required)
        emit_silence: Whether to emit silence markers
    """

    def __init__(self,
                 chunk_queue: queue.Queue,
                 config: Optional[Dict[str, Any]] = None,
                 emit_silence: bool = False):

        if config is None:
            raise ValueError("config is required - must be loaded from stt_config.json")

        self.chunk_queue: queue.Queue = chunk_queue

        # Extract values from config
        self.sample_rate: int = config['audio']['sample_rate']
        chunk_duration = config['audio']['chunk_duration']

        self.chunk_size: int = int(self.sample_rate * chunk_duration)
        self.is_running: bool = False
        self.stream: Optional[sd.InputStream] = None
        self.chunk_id_counter: int = 0

        # VAD integration (always enabled)
        self.emit_silence: bool = emit_silence
        self.vad: VoiceActivityDetector = VoiceActivityDetector(config)

        # Verify chunk_duration matches VAD frame_duration
        expected_chunk_duration = config['vad']['frame_duration_ms'] / 1000.0
        if abs(chunk_duration - expected_chunk_duration) > 0.001:
            raise ValueError(
                f"chunk_duration ({chunk_duration}) must match VAD frame_duration "
                f"({expected_chunk_duration})"
            )

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Callback from sounddevice that processes incoming audio data.

        This callback is called automatically by sounddevice when audio data
        is available from the microphone. Processes through VAD and emits
        speech segments.

        Args:
            indata: Input audio data as numpy array
            frames: Number of audio frames
            time_info: Timing information from sounddevice
            status: Status information from sounddevice
        """
        if status:
            print(f"Audio error: {status}")

        # Convert to float32 mono
        audio_float: np.ndarray = indata[:, 0].astype(np.float32)
        current_time = time.time()

        # Process through VAD
        self.process_chunk_with_vad(audio_float, current_time)


    def process_chunk_with_vad(self, audio_chunk: np.ndarray, timestamp: float) -> None:
        """Process audio chunk through VAD frame by frame.

        Since chunk_duration matches VAD frame_duration (32ms), each chunk
        is processed directly as a single VAD frame. VAD maintains internal state
        across calls and emits complete speech segments when detected.

        Args:
            audio_chunk: Audio data to process (32ms frame)
            timestamp: Timestamp of this chunk
        """
        # Process chunk as single VAD frame in streaming mode (reset_state=False)
        # VAD maintains state across calls to track ongoing speech
        segments = self.vad.process_audio(audio_chunk, reset_state=False)

        # Emit detected segments
        for segment in segments:
            segment['timestamp'] = segment['start_time']
            self.chunk_queue.put(segment)

    def start(self) -> None:
        """Start capturing audio from the microphone.

        Creates and starts a sounddevice InputStream that will begin
        capturing audio and calling the audio_callback method.
        """
        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.chunk_size
        )
        self.stream.start()

    def stop(self) -> None:
        """Stop capturing audio from the microphone.

        Stops the sounddevice InputStream and flushes any pending VAD segments.
        """
        self.is_running = False
        if self.stream:
            self.stream.stop()

        # Flush any pending VAD segments
        segments = self.vad.flush()
        for segment in segments:
            self.chunk_queue.put(segment)


    def flush_vad(self) -> None:
        """Manually flush any pending VAD segments.

        Useful in testing or when forcing finalization of the current segment.
        """
        segments = self.vad.flush()
        for segment in segments:
            self.chunk_queue.put(segment)