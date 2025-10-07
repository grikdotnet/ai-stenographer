# src/AudioSource.py
from __future__ import annotations
import sounddevice as sd
import queue
import numpy as np
import time
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from src.VoiceActivityDetector import VoiceActivityDetector
from src.types import AudioSegment

if TYPE_CHECKING:
    from src.AdaptiveWindower import AdaptiveWindower

class AudioSource:
    """Captures audio from microphone and processes through VAD.

    AudioSource uses sounddevice to capture real-time audio from the microphone,
    processes it through Voice Activity Detection, and emits variable-length
    speech segments to chunk_queue.

    Args:
        chunk_queue: Queue to receive AudioSegment instances (preliminary segments)
        vad: VoiceActivityDetector instance for speech detection
        windower: AdaptiveWindower instance to aggregate segments into windows
        config: Configuration dictionary loaded from stt_config.json (required)
        emit_silence: Whether to emit silence markers
    """

    def __init__(self,
                 chunk_queue: queue.Queue,
                 vad: VoiceActivityDetector,
                 windower: Optional[AdaptiveWindower] = None,
                 config: Optional[Dict[str, Any]] = None,
                 emit_silence: bool = False):

        if config is None:
            raise ValueError("config is required - must be loaded from stt_config.json")

        self.chunk_queue: queue.Queue = chunk_queue
        self.vad: VoiceActivityDetector = vad
        self.windower: Optional[AdaptiveWindower] = windower

        # Extract values from config
        self.sample_rate: int = config['audio']['sample_rate']
        chunk_duration = config['audio']['chunk_duration']

        self.chunk_size: int = int(self.sample_rate * chunk_duration)
        self.is_running: bool = False
        self.stream: Optional[sd.InputStream] = None
        self.chunk_id_counter: int = 0

        # VAD integration (always enabled)
        self.emit_silence: bool = emit_silence

        # Speech buffering state (moved from VAD)
        self.speech_buffer: List[np.ndarray] = []
        self.silence_frames: int = 0
        self.is_speech_active: bool = False
        self.speech_start_time: float = 0.0
        self.current_stream_time: float = 0.0

        # Load timing configs from config (used for buffering logic)
        self.min_speech_duration_ms: int = config['vad']['min_speech_duration_ms']
        self.silence_timeout_ms: int = config['vad']['silence_timeout_ms']
        self.max_speech_duration_ms: int = config['vad']['max_speech_duration_ms']
        self.frame_duration_ms: int = config['vad']['frame_duration_ms']

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
        """Process audio chunk through VAD and buffer based on speech detection.

        Calls VAD for speech detection on the frame, then handles buffering
        logic locally. Emits segments when silence timeout is reached or
        max duration exceeded.

        Args:
            audio_chunk: Audio data to process (32ms frame)
            timestamp: Timestamp of this chunk
        """
        # Call VAD for single frame detection
        vad_result = self.vad.process_frame(audio_chunk)

        if vad_result['is_speech']:
            self._handle_speech_frame(audio_chunk, timestamp)
        else:
            self._handle_silence_frame(audio_chunk, timestamp)

        # Update stream time
        self.current_stream_time += len(audio_chunk) / self.sample_rate


    def _handle_speech_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Handle a frame detected as speech.

        Accumulates speech frames into buffer, starts new segment if needed,
        and enforces maximum duration limit by splitting long segments.

        Args:
            frame: Audio frame data
            timestamp: Timestamp of this frame
        """
        self.silence_frames = 0

        if not self.is_speech_active:
            # Start new speech segment
            self.is_speech_active = True
            self.speech_start_time = timestamp
            self.speech_buffer = [frame]
        else:
            # Continue accumulating speech
            self.speech_buffer.append(frame)

            # Check if exceeded max duration
            current_duration_ms = len(self.speech_buffer) * self.frame_duration_ms
            if current_duration_ms >= self.max_speech_duration_ms:
                self._finalize_segment()
                # Start new segment immediately
                self.is_speech_active = True
                self.speech_start_time = timestamp
                self.speech_buffer = [frame]


    def _handle_silence_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Handle a frame detected as silence.

        Tracks consecutive silence frames and finalizes speech segment
        when silence timeout is exceeded.

        Args:
            frame: Audio frame data
            timestamp: Timestamp of this frame
        """
        if self.is_speech_active:
            self.silence_frames += 1
            silence_duration_ms = self.silence_frames * self.frame_duration_ms

            if silence_duration_ms >= self.silence_timeout_ms:
                # Silence timeout reached - finalize segment
                self._finalize_segment()


    def _finalize_segment(self) -> None:
        """Finalize accumulated speech segment and emit to queue.

        Applies minimum duration filter and creates AudioSegment instance.
        Sends to both chunk_queue and windower.
        """
        if not self.speech_buffer:
            return

        # Check minimum duration
        duration_ms = len(self.speech_buffer) * self.frame_duration_ms
        if duration_ms < self.min_speech_duration_ms:
            # Too short, discard
            self._reset_segment_state()
            return

        # Create segment
        audio_data = np.concatenate(self.speech_buffer)
        end_time = self.speech_start_time + (len(audio_data) / self.sample_rate)

        chunk_id = self.chunk_id_counter
        self.chunk_id_counter += 1

        segment = AudioSegment(
            type='preliminary',
            data=audio_data,
            start_time=self.speech_start_time,
            end_time=end_time,
            chunk_ids=[chunk_id]
        )

        # Send to queue for instant recognition
        self.chunk_queue.put(segment)

        # Send to windower for aggregation into finalized windows
        if self.windower:
            self.windower.process_segment(segment)

        # Reset state
        self._reset_segment_state()


    def _reset_segment_state(self) -> None:
        """Clear buffering state after segment emission or discard."""
        self.is_speech_active = False
        self.speech_buffer = []
        self.silence_frames = 0


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

        Stops the sounddevice InputStream and flushes any pending buffered segments.
        Also flushes windower to emit final window.
        """
        self.is_running = False
        if self.stream:
            self.stream.stop()

        # Flush pending segments
        self.flush()


    def flush(self) -> None:
        """Flush any pending buffered speech segment.

        Emits any accumulated speech in the buffer, then flushes windower.
        Useful for end-of-stream or testing scenarios.
        """
        # Finalize any pending speech segment
        if self.is_speech_active and len(self.speech_buffer) > 0:
            self._finalize_segment()

        # Flush windower to emit final window
        if self.windower:
            self.windower.flush()