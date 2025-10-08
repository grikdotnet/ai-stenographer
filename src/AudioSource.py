# src/AudioSource.py
from __future__ import annotations
import sounddevice as sd
import queue
import numpy as np
import time
from typing import Dict, Any, List, TYPE_CHECKING
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
        config: Configuration dictionary loaded from stt_config.json
        emit_silence: Whether to emit silence markers
    """

    def __init__(self,
                 chunk_queue: queue.Queue,
                 vad: VoiceActivityDetector,
                 windower: AdaptiveWindower,
                 config: Dict[str, Any],
                 emit_silence: bool = False):

        self.chunk_queue: queue.Queue = chunk_queue
        self.vad: VoiceActivityDetector = vad
        self.windower: AdaptiveWindower = windower

        # Extract values from config
        self.sample_rate: int = config['audio']['sample_rate']
        chunk_duration = config['audio']['chunk_duration']

        self.chunk_size: int = int(self.sample_rate * chunk_duration)
        self.is_running: bool = False
        self.stream: sd.InputStream | None = None
        self.chunk_id_counter: int = 0

        # VAD integration (always enabled)
        self.emit_silence: bool = emit_silence

        # Speech buffering state (moved from VAD)
        self.speech_buffer: List[np.ndarray] = []
        self.is_speech_active: bool = False
        self.speech_start_time: float = 0.0

        # Load timing configs from config (used for buffering logic)
        self.max_speech_duration_ms: int = config['vad']['max_speech_duration_ms']
        self.frame_duration_ms: int = config['vad']['frame_duration_ms']

        # Silence energy computation (cumulative probability scoring)
        self.silence_energy: float = 0.0
        self.silence_energy_threshold: float = config['vad'].get('silence_energy_threshold', 1.5)

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
        """Process audio chunk through VAD using cumulative probability scoring.

        Calculates "amount" of silence as a sum of no-speech probability.

        Silence Energy Logic:
        - High speech prob (over vad threshold): reset energy to 0.0
        - Lower prob is added up (1.0 - prob)
        - Finalize when energy >= silence_energy_threshold (e.g., 1.5)

        This makes strong silence (prob=0.1) finalize faster than weak silence (prob=0.3).

        Args:
            audio_chunk: Audio data to process (32ms frame)
            timestamp: Timestamp of this chunk
        """
        # Call VAD for single frame detection
        vad_result = self.vad.process_frame(audio_chunk)
        speech_prob = vad_result['speech_probability']

        if vad_result['is_speech']:
            self.silence_energy = 0.0
            self._handle_speech_frame(audio_chunk, timestamp)
        elif self.is_speech_active:
            self.silence_energy += (1.0 - speech_prob)

            if self.silence_energy >= self.silence_energy_threshold:
                # Silence ended speech - finalize segment
                self.is_speech_active = False
                self._finalize_segment()


    def _handle_speech_frame(self, audio_chunk: np.ndarray, timestamp: float) -> None:
        """Buffer a speech frame into the current segment.

        Accumulates speech frames into buffer, starts new segment if needed,
        and enforces maximum duration limit by splitting long segments.

        Args:
            audio_chunk: Audio frame data
            timestamp: Timestamp of this frame
        """
        if not self.is_speech_active:
            self.is_speech_active = True
            self.speech_start_time = timestamp
            self.speech_buffer = [audio_chunk]
        else:
            # Accumulate speech
            self.speech_buffer.append(audio_chunk)

            # Check if exceeded max duration
            current_duration_ms = len(self.speech_buffer) * self.frame_duration_ms
            if current_duration_ms >= self.max_speech_duration_ms:
                self._finalize_segment()
                # Start new segment immediately
                self.is_speech_active = True
                self.speech_start_time = timestamp
                self.speech_buffer = [audio_chunk]


    def _finalize_segment(self) -> None:
        """Finalize accumulated speech segment and emit to queue.

        Creates AudioSegment instance and sends to both chunk_queue and windower.
        """
        if not self.speech_buffer:
            return

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
        self.windower.process_segment(segment)

        # Reset state
        self._reset_segment_state()


    def _reset_segment_state(self) -> None:
        """Clear buffering state after segment emission or discard.

        Note: Does not modify is_speech_active - that's managed by
        process_chunk_with_vad() based on speech/silence detection.
        """
        self.speech_buffer = []
        self.silence_energy = 0.0


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
        self.windower.flush()