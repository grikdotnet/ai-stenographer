# src/AudioSource.py
from __future__ import annotations
import sounddevice as sd
import queue
import numpy as np
import time
import logging
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

    Dual-Path Processing:
    - Normalized audio is sent to VAD for consistent speech detection
    - Raw (unnormalized) audio is buffered and sent to STT for maximum accuracy
    - This addresses Silero VAD's volume-dependency without harming Parakeet STT

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
                 emit_silence: bool = False,
                 verbose: bool = False):

        self.chunk_queue: queue.Queue = chunk_queue
        self.vad: VoiceActivityDetector = vad
        self.windower: AdaptiveWindower = windower
        self.verbose: bool = verbose

        self.sample_rate: int = config['audio']['sample_rate']
        chunk_duration = config['audio']['chunk_duration']

        self.chunk_size: int = int(self.sample_rate * chunk_duration)
        self.is_running: bool = False
        self.stream: sd.InputStream | None = None
        self.chunk_id_counter: int = 0

        self.emit_silence: bool = emit_silence

        self.speech_buffer: List[np.ndarray] = []
        self.is_speech_active: bool = False
        self.speech_start_time: float = 0.0

        self.max_speech_duration_ms: int = config['windowing']['max_speech_duration_ms']
        self.frame_duration_ms: int = config['vad']['frame_duration_ms']

        self.silence_energy: float = 0.0
        self.silence_energy_threshold: float = config['audio'].get('silence_energy_threshold', 1.5)

        self.last_speech_timestamp: float = 0.0
        self.speech_before_silence: bool = False
        self.silence_timeout: float = config['windowing'].get('silence_timeout', 0.5)  # Default 0.5s

        # RMS normalization (AGC)
        rms_config = config['audio'].get('rms_normalization', {})
        self.target_rms: float = rms_config.get('target_rms', 0.05)
        self.rms_silence_threshold: float = rms_config.get('silence_threshold', 0.001)
        self.gain_smoothing: float = rms_config.get('gain_smoothing', 0.9)
        self.current_gain: float = 1.0  # Start with unity gain


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
            logging.error(f"Audio error: {status}")

        audio_float: np.ndarray = indata[:, 0].astype(np.float32)
        current_time = time.time()

        self.process_chunk_with_vad(audio_float, current_time)


    def process_chunk_with_vad(self, audio_chunk: np.ndarray, timestamp: float) -> None:
        """Process audio chunk through VAD using cumulative probability scoring.

        Dual-Path Audio Processing:
        - Normalized audio → VAD (Silero is volume-dependent, needs consistent levels)
        - Raw audio → speech_buffer → STT (Parakeet requires original amplitude)
        - This ensures accurate VAD detection AND maximum STT accuracy

        RMS Normalization (for VAD only):
        - Normalizes to target RMS level (default: 0.05)
        - Skips normalization for silence (RMS < silence_threshold)
        - Prevents boosting background noise
        - Uses temporal smoothing (α=0.85) to prevent sudden gain changes
        - Gain state maintained in self.current_gain

        Silence Energy Logic:
        - High speech prob (over vad threshold): reset energy to 0.0
        - Lower prob is added up (1.0 - prob)
        - Finalize when energy >= silence_energy_threshold (e.g., 1.5)
        - Strong silence (prob=0.1) finalizes faster than weak silence (prob=0.3)

        Silence Timeout Logic:
        - After speech ends, track silence duration from timestamp parameter
        - If silence exceeds timeout AND speech was detected before, flush windower
        - Reset speech_before_silence flag after flush to prevent repeated flushes

        Args:
            audio_chunk: Raw audio data to process (32ms frame)
            timestamp: Timestamp of this chunk
        """
        normalized_audio_chunk = self._normalize_rms(audio_chunk)

        vad_result = self.vad.process_frame(normalized_audio_chunk)
        speech_prob = vad_result['speech_probability']

        if vad_result['is_speech']:
            self.silence_energy = 0.0
            self.last_speech_timestamp = timestamp
            self.speech_before_silence = True
            self._handle_speech_frame(audio_chunk, timestamp)
        else:
            silence_duration = timestamp - self.last_speech_timestamp
            if self.is_speech_active:
                if self.verbose:
                    logging.debug(f"AudioSource: silence_energy={self.silence_energy}, duration: {silence_duration:.2f}s chunk {self.chunk_id_counter}")
                self.silence_energy += (1.0 - speech_prob)

                if self.silence_energy >= self.silence_energy_threshold:
                    if self.verbose:
                        logging.debug(f"AudioSource: silence_energy_threshold reached")
                    self.is_speech_active = False
                    self._finalize_segment()

            # Check silence timeout for windower flush
            if self.speech_before_silence:
                if silence_duration >= self.silence_timeout:
                    if self.verbose:
                        logging.debug(f"AudioSource: silence_timeout reached")
                    self.flush()
                    self.speech_before_silence = False  # Reset after flush to prevent repeated flushes
                    self.last_speech_timestamp = 0.0


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
            if self.verbose:
                logging.debug(f"AudioSource: Starting segment at {self.speech_start_time:.2f}")
        else:
            if self.verbose:
                logging.debug(f"AudioSource: adding chunk to segment {self.chunk_id_counter}")
            self.speech_buffer.append(audio_chunk)

            current_duration_ms = len(self.speech_buffer) * self.frame_duration_ms
            if current_duration_ms >= self.max_speech_duration_ms:
                self._finalize_segment()
                # Start new segment (empty, ready for next chunk)
                self.is_speech_active = True
                self.speech_start_time = timestamp
                self.speech_buffer = []


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

        if self.verbose:
            logging.debug(f"AudioSource._finalize_segment(): pushed chunk_id={chunk_id}")

        self.chunk_queue.put(segment)

        self.windower.process_segment(segment)

        self.speech_buffer = []
        self.silence_energy = 0.0

    def _normalize_rms(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Normalize audio chunk to target RMS level with temporal smoothing (AGC).

        Purpose: Creates normalized audio for VAD only. The raw audio is preserved
        for STT downstream (Parakeet requires original amplitude per NeMo design).

        Automatic Gain Control that boosts quiet audio and reduces loud audio
        to maintain consistent volume level for VAD. Uses temporal smoothing to
        prevent sudden gain changes and transient noise amplification.

        Temporal Smoothing Algorithm:
        - Calculate target gain for current chunk: target_gain = target_rms / current_rms
        - Smooth with previous gain: current_gain = α * current_gain + (1-α) * target_gain
        - Apply smoothed gain to chunk
        - Prevents sudden volume jumps between chunks (e.g., speech → pause → speech)

        Args:
            audio_chunk: Raw audio data to normalize

        Returns:
            Normalized audio chunk with smoothed gain applied (for VAD use only)
        """
        rms = np.sqrt(np.mean(audio_chunk**2))

        # Don't normalize silence (avoids boosting background noise)
        if rms < self.rms_silence_threshold:
            return audio_chunk

        target_gain = self.target_rms / rms

        # Apply temporal smoothing to prevent sudden gain changes
        # current_gain = α * old_gain + (1-α) * new_gain
        # Higher α (e.g., 0.9) = slower, smoother changes
        # Lower α (e.g., 0.5) = faster response
        self.current_gain = self.gain_smoothing * self.current_gain + (1 - self.gain_smoothing) * target_gain

        return audio_chunk * self.current_gain

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

        self.flush()


    def flush(self) -> None:
        """Flush any pending buffered speech segment.

        Emits any accumulated speech in the buffer, then flushes windower.
        Useful for end-of-stream or testing scenarios.
        """
        if self.is_speech_active and len(self.speech_buffer) > 0:
            self._finalize_segment()

        self.windower.flush()

        if self.verbose:
            logging.debug("AudioSource: flush()")
