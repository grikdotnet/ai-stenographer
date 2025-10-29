# src/SoundPreProcessor.py
from __future__ import annotations
import queue
import threading
import numpy as np
import logging
from typing import Dict, Any, List, TYPE_CHECKING
from src.types import AudioSegment

if TYPE_CHECKING:
    from src.VoiceActivityDetector import VoiceActivityDetector
    from src.AdaptiveWindower import AdaptiveWindower


class SoundPreProcessor:
    """Processes raw audio chunks through VAD and buffering.

    Runs in dedicated thread to avoid blocking audio capture.
    Orchestrates VAD, speech buffering, and AdaptiveWindower.

    Dual-Path Audio Processing:
    - Normalized audio → VAD (Silero is volume-dependent, needs consistent levels)
    - Raw audio → speech_buffer → STT (Parakeet requires original amplitude)
    - This ensures accurate VAD detection AND maximum STT accuracy

    RMS Normalization (for VAD only):
    - Normalizes to target RMS level (default: 0.05)
    - Skips normalization for silence (RMS < silence_threshold)
    - Prevents boosting background noise
    - Uses temporal smoothing (α=0.9) to prevent sudden gain changes
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
        chunk_queue: Queue to read raw audio chunks from AudioSource
        speech_queue: Queue to write AudioSegments (preliminary + finalized)
        vad: VoiceActivityDetector instance
        windower: AdaptiveWindower instance (called synchronously)
        config: Configuration dictionary
        verbose: Enable verbose logging
    """

    def __init__(self,
                 chunk_queue: queue.Queue,
                 speech_queue: queue.Queue,
                 vad: VoiceActivityDetector,
                 windower: AdaptiveWindower,
                 config: Dict[str, Any],
                 verbose: bool = False):

        self.chunk_queue: queue.Queue = chunk_queue      # INPUT: raw audio
        self.speech_queue: queue.Queue = speech_queue    # OUTPUT: preliminary segments
        self.vad: VoiceActivityDetector = vad            # Called for speech detection
        self.windower: AdaptiveWindower = windower       # Called synchronously

        # Configuration
        self.sample_rate: int = config['audio']['sample_rate']
        self.frame_duration_ms: int = config['vad']['frame_duration_ms']
        self.max_speech_duration_ms: int = config['windowing']['max_speech_duration_ms']
        self.silence_energy_threshold: float = config['audio']['silence_energy_threshold']
        self.silence_timeout: float = config['windowing']['silence_timeout']

        # RMS normalization (AGC)
        rms_config = config['audio']['rms_normalization']
        self.target_rms: float = rms_config['target_rms']
        self.rms_silence_threshold: float = rms_config['silence_threshold']
        self.gain_smoothing: float = rms_config['gain_smoothing']
        self.current_gain: float = 1.0  # Start with unity gain

        # Speech buffering state
        # Each buffer item: {'audio': np.ndarray, 'timestamp': float, 'chunk_id': int}
        self.speech_buffer: List[Dict[str, Any]] = []
        self.is_speech_active: bool = False
        self.speech_start_time: float = 0.0
        self.silence_energy: float = 0.0
        self.last_speech_timestamp: float = 0.0
        self.speech_before_silence: bool = False

        # Threading
        self.is_running: bool = False
        self.thread: threading.Thread | None = None
        self.verbose: bool = verbose


    def start(self) -> None:
        """Start processing thread"""
        self.is_running = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()


    def stop(self) -> None:
        """Stop processing thread and flush pending segments"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.flush()


    def process(self) -> None:
        """Main loop: read chunk_queue → emit to speech_queue"""
        while self.is_running:
            try:
                chunk_data = self.chunk_queue.get(timeout=0.1)
                self._process_chunk(chunk_data)
            except queue.Empty:
                continue


    def _process_chunk(self, chunk_data: Dict[str, Any]) -> None:
        """Process single raw audio chunk (was process_chunk_with_vad in AudioSource).

        Dual-path processing:
        1. Normalize audio for VAD → ensures consistent speech detection
        2. Call VAD with normalized audio → get speech probability
        3. Buffer raw audio for STT → preserves original quality

        Args:
            chunk_data: Dict with 'audio', 'timestamp', 'chunk_id'
        """
        audio = chunk_data['audio']
        timestamp = chunk_data['timestamp']
        chunk_id = chunk_data['chunk_id']

        # 1. Normalize audio for VAD
        normalized_audio = self._normalize_rms(audio)

        # 2. Call VAD with normalized audio
        vad_result = self.vad.process_frame(normalized_audio)
        speech_prob = vad_result['speech_probability']

        # 3. Handle speech/silence (buffer raw audio, not normalized)
        if vad_result['is_speech']:
            self.silence_energy = 0.0
            self.last_speech_timestamp = timestamp
            self.speech_before_silence = True
            self._handle_speech_frame(audio, timestamp, chunk_id)
        else:
            self.silence_energy += (1.0 - speech_prob)
            self._handle_silence_frame(audio, timestamp, chunk_id)

        # 4. Check silence timeout → flush windower
        if self.speech_before_silence:
            silence_duration = timestamp - self.last_speech_timestamp
            if silence_duration >= self.silence_timeout:
                if self.verbose:
                    logging.debug(f"SoundPreProcessor: silence_timeout reached ({silence_duration:.2f}s)")
                self.windower.flush()
                self.speech_before_silence = False


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


    def _handle_speech_frame(self, audio_chunk: np.ndarray, timestamp: float, chunk_id: int) -> None:
        """Buffer a speech frame into the current segment.

        Accumulates speech frames into buffer, starts new segment if needed,
        and enforces maximum duration limit by splitting long segments.

        Args:
            audio_chunk: Audio frame data (raw, not normalized)
            timestamp: Timestamp of this frame
            chunk_id: Unique chunk identifier
        """
        if not self.is_speech_active:
            self.is_speech_active = True
            self.speech_start_time = timestamp
            self.speech_buffer = [{'audio': audio_chunk, 'timestamp': timestamp, 'chunk_id': chunk_id}]
            if self.verbose:
                logging.debug(f"SoundPreProcessor: Starting segment at {self.speech_start_time:.2f}")
        else:
            if self.verbose:
                logging.debug(f"SoundPreProcessor: adding chunk to segment, chunk_id={chunk_id}")
            self.speech_buffer.append({'audio': audio_chunk, 'timestamp': timestamp, 'chunk_id': chunk_id})

            current_duration_ms = len(self.speech_buffer) * self.frame_duration_ms
            if current_duration_ms >= self.max_speech_duration_ms:
                if self.verbose:
                    logging.debug(f"SoundPreProcessor: max_speech_duration reached ({current_duration_ms}ms)")
                self._finalize_segment()
                # Start new segment (empty, ready for next chunk)
                self.is_speech_active = True
                self.speech_start_time = timestamp
                self.speech_buffer = []


    def _handle_silence_frame(self, audio_chunk: np.ndarray, timestamp: float, chunk_id: int) -> None:
        """Handle silence, finalize segment if threshold reached.

        Args:
            audio_chunk: Audio frame data (not used, silence not buffered)
            timestamp: Timestamp of this frame
            chunk_id: Unique chunk identifier
        """
        if self.is_speech_active:
            if self.verbose:
                logging.debug(f"SoundPreProcessor: silence_energy={self.silence_energy:.2f}, chunk_id={chunk_id}")

            if self.silence_energy >= self.silence_energy_threshold:
                if self.verbose:
                    logging.debug(f"SoundPreProcessor: silence_energy_threshold reached")
                self.is_speech_active = False
                self._finalize_segment()


    def _finalize_segment(self) -> None:
        """Emit preliminary AudioSegment and call windower.

        Creates AudioSegment from buffered chunks and:
        1. Emits to speech_queue (preliminary segment)
        2. Calls windower.process_segment() synchronously
        3. Resets buffering state
        """
        if not self.speech_buffer:
            return

        # Extract audio data and chunk IDs
        audio_data = np.concatenate([chunk['audio'] for chunk in self.speech_buffer])
        chunk_ids = [chunk['chunk_id'] for chunk in self.speech_buffer]

        # Calculate end time from audio length
        end_time = self.speech_start_time + (len(audio_data) / self.sample_rate)

        # Create preliminary AudioSegment
        segment = AudioSegment(
            type='preliminary',
            data=audio_data,
            start_time=self.speech_start_time,
            end_time=end_time,
            chunk_ids=chunk_ids
        )

        if self.verbose:
            logging.debug(f"SoundPreProcessor._finalize_segment(): emitting preliminary segment, chunk_ids={chunk_ids}")

        # Emit to speech_queue
        self.speech_queue.put(segment)

        # Call windower synchronously
        self.windower.process_segment(segment)

        # Reset state
        self._reset_segment_state()


    def _reset_segment_state(self) -> None:
        """Reset buffering state after segment finalization"""
        self.speech_buffer = []
        self.silence_energy = 0.0


    def flush(self) -> None:
        """Emit pending segment and flush windower.

        Useful for end-of-stream or testing scenarios.
        """
        if self.is_speech_active and len(self.speech_buffer) > 0:
            self._finalize_segment()

        self.windower.flush()

        if self.verbose:
            logging.debug("SoundPreProcessor: flush()")
