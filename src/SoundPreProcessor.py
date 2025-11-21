# src/SoundPreProcessor.py
from __future__ import annotations
import queue
import threading
import numpy as np
import logging
from collections import deque
from enum import Enum, auto
from typing import Dict, Any, List, TYPE_CHECKING
from src.types import AudioSegment

if TYPE_CHECKING:
    from src.VoiceActivityDetector import VoiceActivityDetector
    from src.AdaptiveWindower import AdaptiveWindower


class ProcessingState(Enum):
    """State machine states for speech processing.

    State Transitions:
    - IDLE: No speech detected, waiting for activity
    - WAITING_CONFIRMATION: 1-2 consecutive speech chunks detected (prevents false positives)
    - ACTIVE_SPEECH: 3+ consecutive speech chunks, actively buffering segment
    - ACCUMULATING_SILENCE: Speech ended, accumulating silence energy before finalization

    Transition Rules:
    IDLE → WAITING_CONFIRMATION: First speech chunk detected
    WAITING_CONFIRMATION → ACTIVE_SPEECH: 3rd consecutive speech chunk
    WAITING_CONFIRMATION → IDLE: Silence chunk (reset counter)
    ACTIVE_SPEECH → ACCUMULATING_SILENCE: First silence chunk after speech
    ACTIVE_SPEECH → ACTIVE_SPEECH: Continues on max_duration split
    ACCUMULATING_SILENCE → ACTIVE_SPEECH: Speech resumes (reset energy)
    ACCUMULATING_SILENCE → IDLE: Silence energy exceeds threshold
    """
    IDLE = auto()
    WAITING_CONFIRMATION = auto()
    ACTIVE_SPEECH = auto()
    ACCUMULATING_SILENCE = auto()


class SoundPreProcessor:
    """Processes raw audio chunks through VAD and buffering.

    Runs in dedicated thread to avoid blocking audio capture.
    Orchestrates VAD, speech buffering, and AdaptiveWindower.

    Idle Buffer Strategy:
    - Maintains circular buffer of last 48 chunks (1.536s @ 32ms/chunk)
    - Accumulates during IDLE state
    - WAITING_CONFIRMATION uses separate confirmation_chunks list
    - Stops accumulating during ACTIVE_SPEECH and ACCUMULATING_SILENCE
    - left_context: entire idle_buffer snapshot at speech start
    - right_context: trailing silence after speech ends
    - Enables better STT quality for short words and prevents hallucinations

    Dual-Path Audio Processing:
    - Normalized audio → VAD (Silero is volume-dependent, needs consistent levels)
    - Raw audio → speech_buffer → STT (Parakeet requires original amplitude)
    - This ensures accurate VAD detection AND maximum STT accuracy

    RMS Normalization (for VAD only):
    - Normalizes to target RMS level (default: 0.05)
    - Skips normalization for silence (RMS < silence_threshold)
    - Prevents boosting background noise
    - Uses temporal smoothing to prevent sudden gain changes

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

    CONTEXT_BUFFER_SIZE = 48  # 1.536s @ 32ms/chunk

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

        self._state: ProcessingState = ProcessingState.IDLE

        # Speech buffering state
        # Each buffer item: {'audio': np.ndarray, 'timestamp': float, 'chunk_id': int}
        self.speech_buffer: List[Dict[str, Any]] = []
        self.speech_start_time: float = 0.0
        self.silence_energy: float = 0.0
        self.last_speech_timestamp: float = 0.0
        self.speech_before_silence: bool = False

        # Consecutive speech chunk counter for false positive prevention
        # Requires 3 consecutive chunks above threshold before starting segment
        self.consecutive_speech_count: int = 0
        self.CONSECUTIVE_SPEECH_CHUNKS: int = 3

        # Idle buffer: circular buffer for gathering pre-speech context (only in IDLE/WAITING_CONFIRMATION)
        # Each item: {'audio': np.ndarray, 'timestamp': float, 'chunk_id': int|None, 'is_speech': bool}
        self.idle_buffer: deque = deque(maxlen=self.CONTEXT_BUFFER_SIZE)
        self.left_context_snapshot: List[np.ndarray] | None = None

        # Confirmation chunks: temporary list for WAITING_CONFIRMATION state
        # Holds 1-2 chunks before confirming speech started
        self.confirmation_chunks: List[Dict[str, Any]] = []

        self.chunk_id_counter: int = 0

        self.is_running: bool = False
        self.thread: threading.Thread | None = None
        self.verbose: bool = verbose

        self._first_chunk_timestamp = 0

    @property
    def state(self) -> ProcessingState:
        """Current processing state."""
        return self._state

    @state.setter
    def state(self, value: ProcessingState) -> None:
        """Set processing state."""
        self._state = value

    @property
    def is_speech_active(self) -> bool:
        """Check if speech is currently active (ACTIVE_SPEECH or ACCUMULATING_SILENCE)."""
        return self._state in (ProcessingState.ACTIVE_SPEECH, ProcessingState.ACCUMULATING_SILENCE)

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
        """Process single raw audio chunk using state machine.

        Dual-path processing:
        1. Normalize audio for VAD → ensures consistent speech detection
        2. Call VAD with normalized audio → get speech probability
        3. Buffer raw audio for STT → preserves original quality

        Args:
            chunk_data: Dict with 'audio', 'timestamp'
        """
        audio = chunk_data['audio']
        timestamp = chunk_data['timestamp']

        # Track first chunk timestamp
        if self._first_chunk_timestamp == 0:
            self._first_chunk_timestamp = timestamp

        # Dual-path: normalize for VAD, keep raw for STT
        vad_result = self.vad.process_frame( self._normalize_rms(audio) )
        speech_prob = vad_result['speech_probability']
        is_speech = vad_result['is_speech']

        # State machine
        match self.state:
            case ProcessingState.IDLE:
                if is_speech:
                    # IDLE → WAITING_CONFIRMATION
                    self.consecutive_speech_count = 1
                    self.state = ProcessingState.WAITING_CONFIRMATION
                    # Start confirmation_chunks, don't add to idle_buffer yet
                    self._append_to_confirmation_chunks(audio, timestamp, is_speech=True)
                else:
                    # IDLE → IDLE
                    self._append_to_idle_buffer(audio, timestamp, is_speech=False, chunk_id=None)

            case ProcessingState.WAITING_CONFIRMATION:
                if is_speech:
                    self.consecutive_speech_count += 1

                    if self.consecutive_speech_count >= self.CONSECUTIVE_SPEECH_CHUNKS:
                        # WAITING_CONFIRMATION → ACTIVE_SPEECH
                        self.state = ProcessingState.ACTIVE_SPEECH
                        self._capture_left_context_from_buffer()
                        self._initialize_speech_buffer_from_confirmation()

                        # Set speech tracking variables
                        self.last_speech_timestamp = timestamp
                        self.speech_before_silence = True

                        if self.verbose:
                            offset = self.speech_start_time - self._first_chunk_timestamp
                            logging.debug(f"----------------------------")
                            logging.debug(f"SoundPreProcessor: Starting segment at {offset:.2f}")
                            logging.debug(f"captured {len(self.left_context_snapshot)} left context chunks")

                        # Add current chunk to speech buffer (3rd confirmation chunk)
                        self._append_to_speech_buffer(audio, timestamp, is_speech=True, chunk_id=self.chunk_id_counter)
                        self.chunk_id_counter += 1
                    else:
                        # WAITING_CONFIRMATION → WAITING_CONFIRMATION
                        self._append_to_confirmation_chunks(audio, timestamp, is_speech=True)
                else:
                    # WAITING_CONFIRMATION → IDLE
                    self.state = ProcessingState.IDLE
                    self.consecutive_speech_count = 0
                    # Move confirmation chunks back to idle_buffer before clearing
                    for conf_chunk in self.confirmation_chunks:
                        self._append_to_idle_buffer(conf_chunk['audio'], conf_chunk['timestamp'],
                                                   is_speech=True, chunk_id=None)
                    self.confirmation_chunks = []
                    self._append_to_idle_buffer(audio, timestamp, is_speech=False, chunk_id=None)

            case ProcessingState.ACTIVE_SPEECH:
                if is_speech:
                    # ACTIVE_SPEECH → ACTIVE_SPEECH
                    self.silence_energy = 0.0
                    self.last_speech_timestamp = timestamp
                    self.speech_before_silence = True

                    self._append_to_speech_buffer(audio, timestamp, is_speech=True, chunk_id=self.chunk_id_counter)
                    # Note: Do NOT append to idle_buffer during ACTIVE_SPEECH
                    self.chunk_id_counter += 1

                    # Check max duration
                    current_duration_ms = len(self.speech_buffer) * self.frame_duration_ms
                    if current_duration_ms >= self.max_speech_duration_ms:
                        if self.verbose:
                            offset = timestamp - self._first_chunk_timestamp
                            logging.debug(f"----------------------------")
                            logging.debug(f"SoundPreProcessor: max_speech_duration reached, splitting at {offset:.2f}")

                        # Search for silence breakpoint
                        breakpoint_idx = self._find_silence_breakpoint()

                        if breakpoint_idx is not None:
                            # Breakpoint split
                            if self.verbose:
                                breakpoint_file_time = self.speech_buffer[breakpoint_idx]['timestamp']
                                offset = breakpoint_file_time - self._first_chunk_timestamp
                                logging.debug(f"SoundPreProcessor: silence breakpoint at chunk {breakpoint_idx}, offset={offset:.2f}s)")

                            segment = self._build_audio_segment(breakpoint_idx=breakpoint_idx)
                            self.speech_queue.put(segment)
                            self.windower.process_segment(segment)
                            self._keep_remainder_after_breakpoint(breakpoint_idx)

                            if self.verbose:
                                logging.debug(f"SoundPreProcessor: emitting preliminary segment")
                                logging.debug(f"  chunk_ids={segment.chunk_ids}")
                                logging.debug(f"  left_context={len(segment.left_context)/512} chunks, right_context={len(segment.right_context)/512} chunks")

                            # ACTIVE_SPEECH → ACTIVE_SPEECH (continuation)
                        else:
                            # Hard cut - emit current buffer and reset for immediate continuation
                            if self.verbose:
                                logging.debug(f"SoundPreProcessor: no silence breakpoint found, using hard cut")

                            segment = self._build_audio_segment(breakpoint_idx=None)
                            self.speech_queue.put(segment)
                            self.windower.process_segment(segment)

                            if self.verbose:
                                logging.debug(f"SoundPreProcessor: emitting preliminary segment")
                                logging.debug(f"  chunk_ids={segment.chunk_ids}")
                                logging.debug(f"  left_context={len(segment.left_context)/512} chunks, right_context={len(segment.right_context)/512} chunks")

                            # Reset buffer but stay in ACTIVE_SPEECH (speech continues)
                            self.speech_buffer = []
                            self.speech_start_time = timestamp
                            # Keep state = ACTIVE_SPEECH
                            # Don't reset left_context_snapshot - preserve for continuity
                else:
                    # ACTIVE_SPEECH → ACCUMULATING_SILENCE
                    self.state = ProcessingState.ACCUMULATING_SILENCE
                    self.consecutive_speech_count = 0
                    self.silence_energy += (1.0 - speech_prob)

                    self._append_to_speech_buffer(audio, timestamp, is_speech=False, chunk_id=self.chunk_id_counter)
                    # Note: Do NOT append to idle_buffer during ACCUMULATING_SILENCE
                    self.chunk_id_counter += 1

                    if self.verbose:
                        logging.debug(f"SoundPreProcessor: silence_energy={self.silence_energy:.2f}")

            case ProcessingState.ACCUMULATING_SILENCE:
                if is_speech:
                    # ACCUMULATING_SILENCE → ACTIVE_SPEECH
                    self.state = ProcessingState.ACTIVE_SPEECH
                    self.silence_energy = 0.0
                    self.consecutive_speech_count = 1
                    self.last_speech_timestamp = timestamp
                    self.speech_before_silence = True

                    self._append_to_speech_buffer(audio, timestamp, is_speech=True, chunk_id=self.chunk_id_counter)
                    # Note: Do NOT append to idle_buffer during ACTIVE_SPEECH
                    self.chunk_id_counter += 1
                else:
                    # ACCUMULATING_SILENCE → ACCUMULATING_SILENCE or IDLE
                    self.silence_energy += (1.0 - speech_prob)

                    self._append_to_speech_buffer(audio, timestamp, is_speech=False, chunk_id=self.chunk_id_counter)
                    # Note: Do NOT append to idle_buffer during ACCUMULATING_SILENCE
                    self.chunk_id_counter += 1

                    if self.verbose:
                        logging.debug(f"SoundPreProcessor: silence_energy={self.silence_energy:.2f}")

                    if self.silence_energy >= self.silence_energy_threshold:
                        # ACCUMULATING_SILENCE → IDLE
                        if self.verbose:
                            logging.debug(f"SoundPreProcessor: silence_energy_threshold reached")

                        segment = self._build_audio_segment(breakpoint_idx=None)
                        self.speech_queue.put(segment)
                        self.windower.process_segment(segment)
                        self._reset_segment_state()

                        if self.verbose:
                            logging.debug(f"SoundPreProcessor: emitting preliminary segment")
                            logging.debug(f"  chunk_ids={segment.chunk_ids}")
                            logging.debug(f"  left_context={len(segment.left_context)/512} chunks, right_context={len(segment.right_context)/512} chunks")

                        self.state = ProcessingState.IDLE
                        self.consecutive_speech_count = 0

        # Silence timeout logic (for windower flushing)
        if self.speech_before_silence and not is_speech:
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

        # Don't normalize silence
        if rms < self.rms_silence_threshold:
            return audio_chunk

        target_gain = self.target_rms / rms

        # Apply temporal smoothing to prevent sudden gain changes
        # current_gain = α * old_gain + (1-α) * new_gain
        # Higher α (e.g., 0.9) = slower, smoother changes
        # Lower α (e.g., 0.5) = faster response
        self.current_gain = self.gain_smoothing * self.current_gain + (1 - self.gain_smoothing) * target_gain

        return audio_chunk * self.current_gain

    # ========================================================================
    # Helper Methods for State Machine
    # ========================================================================

    def _append_to_idle_buffer(self, audio: np.ndarray, timestamp: float, is_speech: bool, chunk_id: int | None) -> None:
        """Append chunk dict to circular idle_buffer (only during IDLE).

        Args:
            audio: Audio chunk data
            timestamp: Timestamp of chunk
            is_speech: Whether chunk is speech
            chunk_id: Chunk ID (None for silence outside active speech)
        """
        chunk = {
            'audio': audio,
            'timestamp': timestamp,
            'chunk_id': chunk_id,
            'is_speech': is_speech
        }
        self.idle_buffer.append(chunk)

    def _append_to_confirmation_chunks(self, audio: np.ndarray, timestamp: float, is_speech: bool) -> None:
        """Append chunk to confirmation_chunks list (during WAITING_CONFIRMATION).

        Args:
            audio: Audio chunk data
            timestamp: Timestamp of chunk
            is_speech: Whether chunk is speech
        """
        chunk = {
            'audio': audio,
            'timestamp': timestamp,
            'is_speech': is_speech
        }
        self.confirmation_chunks.append(chunk)

    def _append_to_speech_buffer(self, audio: np.ndarray, timestamp: float, is_speech: bool, chunk_id: int) -> None:
        """Append chunk dict to speech_buffer.

        Args:
            audio: Audio chunk data
            timestamp: Timestamp of chunk
            is_speech: Whether chunk is speech
            chunk_id: Chunk ID
        """
        chunk = {
            'audio': audio,
            'timestamp': timestamp,
            'chunk_id': chunk_id,
            'is_speech': is_speech
        }
        self.speech_buffer.append(chunk)

    def _capture_left_context_from_buffer(self) -> None:
        """Extract left context from idle_buffer (entire buffer, no slicing).

        Called when transitioning WAITING_CONFIRMATION → ACTIVE_SPEECH.
        Snapshots left context from entire idle_buffer at transition time.
        """
        self.left_context_snapshot = [chunk['audio'] for chunk in self.idle_buffer]

    def _initialize_speech_buffer_from_confirmation(self) -> None:
        """Initialize speech_buffer from confirmation_chunks.

        Called when transitioning WAITING_CONFIRMATION → ACTIVE_SPEECH.
        Assigns chunk_ids and populates speech_buffer with first speech chunks.
        Sets speech_start_time to first chunk timestamp.
        Clears confirmation_chunks after initialization.
        """
        self.speech_buffer = []
        for chunk in self.confirmation_chunks:
            # Assign chunk_id from counter
            chunk_with_id = {
                'audio': chunk['audio'],
                'timestamp': chunk['timestamp'],
                'chunk_id': self.chunk_id_counter,
                'is_speech': chunk['is_speech']
            }
            self.speech_buffer.append(chunk_with_id)
            self.chunk_id_counter += 1

        if self.speech_buffer:
            self.speech_start_time = self.speech_buffer[0]['timestamp']

        # Clear confirmation chunks after initialization
        self.confirmation_chunks = []

    def _build_audio_segment(self, breakpoint_idx: int | None = None) -> AudioSegment:
        """Build AudioSegment from speech_buffer.

        Extracts:
        - left_context from left_context_snapshot
        - data from speech chunks (0 to breakpoint_idx or all)
        - right_context from trailing silence or buffer after breakpoint

        Args:
            breakpoint_idx: Optional index to split at (inclusive in data)

        Returns:
            AudioSegment ready to emit
        """
        left_context = (np.concatenate(self.left_context_snapshot)
                       if self.left_context_snapshot
                       else np.array([], dtype=np.float32))

        if breakpoint_idx is None:
            # Use entire buffer, trailing silence as right_context
            data_chunks = [chunk['audio'] for chunk in self.speech_buffer
                          if chunk.get('is_speech', True)]
            data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float32)

            right_context_chunks = []
            for chunk in reversed(self.speech_buffer):
                if not chunk.get('is_speech', True):
                    right_context_chunks.insert(0, chunk['audio'])
                else:
                    break

            right_context = (np.concatenate(right_context_chunks)
                            if right_context_chunks
                            else np.array([], dtype=np.float32))

            chunk_ids = [chunk['chunk_id'] for chunk in self.speech_buffer
                        if chunk['chunk_id'] is not None]
        else:
            # Breakpoint split: split at breakpoint_idx (inclusive)
            data_buffer = self.speech_buffer[0:breakpoint_idx+1]
            right_context_buffer = self.speech_buffer[breakpoint_idx+1:breakpoint_idx+7]  # max 6 chunks

            data_chunks = [chunk['audio'] for chunk in data_buffer]
            data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float32)

            right_context_chunks = [chunk['audio'] for chunk in right_context_buffer]
            right_context = (np.concatenate(right_context_chunks)
                            if right_context_chunks
                            else np.array([], dtype=np.float32))

            chunk_ids = [chunk['chunk_id'] for chunk in data_buffer
                        if chunk['chunk_id'] is not None]

        end_time = self.speech_start_time + (len(data) / self.sample_rate)

        return AudioSegment(
            type='preliminary',
            data=data,
            left_context=left_context,
            right_context=right_context,
            start_time=self.speech_start_time,
            end_time=end_time,
            chunk_ids=chunk_ids
        )

    def _keep_remainder_after_breakpoint(self, breakpoint_idx: int) -> None:
        """Keep remainder chunks in speech_buffer after breakpoint_idx.

        Updates:
        - speech_buffer to contain only remainder
        - speech_start_time for remainder
        - silence_energy reset to 0

        Args:
            breakpoint_idx: Index of breakpoint (chunks after this are kept)
        """
        self.speech_buffer = self.speech_buffer[breakpoint_idx+1:]
        if self.speech_buffer:
            self.speech_start_time = self.speech_buffer[0]['timestamp']
        self.silence_energy = 0.0

    def _reset_segment_state(self) -> None:
        """Reset segment state after finalization.

        Clears:
        - speech_buffer
        - silence_energy
        - left_context_snapshot
        """
        self.speech_buffer = []
        self.silence_energy = 0.0
        self.left_context_snapshot = None

    def _find_silence_breakpoint(self) -> int | None:
        """Search backward for last silence chunk, skipping last 3 elements.

        Used when max_speech_duration_ms is reached to find a natural breakpoint.
        Searches backward from the end of speech_buffer, skipping the last 3 chunks
        to ensure right_context isn't empty.

        Returns:
            Index of breakpoint (inclusive in data), or None if no silence found
        """
        if len(self.speech_buffer) < 4:
            return None

        # Search range: 0 to len-4 (skip last 3 chunks)
        search_end = len(self.speech_buffer) - 3

        # Search backward for last silence chunk
        for i in range(search_end - 1, -1, -1):
            if not self.speech_buffer[i].get('is_speech', True):
                return i

        return None

    def flush(self) -> None:
        """Emit pending segment and flush windower.

        Useful for end-of-stream or testing scenarios.
        """
        if self.is_speech_active and len(self.speech_buffer) > 0:
            segment = self._build_audio_segment(breakpoint_idx=None)
            self.speech_queue.put(segment)
            self.windower.process_segment(segment)
            self._reset_segment_state()
            self.state = ProcessingState.IDLE

        self.windower.flush()

        if self.verbose:
            logging.debug("SoundPreProcessor: flush()")
