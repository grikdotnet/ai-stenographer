# src/SoundPreProcessor.py
"""
Tests for this module:
- tests/test_sound_preprocessor.py - Core functionality (VAD, buffering, context extraction)
- tests/test_sound_preprocessor_helpers.py - Helper methods (buffering, segment finalization)
- tests/test_sound_preprocessor_state_machine.py - State machine transitions
"""
from __future__ import annotations
import queue
import threading
import numpy as np
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, TYPE_CHECKING
from src.types import AudioSegment

if TYPE_CHECKING:
    from src.VoiceActivityDetector import VoiceActivityDetector
    from src.AdaptiveWindower import AdaptiveWindower


# ========================================================================
# Audio Processing State Machine (Internal Implementation Details)
# ========================================================================

# Buffer size constant for idle buffer (~640ms at 32ms/chunk)
CONTEXT_BUFFER_SIZE = 20


class ProcessingStatesEnum(Enum):
    """State machine states for speech processing.

    State Transitions:
    - IDLE: No speech detected, waiting for activity
    - WAITING_CONFIRMATION: several consecutive speech chunks detected (prevents false positives)
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


@dataclass
class AudioProcessingState:
    """State container for audio segment processing in SoundPreProcessor.

    Groups all mutable state related to audio segment accumulation,
    leaving SoundPreProcessor with logic and technical infrastructure.
    """

    # State machine
    state: ProcessingStatesEnum = ProcessingStatesEnum.IDLE
    consecutive_speech_count: int = 0
    chunk_id_counter: int = 0

    # Buffering - idle_buffer maxlen=20 (~640ms at 32ms/chunk)
    speech_buffer: list[dict[str, Any]] = field(default_factory=list)
    idle_buffer: deque = field(default_factory=lambda: deque(maxlen=CONTEXT_BUFFER_SIZE))

    # Timing
    speech_start_time: float = 0.0
    last_speech_timestamp: float = 0.0
    first_chunk_timestamp: float = 0.0

    # Silence tracking
    silence_energy: float = 0.0
    silence_start_idx: int | None = None

    # Context
    left_context_snapshot: list[np.ndarray] | None = None

    # Control flags
    speech_before_silence: bool = False

    def reset_segment(self) -> None:
        """Reset state after segment finalization.

        Clears buffer, energy, and context but preserves timing flags
        (last_speech_timestamp, speech_before_silence) for timeout logic.
        """
        self.speech_buffer = []
        self.silence_energy = 0.0
        self.silence_start_idx = None
        self.left_context_snapshot = None

    def reset_silence_tracking(self) -> None:
        """Reset silence tracking when speech resumes."""
        self.silence_energy = 0.0
        self.silence_start_idx = None


# ========================================================================
# Main Component
# ========================================================================

class SoundPreProcessor:
    """Processes raw audio chunks through VAD and buffering.

    Runs in dedicated thread to avoid blocking audio capture.
    Orchestrates VAD, speech buffering, and AdaptiveWindower.

    Idle Buffer Strategy:
    - Maintains circular buffer of last 48 chunks (1.536s @ 32ms/chunk)
    - Accumulates during IDLE and WAITING_CONFIRMATION states
    - Stops accumulating during ACTIVE_SPEECH and ACCUMULATING_SILENCE
    - On speech confirmation: extracts last (N-1) chunks to initialize speech_buffer
    - left_context: idle_buffer snapshot (excluding extracted chunks) at speech start
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

    # Consecutive speech chunks required before starting segment (false positive prevention)
    CONSECUTIVE_SPEECH_CHUNKS: int = 3

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

        # Audio processing state (all mutable segment-related state)
        self.audio_state = AudioProcessingState()

        # Threading control
        self.is_running: bool = False
        self.thread: threading.Thread | None = None
        self.verbose: bool = verbose

    @property
    def state(self) -> ProcessingStatesEnum:
        """Current processing state (delegated to audio_state)."""
        return self.audio_state.state

    @state.setter
    def state(self, value: ProcessingStatesEnum) -> None:
        """Set processing state (delegated to audio_state)."""
        self.audio_state.state = value

    @property
    def is_speech_active(self) -> bool:
        """Check if speech is currently active (ACTIVE_SPEECH or ACCUMULATING_SILENCE)."""
        return self.audio_state.state in (ProcessingStatesEnum.ACTIVE_SPEECH, ProcessingStatesEnum.ACCUMULATING_SILENCE)

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
        s = self.audio_state  # shorthand for cleaner code

        if s.first_chunk_timestamp == 0:
            s.first_chunk_timestamp = timestamp

        # normalize for VAD, keep raw for STT
        vad_result = self.vad.process_frame( self._normalize_rms(audio) )
        speech_prob = vad_result['speech_probability']
        is_speech = vad_result['is_speech']

        # State machine
        match self.state:
            case ProcessingStatesEnum.IDLE:
                if is_speech:
                    # IDLE → WAITING_CONFIRMATION
                    s.consecutive_speech_count = 1
                    self.state = ProcessingStatesEnum.WAITING_CONFIRMATION
                self._append_to_idle_buffer(audio, timestamp, is_speech=is_speech)

                # Flush windower if silence timeout exceeded after speech
                if s.speech_before_silence:
                    silence_duration = timestamp - s.last_speech_timestamp
                    if silence_duration >= self.silence_timeout:
                        if self.verbose:
                            logging.debug(f"SoundPreProcessor: silence_timeout reached ({silence_duration:.2f}s)")
                        self.windower.flush()
                        s.speech_before_silence = False

            case ProcessingStatesEnum.WAITING_CONFIRMATION:
                self._append_to_idle_buffer(audio, timestamp, is_speech=is_speech)

                if is_speech:
                    s.consecutive_speech_count += 1

                    if s.consecutive_speech_count >= self.CONSECUTIVE_SPEECH_CHUNKS:
                        # WAITING_CONFIRMATION → ACTIVE_SPEECH
                        self.state = ProcessingStatesEnum.ACTIVE_SPEECH

                        # Extract confirmation chunks from idle_buffer
                        if len(s.idle_buffer) >= self.CONSECUTIVE_SPEECH_CHUNKS:
                            confirmation_chunks_from_idle = [
                                s.idle_buffer.pop()
                                for _ in range(self.CONSECUTIVE_SPEECH_CHUNKS)
                            ]
                            confirmation_chunks_from_idle.reverse()

                            s.left_context_snapshot = [chunk['audio'] for chunk in s.idle_buffer]
                        else:
                            # Edge case: fewer chunks in buffer than expected (shouldn't happen normally)
                            s.left_context_snapshot = []
                            confirmation_chunks_from_idle = list(s.idle_buffer)
                            s.idle_buffer.clear()

                        # Initialize speech_buffer with all N confirmation chunks
                        self._initialize_speech_buffer_from_idle(confirmation_chunks_from_idle)

                        s.last_speech_timestamp = timestamp
                        s.speech_before_silence = True
                        s.silence_energy = 0.0

                        if self.verbose:
                            offset = s.speech_start_time - s.first_chunk_timestamp
                            logging.debug(f"----------------------------")
                            logging.debug(f"SoundPreProcessor: Starting segment at {offset:.2f}")
                            logging.debug(f"captured {len(s.left_context_snapshot)} left context chunks")
                else:
                    # WAITING_CONFIRMATION → IDLE
                    self.state = ProcessingStatesEnum.IDLE
                    s.consecutive_speech_count = 0

            case ProcessingStatesEnum.ACTIVE_SPEECH:
                if is_speech:
                    # ACTIVE_SPEECH → ACTIVE_SPEECH
                    s.last_speech_timestamp = timestamp

                    self._append_to_speech_buffer(audio, timestamp, is_speech=True, chunk_id=s.chunk_id_counter)
                    s.chunk_id_counter += 1

                    current_duration_ms = len(s.speech_buffer) * self.frame_duration_ms
                    if current_duration_ms >= self.max_speech_duration_ms:
                        self._handle_max_duration_split(timestamp)
                else:
                    # ACTIVE_SPEECH → ACCUMULATING_SILENCE
                    self.state = ProcessingStatesEnum.ACCUMULATING_SILENCE
                    s.silence_start_idx = len(s.speech_buffer)  # first silence chunk index
                    s.silence_energy += (1.0 - speech_prob)

                    self._append_to_speech_buffer(audio, timestamp, is_speech=False, chunk_id=s.chunk_id_counter)
                    s.chunk_id_counter += 1

                    if self.verbose:
                        logging.debug(f"SoundPreProcessor: silence_energy={s.silence_energy:.2f}")

            case ProcessingStatesEnum.ACCUMULATING_SILENCE:
                # s.speech_before_silence is still True
                if is_speech:
                    # ACCUMULATING_SILENCE → ACTIVE_SPEECH
                    self.state = ProcessingStatesEnum.ACTIVE_SPEECH
                    s.reset_silence_tracking()
                    s.last_speech_timestamp = timestamp

                    self._append_to_speech_buffer(audio, timestamp, is_speech=True, chunk_id=s.chunk_id_counter)
                    s.chunk_id_counter += 1
                else:
                    # ACCUMULATING_SILENCE → ACCUMULATING_SILENCE or IDLE
                    s.silence_energy += (1.0 - speech_prob)

                    self._append_to_speech_buffer(audio, timestamp, is_speech=False, chunk_id=s.chunk_id_counter)
                    s.chunk_id_counter += 1

                    if self.verbose:
                        logging.debug(f"SoundPreProcessor: silence_energy={s.silence_energy:.2f}")

                    if s.silence_energy >= self.silence_energy_threshold:
                        # ACCUMULATING_SILENCE → IDLE
                        segment = self._build_audio_segment(breakpoint_idx=s.silence_start_idx)
                        self.speech_queue.put(segment)
                        self.windower.process_segment(segment)
                        self._reset_segment_state()

                        self.state = ProcessingStatesEnum.IDLE
                        s.consecutive_speech_count = 0

                        if self.verbose:
                            logging.debug(f"SoundPreProcessor: silence_energy_threshold reached")
                            logging.debug(f"SoundPreProcessor: emitting preliminary segment")
                            logging.debug(f"  chunk_ids={segment.chunk_ids}")
                            logging.debug(f"  left_context={len(segment.left_context)/512} chunks, right_context={len(segment.right_context)/512} chunks")

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

    def _append_to_idle_buffer(self, audio: np.ndarray, timestamp: float, is_speech: bool) -> None:
        """Append chunk dict to circular idle_buffer (only during IDLE).

        Args:
            audio: Audio chunk data
            timestamp: Timestamp of chunk
            is_speech: Whether chunk is speech
        """
        chunk = {
            'audio': audio,
            'timestamp': timestamp,
            'chunk_id': None,
            'is_speech': is_speech
        }
        self.audio_state.idle_buffer.append(chunk)

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
        self.audio_state.speech_buffer.append(chunk)

    def _initialize_speech_buffer_from_idle(self, chunks_from_idle: list) -> None:
        """Initialize speech_buffer from chunks extracted from idle_buffer.

        Called when transitioning WAITING_CONFIRMATION → ACTIVE_SPEECH.
        Assigns chunk_ids and populates speech_buffer with first speech chunks.
        Sets speech_start_time to first chunk timestamp.

        Args:
            chunks_from_idle: List of chunks extracted from end of idle_buffer
                             (typically CONSECUTIVE_SPEECH_CHUNKS - 1 chunks)
        """
        s = self.audio_state
        s.speech_buffer = []
        for chunk in chunks_from_idle:
            # Assign chunk_id from counter
            chunk_with_id = {
                'audio': chunk['audio'],
                'timestamp': chunk['timestamp'],
                'chunk_id': s.chunk_id_counter,
                'is_speech': chunk['is_speech']
            }
            s.speech_buffer.append(chunk_with_id)
            s.chunk_id_counter += 1

        if s.speech_buffer:
            s.speech_start_time = s.speech_buffer[0]['timestamp']

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
        s = self.audio_state
        left_context = (np.concatenate(s.left_context_snapshot)
                       if s.left_context_snapshot
                       else np.array([], dtype=np.float32))

        if breakpoint_idx is None:
            # Hard cut: all data, no right_context (no silence boundary exists)
            data_buffer = s.speech_buffer

            data_chunks = [chunk['audio'] for chunk in data_buffer]
            data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float32)

            right_context = np.array([], dtype=np.float32)

            chunk_ids = [chunk['chunk_id'] for chunk in data_buffer
                        if chunk['chunk_id'] is not None]
        else:
            # Breakpoint split: split at breakpoint_idx (inclusive)
            data_buffer = s.speech_buffer[0:breakpoint_idx+1]
            right_context_buffer = s.speech_buffer[breakpoint_idx+1:breakpoint_idx+7]  # max 6 chunks

            data_chunks = [chunk['audio'] for chunk in data_buffer]
            data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float32)

            right_context_chunks = [chunk['audio'] for chunk in right_context_buffer]
            right_context = (np.concatenate(right_context_chunks)
                            if right_context_chunks
                            else np.array([], dtype=np.float32))

            chunk_ids = [chunk['chunk_id'] for chunk in data_buffer
                        if chunk['chunk_id'] is not None]

        end_time = s.speech_start_time + (len(data) / self.sample_rate)

        return AudioSegment(
            type='preliminary',
            data=data,
            left_context=left_context,
            right_context=right_context,
            start_time=s.speech_start_time,
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
        s = self.audio_state
        s.speech_buffer = s.speech_buffer[breakpoint_idx+1:]
        if s.speech_buffer:
            s.speech_start_time = s.speech_buffer[0]['timestamp']
        s.silence_energy = 0.0

    def _reset_segment_state(self) -> None:
        """Reset segment state after finalization.

        Delegates to AudioProcessingState.reset_segment() which clears:
        - speech_buffer
        - silence_energy
        - silence_start_idx
        - left_context_snapshot
        """
        self.audio_state.reset_segment()

    def _handle_max_duration_split(self, timestamp: float) -> None:
        """Handle segment splitting when max_speech_duration_ms is reached.

        Searches backward for natural silence breakpoint, emits segment, and keeps remainder.
        If no breakpoint found, performs hard cut and resets buffer while staying in ACTIVE_SPEECH.

        Strategy:
        - Searches backward from end of speech_buffer, skipping last 3 chunks
        - Ensures right_context isn't empty (at least 3 chunks reserved)
        - Breakpoint split: emits segment, keeps remainder in buffer
        - Hard cut: emits entire buffer, resets for continuation

        Args:
            timestamp: Current chunk timestamp for logging
        """
        s = self.audio_state
        if self.verbose:
            offset = timestamp - s.first_chunk_timestamp
            logging.debug(f"----------------------------")
            logging.debug(f"SoundPreProcessor: max_speech_duration reached, splitting at {offset:.2f}")

        # Search for silence breakpoint (skip last 3 chunks for right_context)
        breakpoint_idx = None
        if len(s.speech_buffer) >= 4:
            search_end = len(s.speech_buffer) - 3
            for i in range(search_end - 1, -1, -1):
                if not s.speech_buffer[i].get('is_speech', True):
                    breakpoint_idx = i
                    break

        if breakpoint_idx is not None:
            # Breakpoint split
            if self.verbose:
                breakpoint_file_time = s.speech_buffer[breakpoint_idx]['timestamp']
                offset = breakpoint_file_time - s.first_chunk_timestamp
                logging.debug(f"SoundPreProcessor: silence breakpoint at chunk {breakpoint_idx}, offset={offset:.2f}s)")

            segment = self._build_audio_segment(breakpoint_idx=breakpoint_idx)
            self.speech_queue.put(segment)
            self.windower.process_segment(segment)

            # Capture breakpoint chunk as left context for next segment
            breakpoint_chunk = s.speech_buffer[breakpoint_idx]
            s.left_context_snapshot = [breakpoint_chunk['audio']]

            self._keep_remainder_after_breakpoint(breakpoint_idx)

            if self.verbose:
                logging.debug(f"SoundPreProcessor: emitting preliminary segment")
                logging.debug(f"  chunk_ids={segment.chunk_ids}")
                logging.debug(f"  left_context={len(segment.left_context)/512} chunks, right_context={len(segment.right_context)/512} chunks")

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
            s.speech_buffer = []
            s.speech_start_time = timestamp
            s.left_context_snapshot = None  # No context for hard cut

    def flush(self) -> None:
        """Emit pending segment and flush windower.

        Useful for end-of-stream or testing scenarios.
        """
        s = self.audio_state
        if self.is_speech_active and len(s.speech_buffer) > 0:
            segment = self._build_audio_segment(breakpoint_idx=None)
            self.speech_queue.put(segment)
            self.windower.flush(segment)  # Process and flush in one call
            self._reset_segment_state()
            self.state = ProcessingStatesEnum.IDLE
        else:
            self.windower.flush()

        if self.verbose:
            logging.debug("SoundPreProcessor: flush()")
