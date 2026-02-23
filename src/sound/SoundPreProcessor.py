# src/sound/SoundPreProcessor.py
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
from typing import Dict, Any, TYPE_CHECKING, Optional
from ..types import AudioSegment, RecognizerFreeSignal, SpeechEndSignal

if TYPE_CHECKING:
    from src.asr.VoiceActivityDetector import VoiceActivityDetector
    from src.sound.GrowingWindowAssembler import GrowingWindowAssembler
    from src.ApplicationState import ApplicationState


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
    utterance_id_counter: int = 0
    current_utterance_id: int | None = None
    utterance_open: bool = False

    # Buffering - idle_buffer maxlen=20 (~640ms at 32ms/chunk)
    speech_buffer: list[dict[str, Any]] = field(default_factory=list)
    idle_buffer: deque = field(default_factory=lambda: deque(maxlen=CONTEXT_BUFFER_SIZE))

    # Timing
    speech_start_time: float = 0.0
    last_speech_timestamp: float = 0.0
    first_chunk_timestamp: float = 0.0

    # Silence tracking
    silence_energy: float = 0.0

    # Context
    left_context_snapshot: list[np.ndarray] | None = None

    def reset_segment(self) -> None:
        """Reset state after segment finalization.

        Clears segment buffers and silence energy while preserving timeline markers
        used for boundary events and diagnostics.
        """
        self.speech_buffer = []
        self.silence_energy = 0.0
        self.left_context_snapshot = None

    def reset_silence_tracking(self) -> None:
        """Reset silence tracking when speech resumes."""
        self.silence_energy = 0.0


# ========================================================================
# Main Component
# ========================================================================

class SoundPreProcessor:
    """Processes raw audio chunks through VAD and buffering.

    Runs in dedicated thread to avoid blocking audio capture.
    Orchestrates VAD, speech buffering, and GrowingWindowAssembler.

    Idle Buffer Strategy:
    - Maintains circular buffer of last 48 chunks (1.536s @ 32ms/chunk)
    - Accumulates during IDLE and WAITING_CONFIRMATION states
    - Stops accumulating during ACTIVE_SPEECH and ACCUMULATING_SILENCE
    - On speech confirmation: extracts last (N-1) chunks to initialize speech_buffer
    - left_context: idle_buffer snapshot (excluding extracted chunks) at speech start
    - right_context: always empty (uniform hard-cut model)
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

    Args:
        chunk_queue: Queue to read raw audio chunks from AudioSource
        speech_queue: Queue to write AudioSegments (incremental + flush)
        vad: VoiceActivityDetector instance
        windower: GrowingWindowAssembler instance (called synchronously)
        config: Configuration dictionary
        verbose: Enable verbose logging
    """

    # Consecutive speech chunks required before starting segment (false positive prevention)
    CONSECUTIVE_SPEECH_CHUNKS: int = 3

    def __init__(self,
                 chunk_queue: queue.Queue,
                 speech_queue: queue.Queue,
                 vad: VoiceActivityDetector,
                 windower: GrowingWindowAssembler,
                 config: Dict[str, Any],
                 control_queue: queue.Queue,
                 app_state: Optional['ApplicationState'] = None,
                 verbose: bool = False):

        self.chunk_queue: queue.Queue = chunk_queue      # INPUT: raw audio
        self.speech_queue: queue.Queue = speech_queue    # OUTPUT: incremental segments
        self.vad: VoiceActivityDetector = vad            # Called for speech detection
        self.windower: GrowingWindowAssembler = windower # Called synchronously

        # Configuration
        self.sample_rate: int = config['audio']['sample_rate']
        self.frame_duration_ms: int = config['vad']['frame_duration_ms']
        self.max_speech_duration_ms: int = config['windowing']['max_speech_duration_ms']
        self.silence_energy_threshold: float = config['audio']['silence_energy_threshold']

        # RMS normalization (AGC)
        rms_config = config['audio']['rms_normalization']
        self.target_rms: float = rms_config['target_rms']
        self.rms_silence_threshold: float = rms_config['silence_threshold']
        self.gain_smoothing: float = rms_config['gain_smoothing']
        self.current_gain: float = 1.0  # Start with unity gain

        # Audio processing state (all mutable segment-related state)
        self.audio_state = AudioProcessingState()

        # Control queue for ACK-driven early cut
        self.control_queue: queue.Queue = control_queue

        # Config with fallback (no strict key requirement in this PR)
        self.min_segment_duration_ms: int = config['windowing'].get('min_segment_duration_ms', 200)

        # ACK sequencing state
        self._last_ack_seq_processed: int = -1
        self._pending_ack_signal: RecognizerFreeSignal | None = None

        # Threading control
        self.is_running: bool = False
        self.thread: threading.Thread | None = None
        self.verbose: bool = verbose
        self.app_state = app_state
        self.dropped_segments: int = 0

        # Register as component observer if app_state provided
        if self.app_state:
            self.app_state.register_component_observer(self.on_state_change)

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

        # Check for ACK-driven cut BEFORE state machine
        self._check_ack_signals(timestamp)

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

            case ProcessingStatesEnum.WAITING_CONFIRMATION:
                self._append_to_idle_buffer(audio, timestamp, is_speech=is_speech)

                if is_speech:
                    s.consecutive_speech_count += 1

                    if s.consecutive_speech_count >= self.CONSECUTIVE_SPEECH_CHUNKS:
                        # WAITING_CONFIRMATION → ACTIVE_SPEECH
                        self.state = ProcessingStatesEnum.ACTIVE_SPEECH
                        self._start_new_utterance()

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
                    s.silence_energy += (1.0 - speech_prob)

                    self._append_to_speech_buffer(audio, timestamp, is_speech=False, chunk_id=s.chunk_id_counter)
                    s.chunk_id_counter += 1

            case ProcessingStatesEnum.ACCUMULATING_SILENCE:
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

                    if s.silence_energy >= self.silence_energy_threshold:
                        # ACCUMULATING_SILENCE → IDLE
                        segment = self._build_audio_segment()
                        self.windower.flush(segment)
                        self._reset_segment_state()
                        self._emit_speech_end_if_open(end_time=segment.end_time)

                        self.state = ProcessingStatesEnum.IDLE
                        s.consecutive_speech_count = 0

                        if self.verbose:
                            logging.debug(f"SoundPreProcessor: silence_energy_threshold reached")
                            logging.debug(f"SoundPreProcessor: emitting segment")
                            logging.debug(f"  chunk_ids=[{segment.chunk_ids[0] if segment.chunk_ids else ''}...{segment.chunk_ids[-1] if segment.chunk_ids else ''}]")
                            logging.debug(f"  left_context={len(segment.left_context)/512} chunks, right_context={len(segment.right_context)/512} chunks")

        self._check_pending_ack_cut(timestamp)

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

    def _put_segment_nonblocking(self, segment: AudioSegment) -> None:
        """Enqueue segment to speech_queue with drop-on-full behavior."""
        try:
            self.speech_queue.put_nowait(segment)
        except queue.Full:
            self.dropped_segments += 1
            if self.verbose:
                logging.warning(
                    f"SoundPreProcessor: speech_queue full, dropped segment "
                    f"(total drops: {self.dropped_segments})"
                )

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

    def _check_ack_signals(self, current_timestamp: float) -> None:
        """Check control_queue for RecognizerFreeSignal and execute ACK-cut if eligible.

        Algorithm:
        1. Drain queue, keeping only the last (highest-seq) signal.
        2. If a new signal arrived and is eligible (seq/utterance/state):
           a. Buffer sufficient: execute cut immediately.
           b. Buffer too small: store as _pending_ack_signal (replacing any prior lower-seq pending).
        3. Return early when a new signal was processed — _process_chunk's bottom call handles
           the pending check for this iteration.
        4. If no new signal arrived: delegate to _check_pending_ack_cut.

        Args:
            current_timestamp: Timestamp of current chunk
        """
        signal: RecognizerFreeSignal | None = None
        try:
            while True:
                signal = self.control_queue.get_nowait()
        except queue.Empty:
            pass

        if signal is not None:
            if self.verbose:
                logging.debug(
                    "SoundPreProcessor: received RecognizerFreeSignal seq=%d message_id=%d utterance_id=%d",
                    signal.seq, signal.message_id, signal.utterance_id
                )
            if self._is_signal_eligible(signal):
                buffered_ms = len(self.audio_state.speech_buffer) * self.frame_duration_ms
                if buffered_ms >= self.min_segment_duration_ms:
                    self._execute_ack_cut(current_timestamp)
                    self._last_ack_seq_processed = signal.seq
                else:
                    if self.verbose:
                        logging.debug(
                            "SoundPreProcessor: deferring ACK-cut seq=%d buffered_ms=%d < min=%d",
                            signal.seq, buffered_ms, self.min_segment_duration_ms
                        )
                    if (self._pending_ack_signal is None
                            or signal.seq > self._pending_ack_signal.seq):
                        self._pending_ack_signal = signal
            return

        self._check_pending_ack_cut(current_timestamp)

    def _check_pending_ack_cut(self, current_timestamp: float) -> None:
        """Fire deferred ACK-cut if pending signal is eligible and buffer threshold is now met.

        Algorithm:
        1. Return immediately if no pending signal.
        2. Re-check eligibility (seq/utterance/state); clear and return if stale.
        3. Measure buffered_ms; fire cut and clear pending if threshold met.

        Args:
            current_timestamp: Timestamp of current chunk
        """
        if self._pending_ack_signal is None:
            return
        pending = self._pending_ack_signal
        if not self._is_signal_eligible(pending):
            if self.verbose:
                logging.debug(
                    "SoundPreProcessor: clearing stale pending ACK-cut seq=%d", pending.seq
                )
            self._pending_ack_signal = None
            return
        buffered_ms = len(self.audio_state.speech_buffer) * self.frame_duration_ms
        if buffered_ms >= self.min_segment_duration_ms:
            if self.verbose:
                logging.debug(
                    "SoundPreProcessor: executing deferred ACK-cut seq=%d buffered_ms=%d",
                    pending.seq, buffered_ms
                )
            self._execute_ack_cut(current_timestamp)
            self._last_ack_seq_processed = pending.seq
            self._pending_ack_signal = None

    def _is_signal_eligible(self, signal: RecognizerFreeSignal) -> bool:
        """Check if signal passes seq, utterance, and state eligibility.

        Does NOT check buffer duration — that is a timing concern handled separately
        by _check_ack_signals and _check_pending_ack_cut.

        Args:
            signal: RecognizerFreeSignal from control_queue

        Returns:
            True if signal has a fresh seq, matches the current utterance,
            and the processor is in an ACK-cuttable state.
        """
        s = self.audio_state
        if (signal.seq <= self._last_ack_seq_processed
                or s.current_utterance_id is None
                or signal.utterance_id != s.current_utterance_id):
            if self.verbose:
                logging.debug(
                    "SoundPreProcessor: RecognizerFreeSignal seq=%d not eligible"
                    " (last_seq=%d, current_utterance_id=%s)",
                    signal.seq, self._last_ack_seq_processed, s.current_utterance_id
                )
            return False

        if s.state not in (ProcessingStatesEnum.ACTIVE_SPEECH, ProcessingStatesEnum.ACCUMULATING_SILENCE):
            if self.verbose:
                logging.debug(
                    "SoundPreProcessor: RecognizerFreeSignal seq=%d not eligible (state=%s)",
                    signal.seq, s.state
                )
            return False

        return True

    def _execute_ack_cut(self, timestamp: float) -> None:
        """Execute ACK-driven hard-cut.

        Buffer operation only — emits accumulated audio and resets the buffer.
        Does not modify state machine state: the state machine determines state
        from VAD results independently of buffer cuts.

        Algorithm:
        1. Build segment (hard cut, no right_context)
        2. Emit to windower
        3. Reset buffer

        Args:
            timestamp: Current chunk timestamp for speech_start_time
        """
        s = self.audio_state

        segment = self._build_audio_segment()
        self.windower.process_segment(segment)

        s.speech_buffer = []
        s.speech_start_time = timestamp
        s.left_context_snapshot = None
        self._pending_ack_signal = None

        if self.verbose:
            logging.info(
                "SoundPreProcessor: ACK-cut at %.2fs (buffered_ms=%d, chunk_ids=[%s...%s])",
                timestamp - s.first_chunk_timestamp,
                len(segment.chunk_ids) * self.frame_duration_ms,
                segment.chunk_ids[0] if segment.chunk_ids else '',
                segment.chunk_ids[-1] if segment.chunk_ids else ''
            )

    def _build_audio_segment(self) -> AudioSegment:
        """Build AudioSegment from speech_buffer using a hard cut.

        Extracts:
        - left_context from left_context_snapshot
        - data from all chunks in speech_buffer
        - right_context is always empty (uniform hard-cut model)

        Returns:
            AudioSegment ready to emit
        """
        s = self.audio_state
        left_context = (np.concatenate(s.left_context_snapshot)
                       if s.left_context_snapshot
                       else np.array([], dtype=np.float32))

        data_chunks = [chunk['audio'] for chunk in s.speech_buffer]
        data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float32)
        chunk_ids = [chunk['chunk_id'] for chunk in s.speech_buffer
                     if chunk['chunk_id'] is not None]

        end_time = s.speech_start_time + (len(data) / self.sample_rate)

        return AudioSegment(
            type='incremental',
            data=data,
            left_context=left_context,
            right_context=np.array([], dtype=np.float32),
            start_time=s.speech_start_time,
            end_time=end_time,
            utterance_id=s.current_utterance_id if s.current_utterance_id is not None else 0,
            chunk_ids=chunk_ids
        )

    def _reset_segment_state(self) -> None:
        """Reset segment state after finalization.

        Delegates to AudioProcessingState.reset_segment() which clears:
        - speech_buffer
        - silence_energy
        - left_context_snapshot
        Also clears any pending ACK signal to prevent stale deferred cuts.
        """
        self.audio_state.reset_segment()
        self._pending_ack_signal = None

    def _handle_max_duration_split(self, timestamp: float) -> None:
        """Handle segment splitting when max_speech_duration_ms is reached.

        Always performs a hard cut: emits all buffered audio, resets the buffer,
        and stays in ACTIVE_SPEECH to continue accumulating.

        Args:
            timestamp: Current chunk timestamp for logging
        """
        s = self.audio_state
        segment = self._build_audio_segment()
        self.windower.process_segment(segment)

        if self.verbose:
            offset = timestamp - s.first_chunk_timestamp
            logging.debug(f"----------------------------")
            logging.debug(f"SoundPreProcessor: max_speech_duration reached, splitting at {offset:.2f}")
            logging.debug(f"  chunk_ids=[{segment.chunk_ids[0] if segment.chunk_ids else ''}...{segment.chunk_ids[-1] if segment.chunk_ids else ''}]")

        s.speech_buffer = []
        s.speech_start_time = timestamp
        s.left_context_snapshot = None
        self._pending_ack_signal = None

    def flush(self) -> None:
        """Emit pending segment and flush windower.

        Useful for end-of-stream or testing scenarios.
        """
        s = self.audio_state
        if self.is_speech_active and len(s.speech_buffer) > 0:
            segment = self._build_audio_segment()
            self.windower.flush(segment)  # Process and flush in one call
            self._reset_segment_state()
            self.state = ProcessingStatesEnum.IDLE
            self._emit_speech_end_if_open(end_time=segment.end_time)
        else:
            self.windower.flush()
            self._emit_speech_end_if_open(end_time=s.last_speech_timestamp)

        if self.verbose:
            logging.debug("SoundPreProcessor: flush()")

    def _start_new_utterance(self) -> None:
        """Create a new utterance identity when speech is confirmed."""
        s = self.audio_state
        s.utterance_id_counter += 1
        s.current_utterance_id = s.utterance_id_counter
        s.utterance_open = True

    def _emit_speech_end_if_open(self, end_time: float) -> None:
        """Emit speech-end boundary only when an utterance is currently open."""
        s = self.audio_state
        if not s.utterance_open or s.current_utterance_id is None:
            return

        boundary = SpeechEndSignal(
            utterance_id=s.current_utterance_id,
            end_time=end_time
        )
        try:
            self.speech_queue.put_nowait(boundary)
        except queue.Full:
            self.dropped_segments += 1
            if self.verbose:
                logging.warning(
                    f"SoundPreProcessor: speech_queue full, dropped boundary "
                    f"(total drops: {self.dropped_segments})"
                )

        s.utterance_open = False
        s.current_utterance_id = None

    def on_state_change(self, old_state: str, new_state: str) -> None:
        """
        Observes ApplicationState and reacts to:
        paused: flush pending segments
        shutdown: stop processing and flush

        Args:
            old_state: Previous state
            new_state: New state
        """
        if new_state == 'paused':
            self.flush()
        elif new_state == 'shutdown':
            self.stop()
