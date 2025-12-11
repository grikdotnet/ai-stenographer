"""Type definitions for STT pipeline queue messages."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal
import numpy as np
import numpy.typing as npt

# Buffer size constant for idle buffer (~640ms at 32ms/chunk)
CONTEXT_BUFFER_SIZE = 20


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


@dataclass
class AudioProcessingState:
    """State container for audio segment processing in SoundPreProcessor.

    Groups all mutable state related to audio segment accumulation,
    leaving SoundPreProcessor with logic and technical infrastructure.
    """

    # State machine
    state: ProcessingState = ProcessingState.IDLE
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
    left_context_snapshot: list[npt.NDArray[np.float32]] | None = None

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


@dataclass
class AudioSegment:
    """Speech segment for recognition - either preliminary word, finalized window, or flush signal.

    Preliminary segments represent individual words detected by VAD with a single chunk ID.
    Finalized windows aggregate multiple preliminary segments for high-quality recognition.
    Flush signals indicate end of speech segment (e.g., silence timeout) and trigger pending text finalization.

    Attributes:
        type: Discriminator - 'preliminary' for instant words, 'finalized' for overlapping windows, 'flush' for end-of-segment signal
        data: Speech audio data as float32 numpy array (concatenated speech chunks only)
        left_context: Pre-speech context audio (silence before speech starts)
        right_context: Post-speech context audio (trailing silence after speech ends)
        start_time: Segment start time in seconds (refers to data, not context)
        end_time: Segment end time in seconds (refers to data, not context)
        chunk_ids: List of chunk IDs - [single_id] for preliminary, [id1, id2, ...] for finalized/flush
    """
    type: Literal['preliminary', 'finalized', 'flush']
    data: npt.NDArray[np.float32]
    left_context: npt.NDArray[np.float32]
    right_context: npt.NDArray[np.float32]
    start_time: float
    end_time: float
    chunk_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        """Validate that at least one chunk ID is present."""
        if len(self.chunk_ids) < 1:
            raise ValueError("Must have at least one chunk_id")


# Type alias for chunk_queue items
ChunkQueueItem = AudioSegment


@dataclass
class RecognitionResult:
    """Speech recognition output with timing and type information.

    Attributes:
        text: Recognized text string
        start_time: Recognition start time in seconds
        end_time: Recognition end time in seconds
        status: Status discriminator - 'preliminary' for instant results, 'final' for high-quality results, 'flush' for end-of-segment signal
        chunk_ids: List of chunk IDs that contributed to this recognition result
    """
    text: str
    start_time: float
    end_time: float
    status: Literal['preliminary', 'final', 'flush']
    chunk_ids: list[int] = field(default_factory=list)
