"""Type definitions for STT pipeline queue messages."""

from dataclasses import dataclass, field
from typing import Literal, Union
import numpy as np
import numpy.typing as npt


@dataclass
class AudioSegment:
    """Speech segment for recognition."""

    type: Literal["incremental"]
    data: npt.NDArray[np.float32]
    left_context: npt.NDArray[np.float32]
    right_context: npt.NDArray[np.float32]
    start_time: float
    end_time: float
    utterance_id: int = 0
    chunk_ids: list[int] = field(default_factory=list)
    message_id: int | None = None

    def __post_init__(self) -> None:
        """Validate that at least one chunk ID is present."""
        if len(self.chunk_ids) < 1:
            raise ValueError("Must have at least one chunk_id")


ChunkQueueItem = AudioSegment


@dataclass
class SpeechEndSignal:
    """Boundary signal emitted when an utterance ends."""

    utterance_id: int
    end_time: float
    message_id: int | None = None


@dataclass(kw_only=True)
class RecognitionResult:
    """Speech recognition output with timing and confidence information."""

    text: str
    start_time: float
    end_time: float
    chunk_ids: list[int] = field(default_factory=list)
    confidence: float = 0.0
    token_confidences: list[float] = field(default_factory=list)
    audio_rms: float = 0.0
    confidence_variance: float = 0.0


@dataclass
class RecognitionTextMessage:
    """Recognizer text output tagged with message id."""

    result: RecognitionResult
    message_id: int


@dataclass
class RecognizerAck:
    """Terminal acknowledgement for a consumed audio segment."""

    message_id: int
    ok: bool = True
    error: str | None = None


@dataclass
class RecognizerFreeSignal:
    """Control signal emitted when recognizer becomes free (ACKs in-flight segment).

    Used for ACK-driven early hard-cut: preprocessor can cut buffered speech
    earlier than max_duration when recognizer capacity is available.
    """
    seq: int              # Monotonic sequence number (never resets)
    message_id: int       # AudioSegment message_id that was ACKed
    utterance_id: int     # utterance_id from the ACKed segment


RouterControlItem = RecognizerFreeSignal
SpeechQueueItem = Union[AudioSegment, SpeechEndSignal]
RecognizerOutputItem = Union[RecognitionTextMessage, RecognizerAck]
MatcherQueueItem = Union[RecognitionResult, SpeechEndSignal]


@dataclass
class DisplayInstructions:
    """Pure data DTO - separates WHAT to display from HOW to display it.

    Carries formatting decisions from TextFormatter to TextDisplayWidget.
    TextFormatter calculates these instructions based on recognition results.
    TextDisplayWidget renders these instructions to tkinter GUI.

    Attributes:
        action: Type of display operation to perform
        text_to_append: Text to append for partial updates (pre-formatted with spacing)
        finalized_text: Complete finalized text buffer (for finalize/rerender operations)
        preliminary_segments: Remaining preliminary results to display (for rerender operations)
        paragraph_break_inserted: Whether a paragraph break was just inserted
    """
    action: Literal['update_partial', 'finalize', 'rerender_all']
    text_to_append: str = ""
    finalized_text: str = ""
    preliminary_segments: list['RecognitionResult'] = field(default_factory=list)
    paragraph_break_inserted: bool = False
