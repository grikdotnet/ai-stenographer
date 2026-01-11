"""Type definitions for STT pipeline queue messages."""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import numpy.typing as npt


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
        confidence: Average confidence score for the segment (0.0-1.0 range, 0.0 = no confidence data)
        token_confidences: Per-token confidence scores (parallel to tokens, empty if not available)
        audio_rms: RMS energy of the audio data region (0.0 = no data or silent)
        confidence_variance: Variance of token confidence scores (0.0 = uniform confidence or no data)
    """
    text: str
    start_time: float
    end_time: float
    status: Literal['preliminary', 'final', 'flush']
    chunk_ids: list[int] = field(default_factory=list)
    confidence: float = 0.0
    token_confidences: list[float] = field(default_factory=list)
    audio_rms: float = 0.0
    confidence_variance: float = 0.0


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
