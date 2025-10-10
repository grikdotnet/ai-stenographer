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
        data: Audio data as float32 numpy array
        start_time: Segment start time in seconds
        end_time: Segment end time in seconds
        chunk_ids: List of chunk IDs - [single_id] for preliminary, [id1, id2, ...] for finalized/flush
    """
    type: Literal['preliminary', 'finalized', 'flush']
    data: npt.NDArray[np.float32]
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
