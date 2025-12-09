# src/AdaptiveWindower.py
import numpy as np
import logging
import queue
from typing import Dict, List, Any, Optional
from src.types import AudioSegment


class AdaptiveWindower:
    """Aggregates word-level VAD segments into recognition windows.

    Takes word-level speech segments from VAD and combines them into
    overlapping recognition windows optimized for speech recognition.

    Windowing Strategy:
    - Window duration and step size are defined in config
    - Preserves word-level timing metadata

    Args:
        speech_queue: Queue to output recognition windows (finalized AudioSegments)
        config: Configuration dictionary with audio and windowing parameters
    """

    def __init__(self, speech_queue: queue.Queue, config: Dict[str, Any], verbose: bool = False):
        self.speech_queue: queue.Queue = speech_queue
        self.config: Dict[str, Any] = config
        self.verbose: bool = verbose

        self.sample_rate: int = config['audio']['sample_rate']
        self.window_duration: float = config['windowing']['window_duration']  # 3.0 seconds
        self.step_size: float = config['windowing']['step_size']  # 1.0 second

        # Segment buffer for windowing
        self.segments: List[AudioSegment] = []
        self.window_start_time: float = 0.0
        self.last_window_time: float = 0.0

        self.first_window_emitted: bool = False

    def process_segment(self, segment: AudioSegment) -> None:
        """Process incoming VAD segment and create windows when ready.

        Accumulates word-level segments and emits windows when:
        1. Window reaches configured duration (3s)
        2. Enough time has passed since last window (step_size = 1s)

        Args:
            segment: Preliminary AudioSegment from VAD with type='preliminary'
        """
        self.segments.append(segment)

        # Initialize window start time on first segment
        if len(self.segments) == 1:
            self.window_start_time = segment.start_time
            self.last_window_time = segment.start_time

        # Check if we should emit a window
        current_time = segment.end_time
        buffered_duration = current_time - self.window_start_time

        # Emit window if:
        # 1. We've accumulated enough audio for a full window (3s)
        # 2. And we've passed the step boundary (1s since last window)
        if buffered_duration >= self.window_duration:
            time_since_last_window = current_time - self.last_window_time
            if time_since_last_window >= self.step_size:
                self._emit_window()


    def _emit_window(self) -> None:
        """Create and emit a recognition window from buffered segments.

        Aggregates segments within the window duration, concatenates audio data,
        and collects chunk IDs from all preliminary segments.
        """
        if not self.segments:
            return

        # Calculate window end time
        window_end_time = self.window_start_time + self.window_duration

        # Collect segments within this window
        window_segments = []
        audio_parts = []

        for segment in self.segments:
            # Include segment if it overlaps with window
            if segment.start_time < window_end_time:
                window_segments.append(segment)
                audio_parts.append(segment.data)

        if not window_segments:
            return

        # Concatenate all audio data
        window_audio = np.concatenate(audio_parts)

        # Calculate actual window timing from segments
        actual_start = window_segments[0].start_time
        actual_end = window_segments[-1].end_time

        # Collect unique chunk_ids from all segments in window
        chunk_ids = []
        for seg in window_segments:
            chunk_ids.extend(seg.chunk_ids)
        unique_chunk_ids = sorted(set(chunk_ids))

        # First window gets left_context from first segment, subsequent windows get empty
        left_ctx = np.array([], dtype=np.float32)
        if not self.first_window_emitted and self.segments:
            left_ctx = self.segments[0].left_context
            self.first_window_emitted = True

        # Create finalized AudioSegment
        window = AudioSegment(
            type='finalized',
            data=window_audio,
            left_context=left_ctx,
            right_context=np.array([], dtype=np.float32),  # No right context for non-flush windows
            start_time=actual_start,
            end_time=actual_end,
            chunk_ids=unique_chunk_ids
        )

        if self.verbose:
            duration_ms = len(window_audio) / self.sample_rate * 1000
            logging.debug(f"AdaptiveWindower: created window duration={duration_ms:.0f}ms,"
                          +f" chunk_ids={unique_chunk_ids}, samples={len(window_audio)}")

        self.speech_queue.put(window)

        if self.verbose:
            logging.debug(f"AdaptiveWindower: put finalized window to queue (chunk_ids={unique_chunk_ids})")

        # Update window timing for next window
        self.last_window_time = self.window_start_time
        self.window_start_time = self.window_start_time + self.step_size

        # Remove segments that will never be needed again
        self.segments = [
            seg for seg in self.segments
            if seg.end_time > self.window_start_time
        ]


    def flush(self, final_segment: Optional[AudioSegment] = None) -> None:
        """Flush any remaining segments as a final window.

        Called at a speech pause to emit the last partial window
        even if it doesn't reach the full 3-second duration.

        Args:
            final_segment: Optional final segment to process before flushing.
                          If provided, appends it to segments and uses its right_context.
        """
        if final_segment is not None:
            self.segments.append(final_segment)

        if not self.segments:
            self._reset_state()
            return

        # Emit remaining segments as final window
        audio_parts = [seg.data for seg in self.segments]
        window_audio = np.concatenate(audio_parts)

        actual_start = self.segments[0].start_time
        actual_end = self.segments[-1].end_time

        # Collect unique chunk_ids from all segments
        chunk_ids = []
        for seg in self.segments:
            chunk_ids.extend(seg.chunk_ids)
        unique_chunk_ids = sorted(set(chunk_ids))

        # First window gets left_context, flush gets right_context from last segment
        left_ctx = np.array([], dtype=np.float32)
        if not self.first_window_emitted:
            left_ctx = self.segments[0].left_context

        right_ctx = self.segments[-1].right_context

        # Create flush AudioSegment
        window = AudioSegment(
            type='flush',
            data=window_audio,
            left_context=left_ctx,
            right_context=right_ctx,
            start_time=actual_start,
            end_time=actual_end,
            chunk_ids=unique_chunk_ids
        )

        self.speech_queue.put(window)

        if self.verbose:
            logging.debug(f"AdaptiveWindower.flush(): put to queue (chunk_ids={unique_chunk_ids})")

        # Reset state for next speech sequence
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset windower state for next speech sequence."""
        self.segments = []
        self.first_window_emitted = False
        self.window_start_time = 0.0
        self.last_window_time = 0.0
