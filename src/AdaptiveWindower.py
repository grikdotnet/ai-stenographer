# src/AdaptiveWindower.py
import numpy as np
import logging
import queue
from typing import Dict, List, Any, Optional
from src.types import AudioSegment


class AdaptiveWindower:
    """Aggregates speech segments into recognition windows with overlap.

    Takes speech segments from VAD and combines them into recognition windows
    optimized for speech recognition accuracy.

    Windowing Strategy:
    - Window requires 2+ segments to overlap with next window
    - Window total audio duration >= window_duration
    - After emission, trailing segments >= MIN_OVERLAP_DURATION stay in buffer
    - Flush emits remaining segments regardless of count/duration

    Args:
        speech_queue: Queue to output recognition windows (finalized AudioSegments)
        config: Configuration dictionary with audio and windowing parameters
    """

    MIN_OVERLAP_DURATION: float = 1.0  # seconds - minimum overlap for context continuity

    def __init__(self, speech_queue: queue.Queue, config: Dict[str, Any], verbose: bool = False):
        self.speech_queue: queue.Queue = speech_queue
        self.config: Dict[str, Any] = config
        self.verbose: bool = verbose

        self.sample_rate: int = config['audio']['sample_rate']
        self.window_duration: float = config['windowing']['window_duration']  # 3.0 seconds

        # Segment buffer for windowing
        self.segments: List[AudioSegment] = []
        self.first_window_emitted: bool = False

    def process_segment(self, segment: AudioSegment) -> None:
        """Process incoming VAD segment and create windows when ready.

        Accumulates segments and emits windows when:
        1. Buffer has 2+ segments (for overlap)
        2. Total audio duration >= window_duration

        Args:
            segment: Preliminary AudioSegment from VAD with type='preliminary'
        """
        self.segments.append(segment)

        # Need 2+ segments AND sufficient duration
        if len(self.segments) >= 2:
            total_samples = sum(len(seg.data) for seg in self.segments)
            total_duration = total_samples / self.sample_rate
            if total_duration >= self.window_duration:
                self._emit_window()

    def _emit_window(self) -> None:
        """Create and emit a recognition window from buffered segments.

        Aggregates all segments in buffer, concatenates audio data,
        collects chunk IDs, and keeps last segment for overlap.
        """
        if not self.segments:
            return

        # Concatenate all audio data
        audio_parts = [seg.data for seg in self.segments]
        window_audio = np.concatenate(audio_parts)

        # Calculate actual window timing from segments
        actual_start = self.segments[0].start_time
        actual_end = self.segments[-1].end_time

        # Collect unique chunk_ids from all segments in window
        chunk_ids = []
        for seg in self.segments:
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

        # Keep trailing segments with at least MIN_OVERLAP_DURATION for next window
        overlap_segments = []
        overlap_samples = 0
        min_overlap_samples = int(self.MIN_OVERLAP_DURATION * self.sample_rate)

        for seg in reversed(self.segments):
            overlap_segments.insert(0, seg)
            overlap_samples += len(seg.data)
            if overlap_samples >= min_overlap_samples:
                break

        self.segments = overlap_segments

    def flush(self, final_segment: Optional[AudioSegment] = None) -> None:
        """Flush any remaining segments as a final window.

        Called at a speech pause to emit the last partial window
        even if it doesn't meet the 2-segment or duration requirements.

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
