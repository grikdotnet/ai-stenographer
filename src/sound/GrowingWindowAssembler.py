# src/sound/GrowingWindowAssembler.py
import numpy as np
import logging
import queue
from typing import Dict, List, Any, Optional
from ..types import AudioSegment


def _format_chunk_ids(chunk_ids: list) -> str:
    return f"[{chunk_ids[0]}-{chunk_ids[-1]}]" if chunk_ids else "[]"


class GrowingWindowAssembler:
    """Assembles speech segments into growing recognition windows.

    Takes speech segments from VAD and combines them into growing windows
    that are recognized as a whole. Each new segment extends the current window.

    Growing Window Strategy:
    - Each new segment extends the current window (emit as incremental)
    - Windows grow up to max_window_duration (e.g. 7 seconds)
    - When max duration exceeded, oldest segments are dropped (sliding)
    - Flush emits remaining segments as incremental with right context

    Args:
        speech_queue: Queue to output recognition windows (AudioSegments)
        config: Configuration dictionary with audio and windowing parameters
    """

    def __init__(self, speech_queue: queue.Queue, config: Dict[str, Any], verbose: bool = False):
        self.speech_queue: queue.Queue = speech_queue
        self.config: Dict[str, Any] = config
        self.verbose: bool = verbose
        self.dropped_windows: int = 0

        self.sample_rate: int = config['audio']['sample_rate']
        self.max_window_duration: float = config['windowing']['max_window_duration']

        # Segment buffer for growing window
        self.segments: List[AudioSegment] = []
        self.first_segment_dropped: bool = False

    def _put_window_nonblocking(self, window: AudioSegment) -> None:
        """Enqueue window to speech_queue with drop-on-full behavior."""
        try:
            self.speech_queue.put_nowait(window)
        except queue.Full:
            self.dropped_windows += 1
            if self.verbose:
                logging.warning(
                    f"GrowingWindowAssembler: speech_queue full, dropped window "
                    f"(total drops: {self.dropped_windows})"
                )

    def process_segment(self, segment: AudioSegment) -> None:
        """Process incoming segment and emit growing window.

        Algorithm:
        1. Append segment to buffer
        2. Enforce max_window_duration (drop oldest segments if needed)
        3. Emit entire current window as incremental

        Args:
            segment: AudioSegment from VAD (already type='incremental')
        """
        self.segments.append(segment)
        self._validate_single_utterance()
        self._enforce_max_duration()
        self._emit_window(include_right_context=False)

    def _enforce_max_duration(self) -> None:
        """Drop oldest segments if total duration exceeds max_window_duration."""
        while self.segments:
            total_samples = sum(len(seg.data) for seg in self.segments)
            total_duration = total_samples / self.sample_rate

            if total_duration <= self.max_window_duration:
                break

            # Drop oldest segment
            dropped = self.segments.pop(0)
            if self.verbose:
                logging.debug(
                    f"GrowingWindowAssembler: dropped segment "
                    f"{_format_chunk_ids(dropped.chunk_ids)} (duration={len(dropped.data)/self.sample_rate:.2f}s)"
                )

            # After dropping original first segment, subsequent windows don't get left_context
            if not self.first_segment_dropped:
                self.first_segment_dropped = True

    def _emit_window(self, include_right_context: bool) -> None:
        """Emit current window by concatenating all segments in buffer.

        Algorithm:
        1. Concatenate audio data from all segments
        2. Collect unique chunk_ids
        3. Handle left_context (only on first window)
        4. Handle right_context (optional at utterance boundary)
        5. Create AudioSegment and enqueue
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

        # Windows include left_context until original first segment drops
        left_ctx = np.array([], dtype=np.float32)
        if not self.first_segment_dropped and self.segments:
            left_ctx = self.segments[0].left_context

        # Only flush windows get right_context
        right_ctx = np.array([], dtype=np.float32)
        if include_right_context:
            right_ctx = self.segments[-1].right_context

        utterance_ids = {seg.utterance_id for seg in self.segments}
        utterance_id = next(iter(utterance_ids))

        window = AudioSegment(
            type='incremental',
            data=window_audio,
            left_context=left_ctx,
            right_context=right_ctx,
            start_time=actual_start,
            end_time=actual_end,
            utterance_id=utterance_id,
            chunk_ids=unique_chunk_ids
        )

        if self.verbose:
            duration_ms = len(window_audio) / self.sample_rate * 1000
            logging.debug(
                f"GrowingWindowAssembler: emitted incremental window "
                f"duration={duration_ms:.0f}ms, chunk_ids={_format_chunk_ids(unique_chunk_ids)}, "
                f"samples={len(window_audio)}"
            )

        self._put_window_nonblocking(window)

    def flush(self, final_segment: Optional[AudioSegment] = None) -> None:
        """Flush any remaining segments as a final window.

        Called at a speech pause to emit the last window
        even if it doesn't meet duration requirements.

        Args:
            final_segment: Optional final segment to process before flushing.
                          If provided, appends it to segments and uses its right_context.
        """
        if final_segment is not None:
            self.segments.append(final_segment)
            self._validate_single_utterance()

        if not self.segments:
            self._reset_state()
            return

        # Emit remaining segments as final incremental window including right_context.
        self._emit_window(include_right_context=True)

        if self.verbose:
            chunk_ids = []
            for seg in self.segments:
                chunk_ids.extend(seg.chunk_ids)
            unique_chunk_ids = sorted(set(chunk_ids))
            logging.debug(
                f"GrowingWindowAssembler.flush(): emitted final incremental window "
                f"(chunk_ids={_format_chunk_ids(unique_chunk_ids)})"
            )

        # Reset state for next speech sequence
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset assembler state for next speech sequence."""
        self.segments = []
        self.first_segment_dropped = False

    def _validate_single_utterance(self) -> None:
        """Guard against mixing utterance IDs in one growing window."""
        utterance_ids = {seg.utterance_id for seg in self.segments}
        if len(utterance_ids) > 1:
            raise ValueError(
                f"GrowingWindowAssembler cannot mix utterance IDs in one window: {sorted(utterance_ids)}"
            )
