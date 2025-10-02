# src/AdaptiveWindower.py
import numpy as np
import queue
from typing import Dict, List, Any, Optional


class AdaptiveWindower:
    """Aggregates word-level VAD segments into recognition windows.

    Takes word-level speech segments from VAD and combines them into
    overlapping recognition windows optimized for speech recognition.
    Uses 3-second windows with 1-second step for good context and responsiveness.

    Windowing Strategy:
    - Window duration: 3 seconds (provides sufficient context for recognition)
    - Step size: 1 second (33% overlap for continuity between windows)
    - Preserves word-level timing metadata for downstream processing

    Args:
        window_queue: Queue to output recognition windows
        config: Configuration dictionary with audio and windowing parameters
    """

    def __init__(self, window_queue: queue.Queue, config: Dict[str, Any]):
        self.window_queue: queue.Queue = window_queue
        self.config: Dict[str, Any] = config

        self.sample_rate: int = config['audio']['sample_rate']
        self.window_duration: float = config['windowing']['window_duration']  # 3.0 seconds
        self.step_size: float = config['windowing']['step_size']  # 1.0 second

        # Segment buffer for windowing
        self.segments: List[Dict[str, Any]] = []
        self.window_start_time: float = 0.0
        self.last_window_time: float = 0.0


    def process_segment(self, segment: Dict[str, Any]) -> None:
        """Process incoming VAD segment and create windows when ready.

        Accumulates word-level segments and emits windows when:
        1. Window reaches configured duration (3s)
        2. Enough time has passed since last window (step_size = 1s)

        Args:
            segment: Speech segment from VAD containing:
                - data: Audio data as numpy array
                - start_time: Segment start timestamp
                - end_time: Segment end timestamp
                - duration: Segment duration
                - is_speech: Speech flag (should be True for VAD segments)
        """
        # Add segment to buffer
        self.segments.append(segment)

        # Initialize window start time on first segment
        if len(self.segments) == 1:
            self.window_start_time = segment['start_time']
            self.last_window_time = segment['start_time']

        # Check if we should emit a window
        current_time = segment['end_time']
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
        and preserves word-level timing metadata for downstream processing.
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
            if segment['start_time'] < window_end_time:
                window_segments.append(segment)
                audio_parts.append(segment['data'])

        if not window_segments:
            return

        # Concatenate all audio data
        window_audio = np.concatenate(audio_parts)

        # Calculate actual window timing from segments
        actual_start = window_segments[0]['start_time']
        actual_end = window_segments[-1]['end_time']

        # Create window dict
        window = {
            'data': window_audio,
            'start_time': actual_start,
            'end_time': actual_end,
            'duration': actual_end - actual_start,
            'timestamp': actual_start,
            'segments': window_segments  # Preserve word-level metadata
        }

        self.window_queue.put(window)

        # Update window timing for next window
        self.last_window_time = self.window_start_time
        self.window_start_time = self.window_start_time + self.step_size

        # Remove segments that are completely before the new window start
        # Keep segments that might overlap with next window
        self.segments = [
            seg for seg in self.segments
            if seg['end_time'] > self.window_start_time
        ]


    def flush(self) -> None:
        """Flush any remaining segments as a final window.

        Called at end of stream to emit the last partial window
        even if it doesn't reach the full 3-second duration.
        """
        if not self.segments:
            return

        # Emit remaining segments as final window
        audio_parts = [seg['data'] for seg in self.segments]
        window_audio = np.concatenate(audio_parts)

        actual_start = self.segments[0]['start_time']
        actual_end = self.segments[-1]['end_time']

        window = {
            'data': window_audio,
            'start_time': actual_start,
            'end_time': actual_end,
            'duration': actual_end - actual_start,
            'timestamp': actual_start,
            'segments': self.segments
        }

        self.window_queue.put(window)

        # Clear segments
        self.segments = []
