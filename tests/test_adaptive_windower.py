# tests/test_adaptive_windower.py
import pytest
import numpy as np
import queue
from src.AdaptiveWindower import AdaptiveWindower
from src.types import AudioSegment


class TestAdaptiveWindower:
    """Tests for AdaptiveWindower with word-level VAD segments.

    AdaptiveWindower receives word-level speech segments from VAD and aggregates
    them into 3-second recognition windows with 1-second overlap for optimal
    recognition accuracy.
    """

    @pytest.fixture
    def config(self):
        """Standard configuration for AdaptiveWindower."""
        return {
            'audio': {
                'sample_rate': 16000
            },
            'windowing': {
                'window_duration': 3.0,  # 3 seconds
                'step_size': 1.0  # 1 second overlap
            }
        }

    @pytest.fixture
    def word_segments(self):
        """Generate simulated word-level AudioSegments from VAD.

        Creates segments representing: "Hello world how are you"
        Each word is ~200ms with natural pauses between.
        Returns preliminary AudioSegment instances with unique chunk IDs.
        """
        sample_rate = 16000
        segments = []

        # Word: "Hello" (0.0-0.2s)
        segments.append(AudioSegment(
            type='preliminary',
            data=np.random.randn(int(0.2 * sample_rate)).astype(np.float32) * 0.1,
            start_time=0.0,
            end_time=0.2,
            chunk_ids=[0]
        ))

        # Word: "world" (0.25-0.45s)
        segments.append(AudioSegment(
            type='preliminary',
            data=np.random.randn(int(0.2 * sample_rate)).astype(np.float32) * 0.1,
            start_time=0.25,
            end_time=0.45,
            chunk_ids=[1]
        ))

        # Word: "how" (0.5-0.65s)
        segments.append(AudioSegment(
            type='preliminary',
            data=np.random.randn(int(0.15 * sample_rate)).astype(np.float32) * 0.1,
            start_time=0.5,
            end_time=0.65,
            chunk_ids=[2]
        ))

        # Word: "are" (0.7-0.85s)
        segments.append(AudioSegment(
            type='preliminary',
            data=np.random.randn(int(0.15 * sample_rate)).astype(np.float32) * 0.1,
            start_time=0.7,
            end_time=0.85,
            chunk_ids=[3]
        ))

        # Word: "you" (0.9-1.05s)
        segments.append(AudioSegment(
            type='preliminary',
            data=np.random.randn(int(0.15 * sample_rate)).astype(np.float32) * 0.1,
            start_time=0.9,
            end_time=1.05,
            chunk_ids=[4]
        ))

        return segments


    def test_aggregates_words_into_windows(self, config, word_segments):
        """Windower should aggregate word segments into recognition windows."""
        chunk_queue = queue.Queue()
        windower = AdaptiveWindower(chunk_queue=chunk_queue, config=config)

        # Feed word segments
        for segment in word_segments:
            windower.process_segment(segment)

        # Flush to emit window
        windower.flush()

        # Should create at least one window
        assert not chunk_queue.empty()

        window = chunk_queue.get()
        assert isinstance(window, AudioSegment)
        assert window.type == 'finalized'
        assert isinstance(window.data, np.ndarray)
        assert len(window.chunk_ids) > 1  # Should aggregate multiple chunks


    def test_creates_3_second_windows(self, config, word_segments):
        """Windows should be approximately 3 seconds duration."""
        chunk_queue = queue.Queue()
        windower = AdaptiveWindower(chunk_queue=chunk_queue, config=config)

        # Create longer speech sequence (5 seconds of words)
        long_segments = []
        for i in range(25):  # 25 words across 5 seconds
            start = i * 0.2
            end = start + 0.15
            long_segments.append(AudioSegment(
                type='preliminary',
                data=np.random.randn(int(0.15 * 16000)).astype(np.float32) * 0.1,
                start_time=start,
                end_time=end,
                chunk_ids=[i]
            ))

        for segment in long_segments:
            windower.process_segment(segment)

        # Flush to get final window
        windower.flush()

        # Collect windows
        windows = []
        while not chunk_queue.empty():
            windows.append(chunk_queue.get())

        # Should have created multiple windows
        assert len(windows) >= 2

        # Each window should be close to 3 seconds
        for window in windows:
            duration = window.end_time - window.start_time
            # Allow some variance at boundaries
            assert 2.0 <= duration <= 3.5


    def test_creates_overlapping_windows(self, config, word_segments):
        """Windows should overlap by step_size (1 second)."""
        chunk_queue = queue.Queue()
        windower = AdaptiveWindower(chunk_queue=chunk_queue, config=config)

        # Create 6 seconds of speech
        long_segments = []
        for i in range(30):
            start = i * 0.2
            end = start + 0.15
            long_segments.append(AudioSegment(
                type='preliminary',
                data=np.random.randn(int(0.15 * 16000)).astype(np.float32) * 0.1,
                start_time=start,
                end_time=end,
                chunk_ids=[i]
            ))

        for segment in long_segments:
            windower.process_segment(segment)

        windower.flush()

        windows = []
        while not chunk_queue.empty():
            windows.append(chunk_queue.get())

        # Should have multiple overlapping windows
        assert len(windows) >= 3

        # Check overlap: each window should start ~1 second after previous
        for i in range(len(windows) - 1):
            time_diff = windows[i + 1].start_time - windows[i].start_time
            # Should be approximately step_size (1.0s)
            assert 0.8 <= time_diff <= 1.5


    def test_preserves_chunk_ids(self, config, word_segments):
        """Windows should aggregate chunk_ids from all preliminary segments."""
        chunk_queue = queue.Queue()
        windower = AdaptiveWindower(chunk_queue=chunk_queue, config=config)

        for segment in word_segments:
            windower.process_segment(segment)

        windower.flush()

        window = chunk_queue.get()

        # Window should have aggregated chunk_ids
        assert isinstance(window, AudioSegment)
        assert window.type == 'finalized'
        assert len(window.chunk_ids) == len(word_segments)  # All 5 words
        # Chunk IDs should be sorted and unique
        assert window.chunk_ids == sorted(set(window.chunk_ids))


    def test_handles_single_word(self, config):
        """Windower should handle single word segments."""
        chunk_queue = queue.Queue()
        windower = AdaptiveWindower(chunk_queue=chunk_queue, config=config)

        segment = AudioSegment(
            type='preliminary',
            data=np.random.randn(3200).astype(np.float32) * 0.1,
            start_time=0.0,
            end_time=0.2,
            chunk_ids=[0]
        )

        windower.process_segment(segment)
        windower.flush()

        # Should emit window with single word
        assert not chunk_queue.empty()
        window = chunk_queue.get()
        assert isinstance(window, AudioSegment)
        assert window.type == 'finalized'
        assert len(window.data) > 0
        assert window.chunk_ids == [0]


    def test_flush_emits_partial_window(self, config, word_segments):
        """Flush should emit window even if not full 3 seconds."""
        chunk_queue = queue.Queue()
        windower = AdaptiveWindower(chunk_queue=chunk_queue, config=config)

        # Feed only 2 words (~0.5 seconds)
        for segment in word_segments[:2]:
            windower.process_segment(segment)

        # No window yet (not 3 seconds)
        # Note: This might emit window depending on implementation

        # Flush should emit partial window
        windower.flush()

        assert not chunk_queue.empty()
        window = chunk_queue.get()
        assert isinstance(window, AudioSegment)
        assert window.type == 'finalized'
        duration = window.end_time - window.start_time
        assert duration < 1.0  # Less than full window
        assert window.chunk_ids == [0, 1]  # Two words


    def test_flush_idempotent_with_empty_buffer(self, config):
        """flush() should be safe to call multiple times with empty buffer."""
        chunk_queue = queue.Queue()
        windower = AdaptiveWindower(chunk_queue, config)

        # Call flush multiple times on empty windower
        windower.flush()
        windower.flush()
        windower.flush()

        # Should not crash or emit anything
        assert chunk_queue.empty()
