# tests/test_adaptive_windower.py
import pytest
import numpy as np
import queue
from src.AdaptiveWindower import AdaptiveWindower


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
        """Generate simulated word-level segments from VAD.

        Creates segments representing: "Hello world how are you"
        Each word is ~200ms with natural pauses between.
        """
        sample_rate = 16000
        segments = []

        # Word: "Hello" (0.0-0.2s)
        segments.append({
            'is_speech': True,
            'start_time': 0.0,
            'end_time': 0.2,
            'duration': 0.2,
            'data': np.random.randn(int(0.2 * sample_rate)).astype(np.float32) * 0.1,
            'timestamp': 0.0
        })

        # Word: "world" (0.25-0.45s)
        segments.append({
            'is_speech': True,
            'start_time': 0.25,
            'end_time': 0.45,
            'duration': 0.2,
            'data': np.random.randn(int(0.2 * sample_rate)).astype(np.float32) * 0.1,
            'timestamp': 0.25
        })

        # Word: "how" (0.5-0.65s)
        segments.append({
            'is_speech': True,
            'start_time': 0.5,
            'end_time': 0.65,
            'duration': 0.15,
            'data': np.random.randn(int(0.15 * sample_rate)).astype(np.float32) * 0.1,
            'timestamp': 0.5
        })

        # Word: "are" (0.7-0.85s)
        segments.append({
            'is_speech': True,
            'start_time': 0.7,
            'end_time': 0.85,
            'duration': 0.15,
            'data': np.random.randn(int(0.15 * sample_rate)).astype(np.float32) * 0.1,
            'timestamp': 0.7
        })

        # Word: "you" (0.9-1.05s)
        segments.append({
            'is_speech': True,
            'start_time': 0.9,
            'end_time': 1.05,
            'duration': 0.15,
            'data': np.random.randn(int(0.15 * sample_rate)).astype(np.float32) * 0.1,
            'timestamp': 0.9
        })

        return segments


    def test_aggregates_words_into_windows(self, config, word_segments):
        """Windower should aggregate word segments into recognition windows."""
        window_queue = queue.Queue()
        windower = AdaptiveWindower(window_queue=window_queue, config=config)

        # Feed word segments
        for segment in word_segments:
            windower.process_segment(segment)

        # Flush to emit window
        windower.flush()

        # Should create at least one window
        assert not window_queue.empty()

        window = window_queue.get()
        assert 'data' in window
        assert 'start_time' in window
        assert 'end_time' in window
        assert isinstance(window['data'], np.ndarray)


    def test_creates_3_second_windows(self, config, word_segments):
        """Windows should be approximately 3 seconds duration."""
        window_queue = queue.Queue()
        windower = AdaptiveWindower(window_queue=window_queue, config=config)

        # Create longer speech sequence (5 seconds of words)
        long_segments = []
        for i in range(25):  # 25 words across 5 seconds
            start = i * 0.2
            end = start + 0.15
            long_segments.append({
                'is_speech': True,
                'start_time': start,
                'end_time': end,
                'duration': 0.15,
                'data': np.random.randn(int(0.15 * 16000)).astype(np.float32) * 0.1,
                'timestamp': start
            })

        for segment in long_segments:
            windower.process_segment(segment)

        # Flush to get final window
        windower.flush()

        # Collect windows
        windows = []
        while not window_queue.empty():
            windows.append(window_queue.get())

        # Should have created multiple windows
        assert len(windows) >= 2

        # Each window should be close to 3 seconds
        for window in windows:
            duration = window['end_time'] - window['start_time']
            # Allow some variance at boundaries
            assert 2.0 <= duration <= 3.5


    def test_creates_overlapping_windows(self, config, word_segments):
        """Windows should overlap by step_size (1 second)."""
        window_queue = queue.Queue()
        windower = AdaptiveWindower(window_queue=window_queue, config=config)

        # Create 6 seconds of speech
        long_segments = []
        for i in range(30):
            start = i * 0.2
            end = start + 0.15
            long_segments.append({
                'is_speech': True,
                'start_time': start,
                'end_time': end,
                'duration': 0.15,
                'data': np.random.randn(int(0.15 * 16000)).astype(np.float32) * 0.1,
                'timestamp': start
            })

        for segment in long_segments:
            windower.process_segment(segment)

        windower.flush()

        windows = []
        while not window_queue.empty():
            windows.append(window_queue.get())

        # Should have multiple overlapping windows
        assert len(windows) >= 3

        # Check overlap: each window should start ~1 second after previous
        for i in range(len(windows) - 1):
            time_diff = windows[i + 1]['start_time'] - windows[i]['start_time']
            # Should be approximately step_size (1.0s)
            assert 0.8 <= time_diff <= 1.5


    def test_preserves_word_timing_metadata(self, config, word_segments):
        """Windows should preserve word-level timing information."""
        window_queue = queue.Queue()
        windower = AdaptiveWindower(window_queue=window_queue, config=config)

        for segment in word_segments:
            windower.process_segment(segment)

        windower.flush()

        window = window_queue.get()

        # Window should have segment metadata
        assert 'segments' in window
        assert len(window['segments']) > 0

        # Each segment should have timing info
        for seg in window['segments']:
            assert 'start_time' in seg
            assert 'end_time' in seg
            assert 'duration' in seg


    def test_handles_single_word(self, config):
        """Windower should handle single word segments."""
        window_queue = queue.Queue()
        windower = AdaptiveWindower(window_queue=window_queue, config=config)

        segment = {
            'is_speech': True,
            'start_time': 0.0,
            'end_time': 0.2,
            'duration': 0.2,
            'data': np.random.randn(3200).astype(np.float32) * 0.1,
            'timestamp': 0.0
        }

        windower.process_segment(segment)
        windower.flush()

        # Should emit window with single word
        assert not window_queue.empty()
        window = window_queue.get()
        assert len(window['data']) > 0


    def test_flush_emits_partial_window(self, config, word_segments):
        """Flush should emit window even if not full 3 seconds."""
        window_queue = queue.Queue()
        windower = AdaptiveWindower(window_queue=window_queue, config=config)

        # Feed only 2 words (~0.5 seconds)
        for segment in word_segments[:2]:
            windower.process_segment(segment)

        # No window yet (not 3 seconds)
        # Note: This might emit window depending on implementation

        # Flush should emit partial window
        windower.flush()

        assert not window_queue.empty()
        window = window_queue.get()
        duration = window['end_time'] - window['start_time']
        assert duration < 1.0  # Less than full window
