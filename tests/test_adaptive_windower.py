# tests/test_adaptive_windower.py
import pytest
import numpy as np
import queue
from src.AdaptiveWindower import AdaptiveWindower
from src.types import AudioSegment


class TestAdaptiveWindower:
    """Tests for AdaptiveWindower with segment-based windowing.

    AdaptiveWindower receives speech segments from VAD and aggregates them into
    recognition windows. Window emission rules:
    - Requires 2+ segments (for overlap)
    - Total audio duration >= window_duration
    - 1-segment overlap between consecutive windows
    """

    @pytest.fixture
    def config(self):
        """Standard configuration for AdaptiveWindower."""
        return {
            'audio': {
                'sample_rate': 16000
            },
            'windowing': {
                'window_duration': 3.0  # 3 seconds minimum
            }
        }

    @pytest.fixture
    def sample_rate(self):
        return 16000

    def _create_segment(self, start_time: float, duration_sec: float, chunk_id: int,
                        sample_rate: int = 16000,
                        left_context: np.ndarray = None,
                        right_context: np.ndarray = None) -> AudioSegment:
        """Helper to create preliminary AudioSegment."""
        samples = int(duration_sec * sample_rate)
        data = np.random.randn(samples).astype(np.float32) * 0.1
        return AudioSegment(
            type='preliminary',
            data=data,
            left_context=left_context if left_context is not None else np.array([], dtype=np.float32),
            right_context=right_context if right_context is not None else np.array([], dtype=np.float32),
            start_time=start_time,
            end_time=start_time + duration_sec,
            chunk_ids=[chunk_id]
        )

    def test_single_segment_no_window(self, config, sample_rate):
        """Single segment should NOT emit window (need 2+ for overlap)."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        # One 4-second segment (exceeds window_duration but only 1 segment)
        segment = self._create_segment(0.0, 4.0, chunk_id=0, sample_rate=sample_rate)
        windower.process_segment(segment)

        assert speech_queue.empty()

    def test_two_segments_below_duration_no_window(self, config, sample_rate):
        """Two segments with total duration < window_duration should NOT emit window."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        # Two 1-second segments = 2 seconds total (< window_duration)
        seg1 = self._create_segment(0.0, 1.0, chunk_id=0, sample_rate=sample_rate)
        seg2 = self._create_segment(1.0, 1.0, chunk_id=1, sample_rate=sample_rate)

        windower.process_segment(seg1)
        windower.process_segment(seg2)

        assert speech_queue.empty()

    def test_two_segments_meeting_duration_emits_window(self, config, sample_rate):
        """Two segments with total duration >= window_duration should emit window."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        # Two 1.5-second segments = window_duration (3 seconds)
        seg1 = self._create_segment(0.0, 1.5, chunk_id=0, sample_rate=sample_rate)
        seg2 = self._create_segment(1.5, 1.5, chunk_id=1, sample_rate=sample_rate)

        windower.process_segment(seg1)
        windower.process_segment(seg2)

        # Should emit window
        assert not speech_queue.empty()
        window = speech_queue.get()
        assert window.type == 'finalized'
        assert 0 in window.chunk_ids
        assert 1 in window.chunk_ids

    def test_three_segments_trigger_window(self, config, sample_rate):
        """Third segment pushing total >= window_duration should emit window."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        # Three 1-second segments = 3 seconds total
        seg1 = self._create_segment(0.0, 1.0, chunk_id=0, sample_rate=sample_rate)
        seg2 = self._create_segment(1.0, 1.0, chunk_id=1, sample_rate=sample_rate)
        seg3 = self._create_segment(2.0, 1.0, chunk_id=2, sample_rate=sample_rate)

        windower.process_segment(seg1)
        assert speech_queue.empty()

        windower.process_segment(seg2)
        assert speech_queue.empty()

        windower.process_segment(seg3)
        assert not speech_queue.empty()

        window = speech_queue.get()
        assert window.type == 'finalized'
        assert len(window.chunk_ids) == 3

    def test_one_segment_overlap(self, config, sample_rate):
        """After window emission, last segment stays in buffer for overlap."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        # Create 6 seconds of speech in 1.5s segments
        segments = [
            self._create_segment(i * 1.5, 1.5, chunk_id=i, sample_rate=sample_rate)
            for i in range(4)
        ]

        for seg in segments:
            windower.process_segment(seg)

        windows = []
        while not speech_queue.empty():
            windows.append(speech_queue.get())

        assert len(windows) == 3

        # Check overlap: consecutive windows should share the last segment of previous
        # Last chunk_id of window 1 should be first chunk_id of window 2
        last_of_first = windows[0].chunk_ids[-1]
        first_of_second = windows[1].chunk_ids[0]
        assert last_of_first == first_of_second, \
            f"Expected overlap: last chunk {last_of_first} should equal first chunk {first_of_second}"

    def test_flush_emits_single_segment(self, config, sample_rate):
        """Flush should emit even a single segment."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        segment = self._create_segment(0.0, 0.5, chunk_id=0, sample_rate=sample_rate)
        windower.process_segment(segment)

        assert speech_queue.empty()  # No window yet

        windower.flush()

        assert not speech_queue.empty()
        window = speech_queue.get()
        assert window.type == 'flush'
        assert window.chunk_ids == [0]

    def test_flush_emits_partial_duration(self, config, sample_rate):
        """Flush should emit even if duration < window_duration."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        # Two short segments, 1 second total
        seg1 = self._create_segment(0.0, 0.5, chunk_id=0, sample_rate=sample_rate)
        seg2 = self._create_segment(0.5, 0.5, chunk_id=1, sample_rate=sample_rate)

        windower.process_segment(seg1)
        windower.process_segment(seg2)

        assert speech_queue.empty()  # 2 segments but < 3 seconds

        windower.flush()

        window = speech_queue.get()
        assert window.type == 'flush'
        assert window.chunk_ids == [0, 1]
        assert len(window.data) == sample_rate

    def test_flush_idempotent_with_empty_buffer(self, config):
        """flush() should be safe to call multiple times with empty buffer."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue, config)

        windower.flush()
        windower.flush()
        windower.flush()

        assert speech_queue.empty()


class TestAdaptiveWindowerContext:
    """Tests for left/right context handling in AdaptiveWindower.

    Context logic:
    - First window gets left_context from first segment
    - Subsequent windows get empty left_context (avoid duplication)
    - Flush window gets right_context from last segment
    """

    @pytest.fixture
    def config(self):
        """Standard configuration for AdaptiveWindower."""
        return {
            'audio': {
                'sample_rate': 16000
            },
            'windowing': {
                'window_duration': 3.0
            }
        }

    @pytest.fixture
    def sample_rate(self):
        return 16000

    def _create_segment(self, start_time: float, duration_sec: float, chunk_id: int,
                        sample_rate: int = 16000,
                        left_context: np.ndarray = None,
                        right_context: np.ndarray = None) -> AudioSegment:
        """Helper to create preliminary AudioSegment with context."""
        samples = int(duration_sec * sample_rate)
        data = np.random.randn(samples).astype(np.float32) * 0.1
        return AudioSegment(
            type='preliminary',
            data=data,
            left_context=left_context if left_context is not None else np.array([], dtype=np.float32),
            right_context=right_context if right_context is not None else np.array([], dtype=np.float32),
            start_time=start_time,
            end_time=start_time + duration_sec,
            chunk_ids=[chunk_id]
        )

    def test_first_window_gets_left_context(self, config, sample_rate):
        """First window should have left_context from first segment."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        left_context = np.random.randn(int(0.5 * sample_rate)).astype(np.float32) * 0.05

        # 3 segments x 1s = 3 seconds
        segments = [
            self._create_segment(0.0, 1.0, 0, sample_rate, left_context=left_context),
            self._create_segment(1.0, 1.0, 1, sample_rate),
            self._create_segment(2.0, 1.0, 2, sample_rate),
        ]

        for seg in segments:
            windower.process_segment(seg)

        window = speech_queue.get()
        assert window.left_context.size > 0
        assert np.array_equal(window.left_context, left_context)

    def test_subsequent_windows_empty_left_context(self, config, sample_rate):
        """Windows after the first should have empty left_context."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        left_context = np.random.randn(int(0.5 * sample_rate)).astype(np.float32) * 0.05

        # 6 segments x 1s = 6 seconds (should produce 2+ windows)
        segments = [
            self._create_segment(i * 1.0, 1.0, i, sample_rate,
                                left_context=left_context if i == 0 else None)
            for i in range(6)
        ]

        for seg in segments:
            windower.process_segment(seg)

        windower.flush()

        windows = []
        while not speech_queue.empty():
            windows.append(speech_queue.get())

        assert len(windows) >= 2

        # First window has left_context
        assert windows[0].left_context.size > 0

        # Subsequent windows have empty left_context
        for window in windows[1:]:
            assert window.left_context.size == 0

    def test_flush_includes_right_context(self, config, sample_rate):
        """Flush window should include right_context from last segment."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        right_context = np.random.randn(int(0.2 * sample_rate)).astype(np.float32) * 0.05

        segments = [
            self._create_segment(0.0, 0.5, 0, sample_rate),
            self._create_segment(0.5, 0.5, 1, sample_rate, right_context=right_context),
        ]

        for seg in segments:
            windower.process_segment(seg)

        windower.flush()

        window = speech_queue.get()
        assert window.type == 'flush'
        assert window.right_context.size > 0
        assert np.array_equal(window.right_context, right_context)

    def test_flush_with_segment_processes_and_flushes(self, config, sample_rate):
        """flush(segment) should append segment then flush."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        left_context = np.random.randn(int(0.3 * sample_rate)).astype(np.float32) * 0.05
        right_context = np.random.randn(int(0.2 * sample_rate)).astype(np.float32) * 0.05

        seg1 = self._create_segment(0.0, 0.5, 0, sample_rate, left_context=left_context)
        windower.process_segment(seg1)

        seg2 = self._create_segment(0.5, 0.5, 1, sample_rate, right_context=right_context)
        windower.flush(seg2)

        window = speech_queue.get()
        assert window.type == 'flush'
        assert np.array_equal(window.left_context, left_context)
        assert np.array_equal(window.right_context, right_context)
        assert 0 in window.chunk_ids
        assert 1 in window.chunk_ids

    def test_flush_resets_context_state(self, config, sample_rate):
        """After flush, context state should be reset for next speech sequence."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        # First sequence
        left_ctx_1 = np.ones(1000, dtype=np.float32) * 0.1
        right_ctx_1 = np.ones(500, dtype=np.float32) * 0.2
        seg1 = self._create_segment(0.0, 1.0, 0, sample_rate,
                                    left_context=left_ctx_1, right_context=right_ctx_1)
        windower.flush(seg1)

        window1 = speech_queue.get()
        assert np.array_equal(window1.left_context, left_ctx_1)
        assert np.array_equal(window1.right_context, right_ctx_1)

        # Second sequence - should use NEW contexts
        left_ctx_2 = np.ones(800, dtype=np.float32) * 0.3
        right_ctx_2 = np.ones(400, dtype=np.float32) * 0.4
        seg2 = self._create_segment(5.0, 1.0, 1, sample_rate,
                                    left_context=left_ctx_2, right_context=right_ctx_2)
        windower.flush(seg2)

        window2 = speech_queue.get()
        assert np.array_equal(window2.left_context, left_ctx_2)
        assert np.array_equal(window2.right_context, right_ctx_2)

    def test_context_preserved_for_single_segment(self, config, sample_rate):
        """Single segment flush should preserve both contexts."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        left_context = np.random.randn(int(0.5 * sample_rate)).astype(np.float32) * 0.05
        right_context = np.random.randn(int(0.2 * sample_rate)).astype(np.float32) * 0.05

        segment = self._create_segment(0.0, 0.5, 0, sample_rate,
                                       left_context=left_context,
                                       right_context=right_context)
        windower.flush(segment)

        window = speech_queue.get()
        assert window.type == 'flush'
        assert np.array_equal(window.left_context, left_context)
        assert np.array_equal(window.right_context, right_context)
