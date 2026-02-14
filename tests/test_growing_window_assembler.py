# tests/test_growing_window_assembler.py
import pytest
import numpy as np
import queue
from src.sound.GrowingWindowAssembler import GrowingWindowAssembler
from src.types import AudioSegment


class TestGrowingWindowAssembler:
    """Tests for GrowingWindowAssembler with growing window approach.

    GrowingWindowAssembler receives speech segments from SoundPreProcessor, and assembles them into
    growing windows that are recognized as a whole. Window emission rules:
    - Each new segment extends the current window
    - Windows are emitted as type='incremental' for each growth step
    - When total duration exceeds max_window_duration, oldest segments are dropped (sliding)
    - Flush emits remaining segments as final type='incremental'
    """

    @pytest.fixture
    def config(self):
        """Standard configuration for GrowingWindowAssembler."""
        return {
            'audio': {
                'sample_rate': 16000
            },
            'windowing': {
                'max_window_duration': 7.0
            }
        }

    @pytest.fixture
    def sample_rate(self):
        return 16000

    def _create_segment(self, start_time: float, duration_sec: float, chunk_id: int,
                        sample_rate: int = 16000,
                        left_context: np.ndarray = None,
                        right_context: np.ndarray = None) -> AudioSegment:
        """Helper to create incremental AudioSegment."""
        samples = int(duration_sec * sample_rate)
        data = np.random.randn(samples).astype(np.float32) * 0.1
        return AudioSegment(
            type='incremental',
            data=data,
            left_context=left_context if left_context is not None else np.array([], dtype=np.float32),
            right_context=right_context if right_context is not None else np.array([], dtype=np.float32),
            start_time=start_time,
            end_time=start_time + duration_sec,
            chunk_ids=[chunk_id]
        )

    def test_window_grows_progressively(self, config, sample_rate):
        """Each new segment should extend the window cumulatively."""
        speech_queue = queue.Queue()
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        segments = [
            self._create_segment(i * 1.0, 1.0, chunk_id=i, sample_rate=sample_rate)
            for i in range(4)
        ]

        expected_durations = [1.0, 2.0, 3.0, 4.0]

        for seg, expected_dur in zip(segments, expected_durations):
            assembler.process_segment(seg)
            window = speech_queue.get()
            assert window.type == 'incremental'
            actual_dur = len(window.data) / sample_rate
            assert abs(actual_dur - expected_dur) < 0.01

    def test_window_slides_past_max_duration(self, config, sample_rate):
        """When duration exceeds max_window_duration, oldest segments are dropped."""
        speech_queue = queue.Queue()
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        # Create 10 segments of 1s each = 10s total (exceeds 7s max)
        segments = [
            self._create_segment(i * 1.0, 1.0, chunk_id=i, sample_rate=sample_rate)
            for i in range(10)
        ]

        for seg in segments:
            assembler.process_segment(seg)
            window = speech_queue.get()

            # After segment 7+, window duration should not exceed max_window_duration
            actual_dur = len(window.data) / sample_rate
            assert actual_dur <= config['windowing']['max_window_duration'] + 0.01

    def test_sliding_drops_correct_segments(self, config, sample_rate):
        """Verify that oldest segments are dropped correctly during sliding."""
        speech_queue = queue.Queue()
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        # Create 9 segments of 1s each
        segments = [
            self._create_segment(i * 1.0, 1.0, chunk_id=i, sample_rate=sample_rate)
            for i in range(9)
        ]

        for seg in segments:
            assembler.process_segment(seg)
            window = speech_queue.get()

        # After 9 segments, window should contain last 7 segments (ids 2-8)
        # because first 2 segments should have been dropped
        last_window = window
        expected_data = np.concatenate([seg.data for seg in segments[2:9]])
        assert len(last_window.data) == len(expected_data)
        assert np.array_equal(last_window.data, expected_data)

    def test_flush_emits_remaining_segments(self, config, sample_rate):
        """Flush should emit remaining segments as final type='incremental'."""
        speech_queue = queue.Queue()
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        seg1 = self._create_segment(0.0, 1.0, chunk_id=0, sample_rate=sample_rate)
        seg2 = self._create_segment(1.0, 1.0, chunk_id=1, sample_rate=sample_rate)

        assembler.process_segment(seg1)
        speech_queue.get()  # consume incremental

        assembler.process_segment(seg2)
        speech_queue.get()  # consume incremental

        assembler.flush()

        assert not speech_queue.empty()
        flush_window = speech_queue.get()
        assert flush_window.type == 'incremental'
        assert flush_window.chunk_ids == [0, 1]

    def test_flush_resets_state(self, config, sample_rate):
        """After flush, state should be reset for next speech sequence."""
        speech_queue = queue.Queue()
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        # First sequence
        seg1 = self._create_segment(0.0, 1.0, chunk_id=0, sample_rate=sample_rate)
        assembler.process_segment(seg1)
        speech_queue.get()

        assembler.flush()
        speech_queue.get()

        # Second sequence should start fresh
        seg2 = self._create_segment(5.0, 1.0, chunk_id=1, sample_rate=sample_rate)
        assembler.process_segment(seg2)

        window = speech_queue.get()
        assert window.type == 'incremental'
        assert window.chunk_ids == [1]
        assert len(window.data) == sample_rate  # Only 1 second

    def test_flush_idempotent_with_empty_buffer(self, config):
        """flush() should be safe to call multiple times with empty buffer."""
        speech_queue = queue.Queue()
        assembler = GrowingWindowAssembler(speech_queue, config)

        assembler.flush()
        assembler.flush()
        assembler.flush()

        assert speech_queue.empty()

class TestGrowingWindowAssemblerContext:
    """Tests for left/right context handling in GrowingWindowAssembler.

    Context logic:
    - Windows include left_context until original first segment drops
    - Flush window gets right_context from last segment
    """

    @pytest.fixture
    def config(self):
        """Standard configuration for GrowingWindowAssembler."""
        return {
            'audio': {
                'sample_rate': 16000
            },
            'windowing': {
                'max_window_duration': 7.0
            }
        }

    @pytest.fixture
    def sample_rate(self):
        return 16000

    def _create_segment(self, start_time: float, duration_sec: float, chunk_id: int,
                        sample_rate: int = 16000,
                        left_context: np.ndarray = None,
                        right_context: np.ndarray = None) -> AudioSegment:
        """Helper to create incremental AudioSegment with context."""
        samples = int(duration_sec * sample_rate)
        data = np.random.randn(samples).astype(np.float32) * 0.1
        return AudioSegment(
            type='incremental',
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
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        left_context = np.random.randn(int(0.5 * sample_rate)).astype(np.float32) * 0.05
        segment = self._create_segment(0.0, 1.0, 0, sample_rate, left_context=left_context)

        assembler.process_segment(segment)

        window = speech_queue.get()
        assert window.left_context.size > 0
        assert np.array_equal(window.left_context, left_context)

    def test_windows_include_left_context_until_sliding(self, config, sample_rate):
        """All windows should include left_context while no sliding occurs."""
        speech_queue = queue.Queue()
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        left_context = np.random.randn(int(0.5 * sample_rate)).astype(np.float32) * 0.05

        segments = [
            self._create_segment(0.0, 1.0, 0, sample_rate, left_context=left_context),
            self._create_segment(1.0, 1.0, 1, sample_rate),
            self._create_segment(2.0, 1.0, 2, sample_rate),
        ]

        for seg in segments:
            assembler.process_segment(seg)

        windows = []
        while not speech_queue.empty():
            windows.append(speech_queue.get())

        assert len(windows) == 3

        # All windows should have left_context (no sliding yet)
        for window in windows:
            assert window.left_context.size > 0

    def test_incremental_windows_no_right_context(self, config, sample_rate):
        """Incremental windows should have empty right_context."""
        speech_queue = queue.Queue()
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        segments = [
            self._create_segment(i * 1.0, 1.0, i, sample_rate)
            for i in range(3)
        ]

        for seg in segments:
            assembler.process_segment(seg)

        while not speech_queue.empty():
            window = speech_queue.get()
            assert window.type == 'incremental'
            assert window.right_context.size == 0

    def test_flush_includes_right_context(self, config, sample_rate):
        """Flush window should include right_context from last segment."""
        speech_queue = queue.Queue()
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        right_context = np.random.randn(int(0.2 * sample_rate)).astype(np.float32) * 0.05

        seg1 = self._create_segment(0.0, 0.5, 0, sample_rate)
        assembler.process_segment(seg1)
        speech_queue.get()  # consume incremental

        seg2 = self._create_segment(0.5, 0.5, 1, sample_rate, right_context=right_context)
        assembler.flush(seg2)

        flush_window = speech_queue.get()
        assert flush_window.type == 'incremental'
        assert flush_window.right_context.size > 0
        assert np.array_equal(flush_window.right_context, right_context)

    def test_flush_resets_context_state(self, config, sample_rate):
        """After flush, context state should be reset for next speech sequence."""
        speech_queue = queue.Queue()
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        # First sequence with contexts
        left_ctx_1 = np.ones(1000, dtype=np.float32) * 0.1
        right_ctx_1 = np.ones(500, dtype=np.float32) * 0.2
        seg1 = self._create_segment(0.0, 1.0, 0, sample_rate,
                                    left_context=left_ctx_1, right_context=right_ctx_1)
        assembler.flush(seg1)

        flush1 = speech_queue.get()
        assert np.array_equal(flush1.left_context, left_ctx_1)
        assert np.array_equal(flush1.right_context, right_ctx_1)

        # Second sequence - should use NEW contexts
        left_ctx_2 = np.ones(800, dtype=np.float32) * 0.3
        right_ctx_2 = np.ones(400, dtype=np.float32) * 0.4
        seg2 = self._create_segment(5.0, 1.0, 1, sample_rate,
                                    left_context=left_ctx_2, right_context=right_ctx_2)
        assembler.flush(seg2)

        flush2 = speech_queue.get()
        assert np.array_equal(flush2.left_context, left_ctx_2)
        assert np.array_equal(flush2.right_context, right_ctx_2)

    def test_context_preserved_after_sliding(self, config, sample_rate):
        """After window slides, left_context should be absent from new windows."""
        speech_queue = queue.Queue()
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        left_context = np.random.randn(int(0.5 * sample_rate)).astype(np.float32) * 0.05

        # Create 9 segments to force sliding (max_window_duration is 7s)
        segments = [
            self._create_segment(i * 1.0, 1.0, i, sample_rate,
                                left_context=left_context if i == 0 else None)
            for i in range(9)
        ]

        for seg in segments:
            assembler.process_segment(seg)

        # Collect all windows
        windows = []
        while not speech_queue.empty():
            windows.append(speech_queue.get())

        # Windows before sliding should include left_context (max window is 7s)
        for window in windows[:7]:
            assert window.left_context.size > 0

        # Windows after sliding should have empty left_context
        for window in windows[7:]:
            assert window.left_context.size == 0


class TestGrowingWindowAssemblerQueueFull:
    """Tests for queue full handling in GrowingWindowAssembler."""

    @pytest.fixture
    def config(self):
        return {
            'audio': {
                'sample_rate': 16000
            },
            'windowing': {
                'max_window_duration': 7.0
            }
        }

    @pytest.fixture
    def sample_rate(self):
        return 16000

    def _create_segment(self, start_time: float, duration_sec: float, chunk_id: int,
                        sample_rate: int = 16000) -> AudioSegment:
        """Helper to create incremental AudioSegment."""
        samples = int(duration_sec * sample_rate)
        data = np.random.randn(samples).astype(np.float32) * 0.1
        return AudioSegment(
            type='incremental',
            data=data,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=start_time,
            end_time=start_time + duration_sec,
            chunk_ids=[chunk_id]
        )

    def test_queue_full_drops_window(self, config, sample_rate):
        """When speech_queue is full, window should be dropped silently."""
        speech_queue = queue.Queue(maxsize=1)
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config)

        seg1 = self._create_segment(0.0, 1.0, 0, sample_rate)
        seg2 = self._create_segment(1.0, 1.0, 1, sample_rate)

        assembler.process_segment(seg1)
        assert speech_queue.full()  # Queue has 1 item

        # This should attempt to put but drop because queue is full
        assembler.process_segment(seg2)

        # Queue should still have only 1 item (the first window)
        assert speech_queue.qsize() == 1
        window = speech_queue.get()
        assert window.chunk_ids == [0]

    def test_dropped_windows_counter(self, config, sample_rate):
        """Dropped windows should be counted."""
        speech_queue = queue.Queue(maxsize=1)
        assembler = GrowingWindowAssembler(speech_queue=speech_queue, config=config, verbose=False)

        segments = [
            self._create_segment(i * 1.0, 1.0, i, sample_rate)
            for i in range(5)
        ]

        for seg in segments:
            assembler.process_segment(seg)

        # First window should be in queue, subsequent 4 should be dropped
        assert assembler.dropped_windows == 4
