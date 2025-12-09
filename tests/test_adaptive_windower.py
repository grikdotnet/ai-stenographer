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
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=0.2,
            chunk_ids=[0]
        ))

        # Word: "world" (0.25-0.45s)
        segments.append(AudioSegment(
            type='preliminary',
            data=np.random.randn(int(0.2 * sample_rate)).astype(np.float32) * 0.1,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.25,
            end_time=0.45,
            chunk_ids=[1]
        ))

        # Word: "how" (0.5-0.65s)
        segments.append(AudioSegment(
            type='preliminary',
            data=np.random.randn(int(0.15 * sample_rate)).astype(np.float32) * 0.1,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.5,
            end_time=0.65,
            chunk_ids=[2]
        ))

        # Word: "are" (0.7-0.85s)
        segments.append(AudioSegment(
            type='preliminary',
            data=np.random.randn(int(0.15 * sample_rate)).astype(np.float32) * 0.1,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.7,
            end_time=0.85,
            chunk_ids=[3]
        ))

        # Word: "you" (0.9-1.05s)
        segments.append(AudioSegment(
            type='preliminary',
            data=np.random.randn(int(0.15 * sample_rate)).astype(np.float32) * 0.1,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.9,
            end_time=1.05,
            chunk_ids=[4]
        ))

        return segments


    def test_aggregates_words_into_windows(self, config, word_segments):
        """Windower should aggregate word segments into recognition windows."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        # Feed word segments
        for segment in word_segments:
            windower.process_segment(segment)

        # Flush to emit window
        windower.flush()

        # Should create at least one window (flush type since we called flush())
        assert not speech_queue.empty()

        window = speech_queue.get()
        assert isinstance(window, AudioSegment)
        assert window.type == 'flush'
        assert isinstance(window.data, np.ndarray)
        assert len(window.chunk_ids) > 1  # Should aggregate multiple chunks


    def test_creates_3_second_windows(self, config, word_segments):
        """Windows should be approximately 3 seconds duration."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        # Create longer speech sequence (5 seconds of words)
        long_segments = []
        for i in range(25):  # 25 words across 5 seconds
            start = i * 0.2
            end = start + 0.15
            long_segments.append(AudioSegment(
                type='preliminary',
                data=np.random.randn(int(0.15 * 16000)).astype(np.float32) * 0.1,
                left_context=np.array([], dtype=np.float32),
                right_context=np.array([], dtype=np.float32),
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
        while not speech_queue.empty():
            windows.append(speech_queue.get())

        # Should have created multiple windows
        assert len(windows) >= 2

        # Each window should be close to 3 seconds
        for window in windows:
            duration = window.end_time - window.start_time
            # Allow some variance at boundaries
            assert 2.0 <= duration <= 3.5


    def test_creates_overlapping_windows(self, config, word_segments):
        """Windows should overlap by step_size (1 second)."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        # Create 6 seconds of speech
        long_segments = []
        for i in range(30):
            start = i * 0.2
            end = start + 0.15
            long_segments.append(AudioSegment(
                type='preliminary',
                data=np.random.randn(int(0.15 * 16000)).astype(np.float32) * 0.1,
                left_context=np.array([], dtype=np.float32),
                right_context=np.array([], dtype=np.float32),
                start_time=start,
                end_time=end,
                chunk_ids=[i]
            ))

        for segment in long_segments:
            windower.process_segment(segment)

        windower.flush()

        windows = []
        while not speech_queue.empty():
            windows.append(speech_queue.get())

        # Should have multiple overlapping windows
        assert len(windows) >= 3

        # Check overlap: each window should start ~1 second after previous
        for i in range(len(windows) - 1):
            time_diff = windows[i + 1].start_time - windows[i].start_time
            # Should be approximately step_size (1.0s)
            assert 0.8 <= time_diff <= 1.5


    def test_preserves_chunk_ids(self, config, word_segments):
        """Windows should aggregate chunk_ids from all preliminary segments."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        for segment in word_segments:
            windower.process_segment(segment)

        windower.flush()

        window = speech_queue.get()

        # Window should have aggregated chunk_ids
        assert isinstance(window, AudioSegment)
        assert window.type == 'flush'
        assert len(window.chunk_ids) == len(word_segments)  # All 5 words
        # Chunk IDs should be sorted and unique
        assert window.chunk_ids == sorted(set(window.chunk_ids))


    def test_handles_single_word(self, config):
        """Windower should handle single word segments."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        segment = AudioSegment(
            type='preliminary',
            data=np.random.randn(3200).astype(np.float32) * 0.1,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=0.2,
            chunk_ids=[0]
        )

        windower.process_segment(segment)
        windower.flush()

        # Should emit window with single word
        assert not speech_queue.empty()
        window = speech_queue.get()
        assert isinstance(window, AudioSegment)
        assert window.type == 'flush'
        assert len(window.data) > 0
        assert window.chunk_ids == [0]


    def test_flush_emits_partial_window(self, config, word_segments):
        """Flush should emit window even if not full 3 seconds."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        # Feed only 2 words (~0.5 seconds)
        for segment in word_segments[:2]:
            windower.process_segment(segment)

        # No window yet (not 3 seconds)
        # Note: This might emit window depending on implementation

        # Flush should emit partial window
        windower.flush()

        assert not speech_queue.empty()
        window = speech_queue.get()
        assert isinstance(window, AudioSegment)
        assert window.type == 'flush'  # UPDATE: was 'finalized'
        duration = window.end_time - window.start_time
        assert duration < 1.0  # Less than full window
        assert window.chunk_ids == [0, 1]  # Two words


    def test_flush_idempotent_with_empty_buffer(self, config):
        """flush() should be safe to call multiple times with empty buffer."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue, config)

        # Call flush multiple times on empty windower
        windower.flush()
        windower.flush()
        windower.flush()

        assert speech_queue.empty()


class TestAdaptiveWindowerContext:
    """Tests for left/right context handling in AdaptiveWindower.

    Context logic:
    - First finalized window gets left_context from first preliminary segment
    - Subsequent windows get empty left_context (duplicate data)
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
                'window_duration': 3.0,
                'step_size': 1.0
            }
        }

    @pytest.fixture
    def sample_rate(self):
        return 16000

    def _create_segment(self, start_time: float, end_time: float, chunk_id: int,
                        sample_rate: int = 16000,
                        left_context: np.ndarray = None,
                        right_context: np.ndarray = None) -> AudioSegment:
        """Helper to create preliminary AudioSegment with context."""
        duration = end_time - start_time
        data = np.random.randn(int(duration * sample_rate)).astype(np.float32) * 0.1
        return AudioSegment(
            type='preliminary',
            data=data,
            left_context=left_context if left_context is not None else np.array([], dtype=np.float32),
            right_context=right_context if right_context is not None else np.array([], dtype=np.float32),
            start_time=start_time,
            end_time=end_time,
            chunk_ids=[chunk_id]
        )

    def test_subsequent_windows_left_context(self, config, sample_rate):
        """Windows after the first should have empty left_context."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        left_context = np.random.randn(int(0.5 * sample_rate)).astype(np.float32) * 0.05

        # Create segments spanning 6 seconds to trigger multiple windows
        segments = [
            self._create_segment(0.0, 1.0, 0, sample_rate, left_context=left_context),
            self._create_segment(1.0, 2.0, 1, sample_rate, left_context=np.ones(100, dtype=np.float32)),
            self._create_segment(2.0, 3.0, 2, sample_rate, left_context=np.ones(100, dtype=np.float32)),
            self._create_segment(3.0, 4.0, 3, sample_rate, left_context=np.ones(100, dtype=np.float32)),
            self._create_segment(4.0, 5.0, 4, sample_rate),
            self._create_segment(5.0, 6.0, 5, sample_rate),
        ]

        for seg in segments:
            windower.process_segment(seg)

        windower.flush()

        # Collect all windows
        windows = []
        while not speech_queue.empty():
            windows.append(speech_queue.get())

        assert len(windows) >= 2, f"Expected at least 2 windows, got {len(windows)}"

        # First window should have left_context
        assert windows[0].left_context.size > 0
        assert np.array_equal(windows[0].left_context, left_context)

        # Subsequent windows should have empty left_context
        for i, window in enumerate(windows[1:], start=1):
            assert window.left_context.size == 0, f"Window {i} should have empty left_context"

    def test_flush_includes_right_context(self, config, sample_rate):
        """Flush window should include right_context from last segment."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        right_context = np.random.randn(int(0.2 * sample_rate)).astype(np.float32) * 0.05

        # Create segments (not enough to trigger window, so flush will emit)
        segments = [
            self._create_segment(0.0, 0.5, 0, sample_rate),
            self._create_segment(0.5, 1.0, 1, sample_rate, right_context=right_context),
        ]

        for seg in segments:
            windower.process_segment(seg)

        windower.flush()

        window = speech_queue.get()
        assert window.type == 'flush'
        assert window.right_context.size > 0
        assert np.array_equal(window.right_context, right_context)

    def test_flush_with_segment_processes_and_flushes(self, config, sample_rate):
        """flush(segment) should process segment then flush."""
        speech_queue = queue.Queue()
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config)

        left_context = np.random.randn(int(0.3 * sample_rate)).astype(np.float32) * 0.05
        right_context = np.random.randn(int(0.2 * sample_rate)).astype(np.float32) * 0.05

        seg1 = self._create_segment(0.0, 0.5, 0, sample_rate, left_context=left_context)
        windower.process_segment(seg1)

        seg2 = self._create_segment(0.5, 1.0, 1, sample_rate, right_context=right_context)
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

        # First speech sequence
        left_ctx_1 = np.ones(1000, dtype=np.float32) * 0.1
        right_ctx_1 = np.ones(500, dtype=np.float32) * 0.2
        seg1 = self._create_segment(0.0, 1.0, 0, sample_rate,
                                    left_context=left_ctx_1, right_context=right_ctx_1)
        windower.flush(seg1)

        window1 = speech_queue.get()
        assert np.array_equal(window1.left_context, left_ctx_1)
        assert np.array_equal(window1.right_context, right_ctx_1)

        # Second speech sequence - should use NEW contexts
        left_ctx_2 = np.ones(800, dtype=np.float32) * 0.3
        right_ctx_2 = np.ones(400, dtype=np.float32) * 0.4
        seg2 = self._create_segment(5.0, 6.0, 1, sample_rate,
                                    left_context=left_ctx_2, right_context=right_ctx_2)
        windower.flush(seg2)

        window2 = speech_queue.get()
        # Should have new contexts, not old ones
        assert np.array_equal(window2.left_context, left_ctx_2)
        assert np.array_equal(window2.right_context, right_ctx_2)

    def test_context_preserved_for_single_segment(self, config, sample_rate):
        """Single segment flush should preserve both left and right contexts."""
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
