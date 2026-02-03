# tests/test_adaptive_windower_queue_full.py
import pytest
import numpy as np
import queue
from src.sound.AdaptiveWindower import AdaptiveWindower
from src.types import AudioSegment


class TestAdaptiveWindowerQueueFull:
    """Tests for AdaptiveWindower queue full handling.

    Verifies that AdaptiveWindower uses non-blocking queue operations
    and properly handles full speech_queue by dropping windows with logging.
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
                        sample_rate: int = 16000) -> AudioSegment:
        """Helper to create preliminary AudioSegment."""
        samples = int(duration_sec * sample_rate)
        data = np.random.randn(samples).astype(np.float32) * 0.1
        return AudioSegment(
            type='preliminary',
            data=data,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=start_time,
            end_time=start_time + duration_sec,
            chunk_ids=[chunk_id]
        )

    def test_speech_queue_full_drops_window_and_logs(self, config, sample_rate, caplog):
        """Full speech_queue causes non-blocking drop with warning.

        Logic: When speech_queue.put_nowait() raises queue.Full,
        window is dropped, counter incremented, warning logged.
        """
        # Create small queue (maxsize=1) to trigger full condition
        speech_queue = queue.Queue(maxsize=1)
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config, verbose=True)

        # Fill the speech_queue
        dummy_segment = AudioSegment(
            type='finalized',
            data=np.zeros(512, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=0.032,
            chunk_ids=[999]
        )
        speech_queue.put(dummy_segment)

        # Process segments that trigger window emission
        # Need 2+ segments AND >= 3.0 seconds duration
        seg1 = self._create_segment(0.0, 2.0, chunk_id=0, sample_rate=sample_rate)
        seg2 = self._create_segment(2.0, 1.5, chunk_id=1, sample_rate=sample_rate)

        windower.process_segment(seg1)
        windower.process_segment(seg2)  # This triggers _emit_window()

        # Verify window was dropped (counter incremented)
        assert windower.dropped_windows == 1

        # Verify warning logged
        assert "speech_queue full" in caplog.text
        assert "total drops: 1" in caplog.text

    def test_flush_with_full_queue_drops_window_and_logs(self, config, sample_rate, caplog):
        """Flush with full speech_queue drops flush window.

        Logic: flush() also uses _put_window_nonblocking(), so full queue drops flush window.
        """
        speech_queue = queue.Queue(maxsize=1)
        windower = AdaptiveWindower(speech_queue=speech_queue, config=config, verbose=True)

        # Fill queue
        dummy_segment = AudioSegment(
            type='finalized',
            data=np.zeros(512, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=0.032,
            chunk_ids=[999]
        )
        speech_queue.put(dummy_segment)

        # Add segment to buffer (not enough to trigger emission)
        seg1 = self._create_segment(0.0, 1.0, chunk_id=0, sample_rate=sample_rate)
        windower.process_segment(seg1)

        # Flush should attempt to put window
        windower.flush()

        # Verify flush window was dropped
        assert windower.dropped_windows == 1
        assert "speech_queue full" in caplog.text
