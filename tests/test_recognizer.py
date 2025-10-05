# tests/test_recognizer.py
import pytest
import queue
import time
import numpy as np
from unittest.mock import MagicMock
from src.Recognizer import Recognizer
from src.types import AudioSegment, RecognitionResult


class TestRecognizer:
    """Tests for Recognizer with AudioSegment and RecognitionResult types.

    Tests recognition of preliminary (instant) and finalized (high-quality) segments,
    output format validation, and silence detection.
    """

    @pytest.fixture
    def mock_model(self):
        """Mock speech recognition model."""
        return MagicMock()

    def test_recognizes_preliminary_segments(self, mock_model):
        """Recognizer should process preliminary segments and return result with is_preliminary=True."""
        mock_model.recognize.return_value = "hello world"

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model)

        # Create preliminary segment
        segment = AudioSegment(
            type='preliminary',
            data=np.full(3200, 0.1, dtype=np.float32),
            start_time=1.0,
            end_time=1.2,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment, is_preliminary=True)

        assert isinstance(result, RecognitionResult)
        assert result.text == "hello world"
        assert result.start_time == 1.0
        assert result.end_time == 1.2
        assert result.is_preliminary is True


    def test_recognizes_finalized_segments(self, mock_model):
        """Recognizer should process finalized segments and return result with is_preliminary=False."""
        mock_model.recognize.return_value = "test output"

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model)

        # Create finalized window
        segment = AudioSegment(
            type='finalized',
            data=np.full(48000, 0.1, dtype=np.float32),
            start_time=0.0,
            end_time=3.0,
            chunk_ids=[0, 1, 2, 3, 4]
        )

        result = recognizer.recognize_window(segment, is_preliminary=False)

        assert isinstance(result, RecognitionResult)
        assert result.text == "test output"
        assert result.start_time == 0.0
        assert result.end_time == 3.0
        assert result.is_preliminary is False


    def test_filters_empty_text(self, mock_model):
        """Recognizer should return None for empty/silence recognition."""
        mock_model.recognize.side_effect = ["hello", "", "world"]  # Empty in middle

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model)

        # Process three segments
        results = []
        for i in range(3):
            segment = AudioSegment(
                type='preliminary',
                data=np.full(3200, 0.1, dtype=np.float32),
                start_time=float(i),
                end_time=float(i + 0.2),
                chunk_ids=[i]
            )
            result = recognizer.recognize_window(segment, is_preliminary=True)
            if result is not None:
                results.append(result)

        # Should only have 2 results (empty filtered out)
        assert len(results) == 2
        assert results[0].text == "hello"
        assert results[1].text == "world"


    def test_process_method_with_type_detection(self, mock_model):
        """Recognizer.process() should auto-detect preliminary vs finalized from segment type."""
        mock_model.recognize.side_effect = ["instant", "quality"]

        chunk_queue = queue.Queue()
        text_queue = queue.Queue()
        recognizer = Recognizer(chunk_queue, text_queue, mock_model)

        # Add preliminary and finalized segments
        preliminary = AudioSegment(
            type='preliminary',
            data=np.full(3200, 0.1, dtype=np.float32),
            start_time=1.0,
            end_time=1.2,
            chunk_ids=[0]
        )
        finalized = AudioSegment(
            type='finalized',
            data=np.full(48000, 0.1, dtype=np.float32),
            start_time=0.0,
            end_time=3.0,
            chunk_ids=[0, 1, 2]
        )

        chunk_queue.put(preliminary)
        chunk_queue.put(finalized)

        # Run process() with controlled execution
        recognizer.is_running = True
        original_get = recognizer.chunk_queue.get
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return original_get(*args, **kwargs)
            else:
                recognizer.is_running = False
                raise queue.Empty()

        recognizer.chunk_queue.get = mock_get
        recognizer.process()

        # Verify both results with correct is_preliminary flags
        results = []
        while not text_queue.empty():
            results.append(text_queue.get())

        assert len(results) == 2
        assert results[0].text == "instant"
        assert results[0].is_preliminary is True
        assert results[1].text == "quality"
        assert results[1].is_preliminary is False

    def test_preserves_timing_information(self, mock_model):
        """RecognitionResult should preserve exact timing from AudioSegment."""
        mock_model.recognize.return_value = "timing test"

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model)

        segment = AudioSegment(
            type='finalized',
            data=np.full(48000, 0.1, dtype=np.float32),
            start_time=5.123,
            end_time=8.456,
            chunk_ids=[10, 11, 12]
        )

        result = recognizer.recognize_window(segment, is_preliminary=False)

        assert result.start_time == 5.123
        assert result.end_time == 8.456
