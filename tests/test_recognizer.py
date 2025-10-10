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
        """Recognizer should process preliminary segments and return result with status='preliminary'."""
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

        result = recognizer.recognize_window(segment)

        assert isinstance(result, RecognitionResult)
        assert result.text == "hello world"
        assert result.start_time == 1.0
        assert result.end_time == 1.2
        assert result.status == 'preliminary'
        assert result.chunk_ids == [0]  # Verify chunk_ids copied from AudioSegment


    def test_recognizes_finalized_segments(self, mock_model):
        """Recognizer should process finalized segments and return result with status='final'."""
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

        result = recognizer.recognize_window(segment)

        assert isinstance(result, RecognitionResult)
        assert result.text == "test output"
        assert result.start_time == 0.0
        assert result.end_time == 3.0
        assert result.status == 'final'
        assert result.chunk_ids == [0, 1, 2, 3, 4]  # Verify chunk_ids copied from AudioSegment


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
            result = recognizer.recognize_window(segment)
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

        # Verify both results with correct status values
        results = []
        while not text_queue.empty():
            results.append(text_queue.get())

        assert len(results) == 2
        assert results[0].text == "instant"
        assert results[0].status == 'preliminary'
        assert results[1].text == "quality"
        assert results[1].status == 'final'

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

        result = recognizer.recognize_window(segment)

        assert result.start_time == 5.123
        assert result.end_time == 8.456

    def test_chunk_ids_preserved_in_recognition(self, mock_model):
        """RecognitionResult should preserve chunk_ids exactly from AudioSegment."""
        mock_model.recognize.return_value = "chunk test"

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model)

        # Test with various chunk_id patterns
        test_cases = [
            [1],              # Single chunk (preliminary)
            [1, 2, 3],        # Multiple consecutive chunks
            [5, 6, 7, 8, 9],  # Larger window
        ]

        for chunk_ids in test_cases:
            segment = AudioSegment(
                type='finalized',
                data=np.full(48000, 0.1, dtype=np.float32),
                start_time=0.0,
                end_time=3.0,
                chunk_ids=chunk_ids
            )

            result = recognizer.recognize_window(segment)

            # Verify chunk_ids are copied exactly (not lost or duplicated)
            assert result.chunk_ids == chunk_ids
            assert len(result.chunk_ids) == len(chunk_ids)

    def test_recognizer_maps_segment_types_to_status(self, mock_model):
        """Recognizer should map AudioSegment.type to RecognitionResult.status."""
        mock_model.recognize.return_value = "test"

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model)

        # Test mapping: type='preliminary' → status='preliminary'
        preliminary_segment = AudioSegment(
            type='preliminary',
            data=np.full(3200, 0.1, dtype=np.float32),
            start_time=1.0,
            end_time=1.2,
            chunk_ids=[0]
        )
        result = recognizer.recognize_window(preliminary_segment)
        assert result.status == 'preliminary'

        # Test mapping: type='finalized' → status='final'
        finalized_segment = AudioSegment(
            type='finalized',
            data=np.full(48000, 0.1, dtype=np.float32),
            start_time=0.0,
            end_time=3.0,
            chunk_ids=[0, 1, 2]
        )
        result = recognizer.recognize_window(finalized_segment)
        assert result.status == 'final'

        # Test mapping: type='flush' → status='flush'
        flush_segment = AudioSegment(
            type='flush',
            data=np.full(16000, 0.1, dtype=np.float32),
            start_time=2.0,
            end_time=3.0,
            chunk_ids=[5]
        )
        result = recognizer.recognize_window(flush_segment)
        assert result.status == 'flush'
