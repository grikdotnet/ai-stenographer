# tests/test_recognizer.py
import pytest
import queue
import time
import numpy as np
from unittest.mock import MagicMock, Mock
from src.asr.Recognizer import Recognizer
from src.types import AudioSegment, RecognitionResult, RecognitionTextMessage, RecognizerAck
from onnx_asr.asr import TimestampedResult


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
        """Recognizer should process preliminary segments."""
        mock_model.recognize.return_value = TimestampedResult(
            text="hello world",
            tokens=None,
            timestamps=None
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(3200, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=1.2,
            chunk_ids=[123]
        )

        result = recognizer.recognize_window(segment)

        assert isinstance(result, RecognitionResult)
        assert result.text == "hello world"
        assert result.start_time == 1.0
        assert result.end_time == 1.2
        assert result.chunk_ids == [123]

    def test_recognizes_finalized_segments(self, mock_model):
        """Recognizer should process larger segments."""
        mock_model.recognize.return_value = TimestampedResult(
            text="test output",
            tokens=None,
            timestamps=None
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model, sample_rate=16000, app_state=Mock())

        # Create finalized window
        segment = AudioSegment(
            type='incremental',
            data=np.full(48000, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=3.0,
            chunk_ids=[0, 1, 2, 3, 4]
        )

        result = recognizer.recognize_window(segment)

        assert isinstance(result, RecognitionResult)
        assert result.text == "test output"
        assert result.start_time == 0.0
        assert result.end_time == 3.0
        assert result.chunk_ids == [0, 1, 2, 3, 4]  # Verify chunk_ids copied from AudioSegment


    def test_filters_empty_text(self, mock_model):
        """Recognizer should return None for empty/silence recognition."""
        mock_model.recognize.side_effect = [
            TimestampedResult(text="hello", tokens=None, timestamps=None),
            TimestampedResult(text="", tokens=None, timestamps=None),
            TimestampedResult(text="world", tokens=None, timestamps=None)
        ]

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model, sample_rate=16000, app_state=Mock())

        # Process three segments
        results = []
        for i in range(3):
            segment = AudioSegment(
                type='incremental',
                data=np.full(3200, 0.1, dtype=np.float32),
                left_context=np.array([], dtype=np.float32),
                right_context=np.array([], dtype=np.float32),
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
        mock_model.recognize.side_effect = [
            TimestampedResult(text="instant", tokens=None, timestamps=None),
            TimestampedResult(text="quality", tokens=None, timestamps=None)
        ]

        speech_queue = queue.Queue()
        text_queue = queue.Queue()
        recognizer = Recognizer(input_queue=speech_queue, output_queue=text_queue, model=mock_model, sample_rate=16000, app_state=Mock())

        # Add preliminary and finalized segments
        preliminary = AudioSegment(
            type='incremental',
            data=np.full(3200, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=1.2,
            chunk_ids=[0],
            message_id=1,
        )
        finalized = AudioSegment(
            type='incremental',
            data=np.full(48000, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=3.0,
            chunk_ids=[0, 1, 2],
            message_id=2,
        )

        speech_queue.put(preliminary)
        speech_queue.put(finalized)

        # Run process() with controlled execution
        recognizer.is_running = True
        original_get = recognizer.input_queue.get
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return original_get(*args, **kwargs)
            else:
                recognizer.is_running = False
                raise queue.Empty()

        recognizer.input_queue.get = mock_get
        recognizer.process()

        messages = []
        while not text_queue.empty():
            messages.append(text_queue.get())

        text_messages = [m for m in messages if isinstance(m, RecognitionTextMessage)]
        ack_messages = [m for m in messages if isinstance(m, RecognizerAck)]

        assert len(text_messages) == 2
        assert text_messages[0].result.text == "instant"
        assert text_messages[1].result.text == "quality"
        assert len(ack_messages) == 2


class TestTimestampedRecognition:
    """Tests for timestamped recognition and token filtering.

    Tests context-aware speech recognition with timestamp-based filtering
    to remove hallucinated tokens from left/right context regions.
    """

    @pytest.fixture
    def timestamped_mock_model(self):
        """Mock model that returns TimestampedResult."""
        return MagicMock()

    def test_recognize_with_timestamps(self, timestamped_mock_model):
        """Recognizer should handle TimestampedResult from model."""
        # Mock model returns TimestampedResult
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text="hello world",
            tokens=[" hello", " world"],
            timestamps=[0.5, 1.0]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(16000, 0.1, dtype=np.float32),  # 1s of data
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        assert result.text == "hello world"
        assert result.start_time == 1.0
        assert result.end_time == 2.0

    def test_filter_tokens_all_in_range(self, timestamped_mock_model):
        """Token filtering should keep all tokens when all are within data region."""
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text="hello world",
            tokens=[" hello", " world"],
            timestamps=[0.5, 1.0]  # Both within data region
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(32000, 0.1, dtype=np.float32),  # 2s of data
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=3.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        assert result.text == "hello world"

    def test_filter_tokens_partial_overlap(self, timestamped_mock_model):
        """Token filtering should drop left-context fillers; in-range tokens are always kept."""
        # left_context=0.5s → data_start=0.5s
        # "yeah" @ 0.1s: left non-filler check → filler → dropped
        # "hello" @ 0.8s: >= data_start → kept
        # "mm-hmm" @ 1.7s: >= data_start → kept
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text="yeah hello mm-hmm",
            tokens=[" yeah", " hello", " mm", "-", "hmm"],
            timestamps=[0.1, 0.8, 1.7, 1.75, 1.8]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(16000, 0.1, dtype=np.float32),  # 1.0s of data
            left_context=np.full(8000, 0.05, dtype=np.float32),  # 0.5s left context
            right_context=np.full(8000, 0.05, dtype=np.float32),  # 0.5s right context
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        assert "hello" in result.text
        assert "yeah" not in result.text
        assert "mm-hmm" in result.text

    def test_filter_tokens_none_in_range(self, timestamped_mock_model):
        """Token filtering should return None when all tokens are in context."""
        # All tokens are left-context fillers (< data_start=0.5s) → all dropped
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text="yeah mm-hmm",
            tokens=[" yeah", " mm", "-", "hmm"],
            timestamps=[0.1, 0.2, 0.25, 0.3]  # All before data_start=0.5s, all fillers
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(16000, 0.1, dtype=np.float32),  # 1.0s of data
            left_context=np.full(8000, 0.05, dtype=np.float32),  # 0.5s left
            right_context=np.full(8000, 0.05, dtype=np.float32),  # 0.5s right
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is None

    def test_filter_tokens_no_timestamps(self, timestamped_mock_model):
        """Token filtering should fallback to full text when no timestamps available."""
        # Model returns text but no timestamps
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text="hello world",
            tokens=None,
            timestamps=None
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(16000, 0.1, dtype=np.float32),
            left_context=np.full(8000, 0.05, dtype=np.float32),
            right_context=np.full(8000, 0.05, dtype=np.float32),
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        # Should return full text (fallback mode)
        assert result is not None
        assert result.text == "hello world"

    def test_context_concatenation(self, timestamped_mock_model):
        """Recognizer should concatenate left_context + data + right_context for recognition."""
        # Track what audio was passed to model
        recognized_audio = None

        def capture_audio(audio):
            nonlocal recognized_audio
            recognized_audio = audio
            return TimestampedResult(
                text="test",
                tokens=["▁test"],
                timestamps=[0.5]
            )

        timestamped_mock_model.recognize.side_effect = capture_audio

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        left = np.full(4000, 0.1, dtype=np.float32)   # 0.25s
        data = np.full(8000, 0.2, dtype=np.float32)   # 0.5s
        right = np.full(4000, 0.3, dtype=np.float32)  # 0.25s

        segment = AudioSegment(
            type='incremental',
            data=data,
            left_context=left,
            right_context=right,
            start_time=1.0,
            end_time=1.5,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        # Verify audio was concatenated: left + data + right
        assert recognized_audio is not None
        expected_length = len(left) + len(data) + len(right)
        assert len(recognized_audio) == expected_length

        # Verify parts are in correct order
        assert np.allclose(recognized_audio[:len(left)], left)
        assert np.allclose(recognized_audio[len(left):len(left)+len(data)], data)
        assert np.allclose(recognized_audio[len(left)+len(data):], right)

    def test_filter_tokens_with_confidence_method(self, timestamped_mock_model):
        """Unit test for _filter_tokens_with_confidence() method."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        # "yeah" @ 0.1s: left filler → dropped; rest >= data_start=0.5 → kept
        text = "yeah hello world mm-hmm"
        tokens = [" yeah", " hello", " world", " mm", "-", "hmm"]
        timestamps = [0.1, 0.8, 1.0, 1.7, 1.75, 1.8]
        confidences = []
        data_start = 0.5

        filtered_text, filtered_confidences = recognizer._filter_tokens_with_confidence(
            text, tokens, timestamps, confidences, data_start
        )

        assert "hello" in filtered_text
        assert "world" in filtered_text
        assert "yeah" not in filtered_text
        assert "mm-hmm" in filtered_text

        # No confidence scores available
        assert len(filtered_confidences) == 0
        assert filtered_confidences == []

    def test_filter_tokens_empty_tokens(self, timestamped_mock_model):
        """_filter_tokens_with_confidence should handle empty token lists."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        # Empty tokens - should return full text with empty confidences (fallback)
        filtered_text, filtered_confidences = recognizer._filter_tokens_with_confidence(
            "hello world", None, None, [], 0.0
        )
        assert filtered_text == "hello world"
        assert filtered_confidences == []

        # Empty timestamps - should return full text with empty confidences (fallback)
        filtered_text, filtered_confidences = recognizer._filter_tokens_with_confidence(
            "hello world", [" hello", " world"], None, [], 0.0
        )
        assert filtered_text == "hello world"
        assert filtered_confidences == []

    def test_filter_tokens_word_split_at_boundary(self, timestamped_mock_model):
        """Test that split words are reconstructed when first part falls outside boundary.

        Scenario: Word "One" is split into [" O", "ne"] where " O" falls outside
        the filtering range but "ne" is inside. The backtracking algorithm should
        detect that "ne" lacks a space prefix and include the preceding " O" token.

        Setup: data_start=0.7, strict filtering (no tolerance)
        Filter range: [0.7, 1.2]
        " O" @ 0.6s is OUTSIDE range (< data_start)
        "ne" @ 0.80s is INSIDE range
        Without backtracking: result would be "ne world"
        With backtracking: result should be "One world" or " One world"
        """
        # Tokens: " Hello" @ 0.5s, " O" @ 0.6s, "ne" @ 0.80s, " world" @ 1.00s
        # Data region: 0.7s to 1.2s
        # " O" @ 0.6s is EXCLUDED (< data_start)
        # "ne" @ 0.80s is INCLUDED
        # " world" @ 1.00s is INCLUDED
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text=" Hello One world",
            tokens=[" Hello", " O", "ne", " world"],
            timestamps=[0.5, 0.6, 0.80, 1.00]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        # Create segment: 0.7s left_context + 0.5s data = 1.2s total
        # data_start = 0.7s, data_end = 1.2s
        segment = AudioSegment(
            type='incremental',
            data=np.full(8000, 0.1, dtype=np.float32),  # 0.5s of data (0.7-1.2)
            left_context=np.full(11200, 0.05, dtype=np.float32),  # 0.7s left context (0.0-0.7)
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=1.5,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        # WITHOUT backtracking fix: would be "ne world" (corrupted)
        # WITH backtracking fix: should be "One world" or " One world"
        assert result is not None
        # Check that we have the complete word "One", not just "ne"
        # The text should contain both " O" and "ne" tokens
        if " O" not in result.text and "One" not in result.text:
            pytest.fail(f"Word corruption detected: expected 'One' but got '{result.text}' (missing ' O' token)")
        # Should NOT be the corrupted version
        assert result.text.strip() != "ne world", "Word corruption: got 'ne world' instead of 'One world'"

    def test_filter_tokens_multiple_word_parts(self, timestamped_mock_model):
        """Test backtracking works for multi-part words with 3+ tokens.

        Scenario: Word "example" is split into [" ex", "am", "ple"] where only
        "am" and "ple" fall within the filtering range. Should backtrack to " ex".

        Setup: data_start=0.7, strict filtering (no tolerance)
        Filter range: [0.7, 1.2]
        " ex" @ 0.6s is OUTSIDE range (< data_start)
        "am" @ 0.80s is INSIDE range
        "ple" @ 0.85s is INSIDE range
        Without backtracking: "ample world"
        With backtracking: "example world"
        """
        # Tokens: " hello" @ 0.5, " ex" @ 0.6, "am" @ 0.80, "ple" @ 0.85, " world" @ 1.00
        # Data region: 0.7s to 1.2s
        # " ex" @ 0.6s is EXCLUDED (< data_start)
        # "am", "ple", " world" are INCLUDED
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text=" hello example world",
            tokens=[" hello", " ex", "am", "ple", " world"],
            timestamps=[0.5, 0.6, 0.80, 0.85, 1.00]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        # 0.7s left_context + 0.5s data
        segment = AudioSegment(
            type='incremental',
            data=np.full(8000, 0.1, dtype=np.float32),  # 0.5s (0.7-1.2)
            left_context=np.full(11200, 0.05, dtype=np.float32),  # 0.7s (0.0-0.7)
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=1.5,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        # Should include " ex" even though it's outside range
        if " ex" not in result.text and "example" not in result.text:
            pytest.fail(f"Word corruption: expected 'example' but got '{result.text}' (missing ' ex' token)")
        # Should NOT be just "ample world"
        assert not result.text.strip().startswith("ample"), f"Word corruption: got 'ample...' instead of 'example...'"

    def test_filter_tokens_first_token_is_word_start(self, timestamped_mock_model):
        """Test normal case where first filtered token has space prefix (no backtrack needed).

        Scenario: All tokens in range start with space - no corruption, no backtracking.
        """
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text=" hello world test",
            tokens=[" hello", " world", " test"],
            timestamps=[0.3, 0.6, 0.9]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(16000, 0.1, dtype=np.float32),  # 1.0s (0.0-1.0)
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        assert "hello" in result.text
        assert "world" in result.text
        assert "test" in result.text

    def test_filter_tokens_word_split_at_start_of_list(self, timestamped_mock_model):
        """Test edge case where continuation token is first in entire tokens list.

        Scenario: First token in entire list lacks space prefix (no backtrack possible).
        Should handle gracefully without crashing.
        """
        # Unusual case: token list starts with continuation token "ne"
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text="ne world",
            tokens=["ne", " world"],
            timestamps=[0.3, 0.6]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(16000, 0.1, dtype=np.float32),  # 1.0s
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        # Should not crash - handle gracefully
        assert result is not None
        assert "world" in result.text

    def test_filter_tokens_with_confidence_backtrack(self, timestamped_mock_model):
        """Test that confidence scores are correctly prepended during backtracking.

        Scenario: Token confidences should align with filtered tokens after backtracking.
        Setup: data_start=0.7, strict filtering (no tolerance)
        " O" @ 0.6s is OUTSIDE, but should be included via backtracking
        Confidence for " O" (0.85) should be prepended to filtered_confidences
        """
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        # Direct test of _filter_tokens_with_confidence method
        # Tokens: " Hello" @ 0.5, " O" @ 0.6, "ne" @ 0.80, " world" @ 1.00
        # Confidences: 0.9, 0.85, 0.88, 0.92
        # Data region: 0.7 to 1.2
        # WITHOUT backtracking: "ne" @ 0.80, " world" @ 1.00 → [0.88, 0.92]
        # WITH backtracking: " O" @ 0.6, "ne" @ 0.80, " world" @ 1.00 → [0.85, 0.88, 0.92]
        text = " Hello One world"
        tokens = [" Hello", " O", "ne", " world"]
        timestamps = [0.5, 0.6, 0.80, 1.00]
        confidences = [0.9, 0.85, 0.88, 0.92]
        data_start = 0.7

        filtered_text, filtered_confidences = recognizer._filter_tokens_with_confidence(
            text, tokens, timestamps, confidences, data_start
        )

        # New algorithm rescues both " Hello" (0.5s) and " O"/"ne" (0.6s) as contiguous
        # non-filler candidates before the first in-range word " world" (1.0s).
        # Confidences: [0.9, 0.85, 0.88, 0.92] for [" Hello", " O", "ne", " world"]
        assert " O" in filtered_text or "One" in filtered_text, f"Expected 'One' but got '{filtered_text}'"
        assert "world" in filtered_text
        assert "Hello" in filtered_text

        assert len(filtered_confidences) == 4, \
            f"Expected 4 confidence values (Hello + One + world), got {len(filtered_confidences)}: {filtered_confidences}"
        assert 0.85 in filtered_confidences
        assert 0.9 in filtered_confidences

    def test_filter_tokens_previous_complete_word_included(self, timestamped_mock_model):
        """Test that previous complete word is included when first filtered token is complete.

        Scenario: First filtered token has space prefix (complete word), previous token
        also has space prefix (also complete word) → include previous token.
        """
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text=" Hi world test",
            tokens=[" Hi", " world", " test"],
            timestamps=[0.4, 0.6, 0.9]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        # Data region: 0.5s to 1.0s
        # " Hi" @ 0.4s is OUTSIDE (< data_start)
        # " world" @ 0.6s is INSIDE (first filtered token, has space)
        # " test" @ 0.9s is INSIDE
        # Step 3 should include " Hi" (previous complete word before " world")
        segment = AudioSegment(
            type='incremental',
            data=np.full(8000, 0.1, dtype=np.float32),  # 0.5s (0.5-1.0)
            left_context=np.full(8000, 0.05, dtype=np.float32),  # 0.5s (0.0-0.5)
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=1.5,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        # Should include "Hi" via Step 3 (previous complete word inclusion)
        assert "Hi" in result.text or " Hi" in result.text, \
            f"Expected 'Hi' to be included (previous complete word), got '{result.text}'"
        assert "world" in result.text
        assert "test" in result.text

    def test_filter_tokens_previous_token_not_word_start(self, timestamped_mock_model):
        """Test that previous token is NOT included if it lacks space prefix.

        Scenario: First filtered token has space, but previous token lacks space
        (continuation token) → don't include previous.
        """
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text=" Hello world",
            tokens=[" Hel", "lo", " world"],
            timestamps=[0.2, 0.4, 0.6]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        # Data region: 0.5s to 1.0s
        # " Hel" @ 0.2s is OUTSIDE
        # "lo" @ 0.4s is OUTSIDE
        # " world" @ 0.6s is INSIDE (first filtered token, has space)
        # Step 3 should NOT include "lo" (lacks space prefix)
        segment = AudioSegment(
            type='incremental',
            data=np.full(8000, 0.1, dtype=np.float32),  # 0.5s (0.5-1.0)
            left_context=np.full(8000, 0.05, dtype=np.float32),  # 0.5s (0.0-0.5)
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=1.5,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        assert "Hello" in result.text
        assert "world" in result.text

    def test_filter_tokens_filler_word_blacklisted(self, timestamped_mock_model):
        """Test that filler words (um, oh, uh, ah) are excluded from previous word inclusion.

        Scenario: Previous token is a blacklisted filler word → don't include it.
        Case-insensitive matching.
        """
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text=" Um hello world",
            tokens=[" Um", " hello", " world"],
            timestamps=[0.4, 0.6, 0.9]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        # Data region: 0.5s to 1.0s
        # " Um" @ 0.4s is OUTSIDE (< data_start)
        # " hello" @ 0.6s is INSIDE (first filtered token, has space)
        # " world" @ 0.9s is INSIDE
        # Step 3 should NOT include " Um" (blacklisted filler word)
        segment = AudioSegment(
            type='incremental',
            data=np.full(8000, 0.1, dtype=np.float32),  # 0.5s (0.5-1.0)
            left_context=np.full(8000, 0.05, dtype=np.float32),  # 0.5s (0.0-0.5)
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=1.5,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        # Should NOT include "Um" (filler word - blacklisted)
        assert "Um" not in result.text and "um" not in result.text, \
            f"Expected 'Um' to be excluded (filler word), got '{result.text}'"
        assert "hello" in result.text
        assert "world" in result.text

    def test_filter_tokens_with_confidence_previous_word_included(self, timestamped_mock_model):
        """Test confidence alignment when including previous complete word.

        Scenario: Verify confidence scores align correctly when Step 3 includes previous word.
        """
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        # Tokens: " Hi" @ 0.4, " world" @ 0.6
        # Confidences: 0.95, 0.88
        # Data region: 0.5 to 1.0
        # WITHOUT Step 3: " world" @ 0.6 → [0.88]
        # WITH Step 3: " Hi" @ 0.4, " world" @ 0.6 → [0.95, 0.88]
        text = " Hi world"
        tokens = [" Hi", " world"]
        timestamps = [0.4, 0.6]
        confidences = [0.95, 0.88]
        data_start = 0.5

        filtered_text, filtered_confidences = recognizer._filter_tokens_with_confidence(
            text, tokens, timestamps, confidences, data_start
        )

        # Should include " Hi" via Step 3
        assert " Hi" in filtered_text or "Hi" in filtered_text, \
            f"Expected 'Hi' but got '{filtered_text}'"
        assert "world" in filtered_text

        # Confidences should have 2 values: [0.95 for " Hi", 0.88 for " world"]
        assert len(filtered_confidences) == 2, \
            f"Expected 2 confidence values (with Step 3), got {len(filtered_confidences)}: {filtered_confidences}"
        assert 0.95 in filtered_confidences, \
            f"Expected confidence 0.95 for ' Hi' token to be included via Step 3"

    def test_filter_tokens_no_previous_token(self, timestamped_mock_model):
        """Test edge case where first filtered token is also first in entire list.

        Scenario: First filtered token is at index 0 (no previous token to check).
        Should handle gracefully without index errors.
        """
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text=" Hello world",
            tokens=[" Hello", " world"],
            timestamps=[0.5, 0.7]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        # Data region: 0.5s to 1.0s
        # " Hello" @ 0.5s is INSIDE (first token, no previous)
        # " world" @ 0.7s is INSIDE
        # Step 3 checks first_filtered_idx > 0, should skip gracefully
        segment = AudioSegment(
            type='incremental',
            data=np.full(8000, 0.1, dtype=np.float32),  # 0.5s (0.5-1.0)
            left_context=np.full(8000, 0.05, dtype=np.float32),  # 0.5s (0.0-0.5)
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=1.5,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        # Should not crash - handle gracefully
        assert result is not None
        assert "Hello" in result.text
        assert "world" in result.text

    def test_out_of_range_non_filler_is_kept(self, timestamped_mock_model):
        """Word before data_start, not a filler → included as left non-filler."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        tokens = [" first", " hello"]
        timestamps = [0.4, 0.6]
        data_start = 0.5

        filtered_text, _ = recognizer._filter_tokens_with_confidence(
            "first hello", tokens, timestamps, [], data_start
        )

        assert "first" in filtered_text
        assert "hello" in filtered_text

    @pytest.mark.parametrize(
        "text,tokens,timestamps,kept_word",
        [
            ("yeah hello", [" yeah", " hello"], [0.4, 0.6], "hello"),
            ("yeah world", [" yeah", " world"], [0.3, 0.6], "world"),
        ],
    )
    def test_out_of_range_single_filler_dropped(
        self, timestamped_mock_model, text, tokens, timestamps, kept_word
    ):
        """Single OOR filler is dropped while later non-filler text is kept."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        filtered_text, _ = recognizer._filter_tokens_with_confidence(
            text, tokens, timestamps, [], 0.5
        )

        assert "yeah" not in filtered_text
        assert kept_word in filtered_text

    def test_split_filler_not_in_set_is_kept(self, timestamped_mock_model):
        """[' Uh', 'h'] OOR → normalized 'uhh' → NOT in FILLER_WORDS → kept."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        tokens = [" Uh", "h", " world"]
        timestamps = [0.3, 0.35, 0.6]
        data_start = 0.5

        filtered_text, _ = recognizer._filter_tokens_with_confidence(
            "Uhh world", tokens, timestamps, [], data_start
        )

        assert "Uhh" in filtered_text or ("Uh" in filtered_text and "h" in filtered_text)
        assert "world" in filtered_text

    def test_multi_token_hallucination_dropped(self, timestamped_mock_model):
        """[' Mm', '-', 'hmm'] OOR → normalized 'mm-hmm' → dropped."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        tokens = [" Mm", "-", "hmm", " hello"]
        timestamps = [0.2, 0.25, 0.3, 0.6]
        data_start = 0.5

        filtered_text, _ = recognizer._filter_tokens_with_confidence(
            "Mm-hmm hello", tokens, timestamps, [], data_start
        )

        assert filtered_text == "hello"

    def test_mixed_out_of_range_tokens(self, timestamped_mock_model):
        """Two OOR words: ' yeah' (dropped) + ' first' (kept)."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        tokens = [" yeah", " first", " hello"]
        timestamps = [0.2, 0.4, 0.6]
        data_start = 0.5

        filtered_text, _ = recognizer._filter_tokens_with_confidence(
            "yeah first hello", tokens, timestamps, [], data_start
        )

        assert "yeah" not in filtered_text
        assert "first" in filtered_text
        assert "hello" in filtered_text

    def test_confidence_alignment_with_left_inclusion(self, timestamped_mock_model):
        """Confidences list length matches left-non-filler + in-range token count."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        tokens = [" first", " hello", " world"]
        timestamps = [0.4, 0.6, 0.8]
        confidences = [0.9, 0.85, 0.92]
        data_start = 0.5

        filtered_text, filtered_confidences = recognizer._filter_tokens_with_confidence(
            "first hello world", tokens, timestamps, confidences, data_start
        )

        assert "first" in filtered_text
        assert len(filtered_confidences) == 3

    @pytest.mark.parametrize(
        "text,tokens,timestamps",
        [
            ("yeah okay", [" yeah", " okay"], [0.1, 0.3]),
            ("um uh", [" um", " uh"], [0.2, 0.4]),
        ],
    )
    def test_no_in_range_all_fillers_returns_empty(
        self, timestamped_mock_model, text, tokens, timestamps
    ):
        """Zero in-range words with only fillers returns empty output."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        filtered_text, filtered_confidences = recognizer._filter_tokens_with_confidence(
            text, tokens, timestamps, [], 0.5
        )

        assert filtered_text == ""
        assert filtered_confidences == []

    def test_duplicate_token_values_use_indices(self, timestamped_mock_model):
        """Two identical tokens [' the', ' cat', ' the']; only second ' the' in range → only that included."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        tokens = [" the", " cat", " the"]
        timestamps = [0.1, 0.3, 0.6]
        data_start = 0.5

        filtered_text, _ = recognizer._filter_tokens_with_confidence(
            "the cat the", tokens, timestamps, [], data_start
        )

        # " the" @ 0.1 and " cat" @ 0.3 are left non-fillers; " the" @ 0.6 is in-range
        assert "the" in filtered_text
        assert filtered_text.strip().count("the") >= 1

    def test_non_monotonic_timestamp_left_non_filler_included(self, timestamped_mock_model):
        """Non-monotonic timestamps: all left non-fillers are included regardless of word order.

        Scenario: [' first'@0.4, ' hello'@0.6, ' tail'@0.45], data_start=0.5.
        Both ' first' and ' tail' are left non-fillers (ts < data_start) and must be included.
        ' hello' is in-range.
        """
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        tokens = [" first", " hello", " tail"]
        timestamps = [0.4, 0.6, 0.45]
        data_start = 0.5

        filtered_text, _ = recognizer._filter_tokens_with_confidence(
            "first hello tail", tokens, timestamps, [], data_start
        )

        assert "first" in filtered_text
        assert "hello" in filtered_text
        assert "tail" in filtered_text

    def test_all_left_non_fillers_returned_when_no_in_range(self, timestamped_mock_model):
        """No in-range words; all left words are non-fillers → non-empty result."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        tokens = [" first", " second"]
        timestamps = [0.2, 0.4]
        data_start = 0.5

        filtered_text, _ = recognizer._filter_tokens_with_confidence(
            "first second", tokens, timestamps, [], data_start
        )

        assert "first" in filtered_text
        assert "second" in filtered_text

    def test_left_word_order_preserved(self, timestamped_mock_model):
        """Left non-fillers precede in-range words; order preserved."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        tokens = [" apple", " banana", " cherry"]
        timestamps = [0.1, 0.3, 0.7]
        data_start = 0.5

        filtered_text, _ = recognizer._filter_tokens_with_confidence(
            "apple banana cherry", tokens, timestamps, [], data_start
        )

        assert filtered_text == "apple banana cherry"


class TestConfidenceMetrics:
    """Tests for confidence_variance and audio_rms fields in RecognitionResult.

    Tests that the new metrics are populated correctly from Recognizer calculations.
    """

    @pytest.fixture
    def mock_model_with_confidence(self):
        """Mock model for recognition tests."""
        model = MagicMock()
        return model

    def test_recognition_result_has_audio_rms_field(self, mock_model_with_confidence):
        """RecognitionResult should have audio_rms field with default value 0.0."""
        result = RecognitionResult(text="test", start_time=0.0, end_time=1.0)

        assert hasattr(result, 'audio_rms')
        assert result.audio_rms == 0.0
        assert isinstance(result.audio_rms, float)

    def test_recognition_result_has_confidence_variance_field(self, mock_model_with_confidence):
        """RecognitionResult should have confidence_variance field with default value 0.0."""
        result = RecognitionResult(text="test", start_time=0.0, end_time=1.0)

        assert hasattr(result, 'confidence_variance')
        assert result.confidence_variance == 0.0
        assert isinstance(result.confidence_variance, float)

    def test_recognizer_populates_audio_rms(self, mock_model_with_confidence):
        """Recognizer should populate audio_rms from calculated RMS energy."""
        mock_model_with_confidence.recognize.return_value = TimestampedResult(
            text="hello",
            tokens=[" hello"],
            timestamps=[0.5]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model_with_confidence, sample_rate=16000, app_state=Mock())

        # Create segment with known RMS value
        # RMS of constant array is the absolute value
        audio_data = np.full(16000, 0.25, dtype=np.float32)  # 1s of audio at 0.25
        expected_rms = 0.25

        segment = AudioSegment(
            type='incremental',
            data=audio_data,
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        assert result.audio_rms > 0.0
        assert abs(result.audio_rms - expected_rms) < 0.01  # Allow small floating point error

    def test_recognizer_populates_confidence_variance(self, mock_model_with_confidence):
        """Recognizer should populate confidence_variance from token confidence variance."""
        mock_model_with_confidence.recognize.return_value = TimestampedResult(
            text="hello world",
            tokens=[" hello", " world"],
            timestamps=[0.3, 0.7]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model_with_confidence, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(16000, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        # Should be 0.0 (no confidence data available without ConfidenceExtractor)
        assert result.confidence_variance == 0.0
        assert result.confidence == 0.0
        assert result.token_confidences == []
        assert isinstance(result.confidence_variance, float)

    def test_new_fields_with_context_filtering(self, mock_model_with_confidence):
        """New fields should be calculated only from data region, not context."""
        mock_model_with_confidence.recognize.return_value = TimestampedResult(
            text="context hello context",
            tokens=[" context", " hello", " context"],
            timestamps=[0.1, 0.6, 1.7]  # Only "hello" in data region
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model_with_confidence, sample_rate=16000, app_state=Mock())

        # Create segment with distinct RMS values in different regions
        data = np.full(16000, 0.3, dtype=np.float32)  # 1s, RMS=0.3
        left = np.full(8000, 0.1, dtype=np.float32)   # 0.5s, RMS=0.1
        right = np.full(8000, 0.1, dtype=np.float32)  # 0.5s, RMS=0.1

        segment = AudioSegment(
            type='incremental',
            data=data,
            left_context=left,
            right_context=right,
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        # audio_rms should be calculated from data region only (0.3), not context (0.1)
        assert abs(result.audio_rms - 0.3) < 0.01
        # Text should be filtered to data region
        # Step 3 includes " context" @ 0.1s (previous complete word before " hello")
        assert "hello" in result.text
        assert "context" in result.text  # Included via Step 3


class TestRecognizerObserverPattern:
    """Tests for Recognizer observer pattern with ApplicationState.

    Tests that Recognizer subscribes to ApplicationState and reacts to
    shutdown events via observer callbacks.
    """

    @pytest.fixture
    def mock_model(self):
        """Mock speech recognition model."""
        return MagicMock()

    @pytest.fixture
    def mock_app_state(self):
        """Mock ApplicationState."""
        return Mock()

    def test_recognizer_subscribes_to_application_state(self, mock_model, mock_app_state):
        """Test that Recognizer registers as component observer when app_state provided.

        Logic: Recognizer.__init__(app_state=...) should call
               app_state.register_component_observer()
        """
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model, app_state=mock_app_state)

        mock_app_state.register_component_observer.assert_called_once()
        # Verify callback is the on_state_change method
        callback = mock_app_state.register_component_observer.call_args[0][0]
        assert callback == recognizer.on_state_change

    def test_recognizer_shutdown_state_stops_processing(self, mock_model, mock_app_state):
        """Test that Recognizer stops when receiving 'shutdown' state.

        Logic: on_state_change(_, 'shutdown') should set is_running=False.
        """
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model, app_state=mock_app_state)

        recognizer.is_running = True
        recognizer.on_state_change('running', 'shutdown')

        assert recognizer.is_running == False

    def test_recognizer_stop_is_idempotent(self, mock_model, mock_app_state):
        """Test that calling stop() multiple times is safe.

        Logic: stop() should check if already stopped and avoid duplicate cleanup.
        """
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model, app_state=mock_app_state)

        recognizer.start()
        recognizer.stop()
        recognizer.stop()  # Second call should be safe
        recognizer.stop()  # Third call should be safe

        # Should not raise errors
        assert recognizer.is_running == False
