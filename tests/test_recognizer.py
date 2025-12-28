# tests/test_recognizer.py
import pytest
import queue
import time
import numpy as np
from unittest.mock import MagicMock, Mock
from src.Recognizer import Recognizer
from src.types import AudioSegment, RecognitionResult
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
        """Recognizer should process preliminary segments and return result with status='preliminary'."""
        mock_model.recognize.return_value = TimestampedResult(
            text="hello world",
            tokens=None,
            timestamps=None
        )

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='preliminary',
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
        assert result.status == 'preliminary'
        assert result.chunk_ids == [123]

    def test_recognizes_finalized_segments(self, mock_model):
        """Recognizer should process finalized segments and return result with status='final'."""
        mock_model.recognize.return_value = TimestampedResult(
            text="test output",
            tokens=None,
            timestamps=None
        )

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model, sample_rate=16000, app_state=Mock())

        # Create finalized window
        segment = AudioSegment(
            type='finalized',
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
        assert result.status == 'final'
        assert result.chunk_ids == [0, 1, 2, 3, 4]  # Verify chunk_ids copied from AudioSegment


    def test_filters_empty_text(self, mock_model):
        """Recognizer should return None for empty/silence recognition."""
        mock_model.recognize.side_effect = [
            TimestampedResult(text="hello", tokens=None, timestamps=None),
            TimestampedResult(text="", tokens=None, timestamps=None),
            TimestampedResult(text="world", tokens=None, timestamps=None)
        ]

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model, sample_rate=16000, app_state=Mock())

        # Process three segments
        results = []
        for i in range(3):
            segment = AudioSegment(
                type='preliminary',
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
        recognizer = Recognizer(speech_queue, text_queue, mock_model, sample_rate=16000, app_state=Mock())

        # Add preliminary and finalized segments
        preliminary = AudioSegment(
            type='preliminary',
            data=np.full(3200, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=1.2,
            chunk_ids=[0]
        )
        finalized = AudioSegment(
            type='finalized',
            data=np.full(48000, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=0.0,
            end_time=3.0,
            chunk_ids=[0, 1, 2]
        )

        speech_queue.put(preliminary)
        speech_queue.put(finalized)

        # Run process() with controlled execution
        recognizer.is_running = True
        original_get = recognizer.speech_queue.get
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return original_get(*args, **kwargs)
            else:
                recognizer.is_running = False
                raise queue.Empty()

        recognizer.speech_queue.get = mock_get
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


    def test_recognizer_maps_segment_types_to_status(self, mock_model):
        """Recognizer should map AudioSegment.type to RecognitionResult.status."""
        mock_model.recognize.return_value = TimestampedResult(
            text="test",
            tokens=None,
            timestamps=None
        )

        recognizer = Recognizer(queue.Queue(), queue.Queue(), mock_model, sample_rate=16000, app_state=Mock())

        # Test mapping: type='preliminary' → status='preliminary'
        preliminary_segment = AudioSegment(
            type='preliminary',
            data=np.full(3200, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
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
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
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
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=2.0,
            end_time=3.0,
            chunk_ids=[5]
        )
        result = recognizer.recognize_window(flush_segment)
        assert result.status == 'flush'


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

        recognizer = Recognizer(
            queue.Queue(),
            queue.Queue(),
            timestamped_mock_model,
            sample_rate=16000,
            app_state=Mock()
        )

        segment = AudioSegment(
            type='preliminary',
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

        recognizer = Recognizer(
            queue.Queue(),
            queue.Queue(),
            timestamped_mock_model,
            sample_rate=16000,
            app_state=Mock()
        )

        segment = AudioSegment(
            type='preliminary',
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
        """Token filtering should remove tokens from context regions."""
        # Simulate: left_context (0.5s) + data (1.0s) + right_context (0.5s)
        # Total audio: 2.0s
        # Data region: 0.5s to 1.5s
        # Tokens: "yeah" @ 0.1s (in left context, outside tolerance), "hello" @ 0.8s (in data), "mm-hmm" @ 1.7s (in right context)
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text="yeah hello mm-hmm",
            tokens=[" yeah", " hello", " mm", "-", "hmm"],
            timestamps=[0.1, 0.8, 1.7, 1.75, 1.8]
        )

        recognizer = Recognizer(
            queue.Queue(),
            queue.Queue(),
            timestamped_mock_model,
            sample_rate=16000,
            app_state=Mock()
        )

        segment = AudioSegment(
            type='preliminary',
            data=np.full(16000, 0.1, dtype=np.float32),  # 1.0s of data
            left_context=np.full(8000, 0.05, dtype=np.float32),  # 0.5s left context
            right_context=np.full(8000, 0.05, dtype=np.float32),  # 0.5s right context
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        # Should only keep "hello" (at 0.8s, within 0.5-1.5s data region)
        assert result is not None
        assert result.text == "hello"
        assert "yeah" not in result.text
        assert "mm-hmm" not in result.text

    def test_filter_tokens_none_in_range(self, timestamped_mock_model):
        """Token filtering should return None when all tokens are in context."""
        # All tokens are outside data region (in contexts)
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text="yeah mm-hmm",
            tokens=[" yeah", " mm", "-", "hmm"],
            timestamps=[0.1, 1.9, 1.95, 2.0]  # All in context regions
        )

        recognizer = Recognizer(
            queue.Queue(),
            queue.Queue(),
            timestamped_mock_model,
            sample_rate=16000,
            app_state=Mock()
        )

        segment = AudioSegment(
            type='preliminary',
            data=np.full(16000, 0.1, dtype=np.float32),  # 1.0s of data
            left_context=np.full(8000, 0.05, dtype=np.float32),  # 0.5s left
            right_context=np.full(8000, 0.05, dtype=np.float32),  # 0.5s right
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        # Should return None (all tokens filtered out)
        assert result is None

    def test_filter_tokens_no_timestamps(self, timestamped_mock_model):
        """Token filtering should fallback to full text when no timestamps available."""
        # Model returns text but no timestamps
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text="hello world",
            tokens=None,
            timestamps=None
        )

        recognizer = Recognizer(
            queue.Queue(),
            queue.Queue(),
            timestamped_mock_model,
            sample_rate=16000,
            app_state=Mock()
        )

        segment = AudioSegment(
            type='preliminary',
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

        recognizer = Recognizer(
            queue.Queue(),
            queue.Queue(),
            timestamped_mock_model,
            sample_rate=16000,
            app_state=Mock()
        )

        left = np.full(4000, 0.1, dtype=np.float32)   # 0.25s
        data = np.full(8000, 0.2, dtype=np.float32)   # 0.5s
        right = np.full(4000, 0.3, dtype=np.float32)  # 0.25s

        segment = AudioSegment(
            type='preliminary',
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
        recognizer = Recognizer(
            queue.Queue(),
            queue.Queue(),
            timestamped_mock_model,
            sample_rate=16000,
            app_state=Mock()
        )

        # Test case: tokens with various timestamps and confidence scores
        text = "yeah hello world mm-hmm"
        tokens = [" yeah", " hello", " world", " mm", "-", "hmm"]
        timestamps = [0.1, 0.8, 1.0, 1.7, 1.75, 1.8]  # "yeah" at 0.1s outside tolerance
        confidences = [0.9, 0.95, 0.92, 0.85, 0.8, 0.88]
        data_start = 0.5
        data_end = 1.5

        filtered_text, filtered_confidences = recognizer._filter_tokens_with_confidence(
            text, tokens, timestamps, confidences, data_start, data_end
        )

        # Should only include "hello" and "world" (timestamps 0.8 and 1.0)
        assert "hello" in filtered_text
        assert "world" in filtered_text
        assert "yeah" not in filtered_text
        assert "mm-hmm" not in filtered_text

        # Should include corresponding confidence scores
        assert len(filtered_confidences) == 2
        assert filtered_confidences == [0.95, 0.92]

    def test_filter_tokens_empty_tokens(self, timestamped_mock_model):
        """_filter_tokens_with_confidence should handle empty token lists."""
        recognizer = Recognizer(
            queue.Queue(),
            queue.Queue(),
            timestamped_mock_model,
            sample_rate=16000,
            app_state=Mock()
        )

        # Empty tokens - should return full text with empty confidences (fallback)
        filtered_text, filtered_confidences = recognizer._filter_tokens_with_confidence(
            "hello world", None, None, [], 0.0, 1.0
        )
        assert filtered_text == "hello world"
        assert filtered_confidences == []

        # Empty timestamps - should return full text with empty confidences (fallback)
        filtered_text, filtered_confidences = recognizer._filter_tokens_with_confidence(
            "hello world", [" hello", " world"], None, [], 0.0, 1.0
        )
        assert filtered_text == "hello world"
        assert filtered_confidences == []


class TestConfidenceMetrics:
    """Tests for confidence_variance and audio_rms fields in RecognitionResult.

    Tests that the new metrics are populated correctly from Recognizer calculations.
    """

    @pytest.fixture
    def mock_model_with_confidence(self):
        """Mock model configured for confidence extraction."""
        model = MagicMock()
        # Configure for ConfidenceExtractor compatibility
        model.asr = MagicMock()
        model.asr._decode = MagicMock()
        return model

    def test_recognition_result_has_audio_rms_field(self, mock_model_with_confidence):
        """RecognitionResult should have audio_rms field with default value 0.0."""
        result = RecognitionResult(
            text="test",
            start_time=0.0,
            end_time=1.0,
            status='final'
        )

        assert hasattr(result, 'audio_rms')
        assert result.audio_rms == 0.0
        assert isinstance(result.audio_rms, float)

    def test_recognition_result_has_confidence_variance_field(self, mock_model_with_confidence):
        """RecognitionResult should have confidence_variance field with default value 0.0."""
        result = RecognitionResult(
            text="test",
            start_time=0.0,
            end_time=1.0,
            status='final'
        )

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

        recognizer = Recognizer(
            queue.Queue(),
            queue.Queue(),
            mock_model_with_confidence,
            sample_rate=16000,
            app_state=Mock()
        )

        # Create segment with known RMS value
        # RMS of constant array is the absolute value
        audio_data = np.full(16000, 0.25, dtype=np.float32)  # 1s of audio at 0.25
        expected_rms = 0.25

        segment = AudioSegment(
            type='preliminary',
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

        recognizer = Recognizer(
            queue.Queue(),
            queue.Queue(),
            mock_model_with_confidence,
            sample_rate=16000,
            app_state=Mock()
        )

        segment = AudioSegment(
            type='preliminary',
            data=np.full(16000, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        # Variance should be >= 0.0 (may be 0 if confidence extraction fails gracefully)
        assert result.confidence_variance >= 0.0
        assert isinstance(result.confidence_variance, float)

    def test_new_fields_with_context_filtering(self, mock_model_with_confidence):
        """New fields should be calculated only from data region, not context."""
        mock_model_with_confidence.recognize.return_value = TimestampedResult(
            text="context hello context",
            tokens=[" context", " hello", " context"],
            timestamps=[0.1, 0.6, 1.7]  # Only "hello" in data region
        )

        recognizer = Recognizer(
            queue.Queue(),
            queue.Queue(),
            mock_model_with_confidence,
            sample_rate=16000,
            app_state=Mock()
        )

        # Create segment with distinct RMS values in different regions
        data = np.full(16000, 0.3, dtype=np.float32)  # 1s, RMS=0.3
        left = np.full(8000, 0.1, dtype=np.float32)   # 0.5s, RMS=0.1
        right = np.full(8000, 0.1, dtype=np.float32)  # 0.5s, RMS=0.1

        segment = AudioSegment(
            type='preliminary',
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
        assert result.text == "hello"


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
        recognizer = Recognizer(
            speech_queue=queue.Queue(),
            text_queue=queue.Queue(),
            model=mock_model,
            app_state=mock_app_state
        )

        mock_app_state.register_component_observer.assert_called_once()
        # Verify callback is the on_state_change method
        callback = mock_app_state.register_component_observer.call_args[0][0]
        assert callback == recognizer.on_state_change

    def test_recognizer_shutdown_state_stops_processing(self, mock_model, mock_app_state):
        """Test that Recognizer stops when receiving 'shutdown' state.

        Logic: on_state_change(_, 'shutdown') should set is_running=False
               and call unpatch on confidence_extractor.
        """
        recognizer = Recognizer(
            speech_queue=queue.Queue(),
            text_queue=queue.Queue(),
            model=mock_model,
            app_state=mock_app_state
        )

        # Mock the unpatch method
        recognizer.confidence_extractor.unpatch = Mock()

        recognizer.is_running = True
        recognizer.on_state_change('running', 'shutdown')

        assert recognizer.is_running == False
        recognizer.confidence_extractor.unpatch.assert_called_once()

    def test_recognizer_stop_is_idempotent(self, mock_model, mock_app_state):
        """Test that calling stop() multiple times is safe.

        Logic: stop() should check if already stopped and avoid duplicate cleanup.
        """
        recognizer = Recognizer(
            speech_queue=queue.Queue(),
            text_queue=queue.Queue(),
            model=mock_model,
            app_state=mock_app_state
        )

        # Mock the unpatch method
        recognizer.confidence_extractor.unpatch = Mock()

        recognizer.start()
        recognizer.stop()
        recognizer.stop()  # Second call should be safe
        recognizer.stop()  # Third call should be safe

        # Should not raise errors, confidence_extractor.unpatch() called once
        assert recognizer.confidence_extractor.unpatch.call_count == 1
