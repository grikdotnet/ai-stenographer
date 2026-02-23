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
    """Tests for timestamped recognition.

    Tests context-aware speech recognition and PostRecognitionFilter delegation.
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

    def test_filter_tokens_filler_word_blacklisted(self, timestamped_mock_model):
        """'um' is always dropped by PostRecognitionFilter."""
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text=" Um hello world",
            tokens=[" Um", " hello", " world"],
            timestamps=[0.4, 0.6, 0.9]
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=timestamped_mock_model, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(8000, 0.1, dtype=np.float32),
            left_context=np.full(8000, 0.05, dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=1.5,
            chunk_ids=[0]
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        assert "Um" not in result.text and "um" not in result.text
        assert "hello" in result.text
        assert "world" in result.text

    def test_recognizer_delegates_to_post_recognition_filter(self, timestamped_mock_model):
        """Recognizer delegates filtering to PostRecognitionFilter and uses its return value."""
        timestamped_mock_model.recognize.return_value = TimestampedResult(
            text="hello world",
            tokens=[" hello", " world"],
            timestamps=[0.3, 0.7],
        )

        mock_filter = Mock()
        mock_filter.filter.return_value = ("filtered result", [0.7, 0.8])

        recognizer = Recognizer(
            input_queue=queue.Queue(),
            output_queue=queue.Queue(),
            model=timestamped_mock_model,
            sample_rate=16000,
            app_state=Mock(),
            post_recognition_filter=mock_filter,
        )

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

        mock_filter.filter.assert_called_once()
        call_args = mock_filter.filter.call_args
        assert call_args[0][0] == "hello world"
        assert call_args[0][1] == [" hello", " world"]

        assert result is not None
        assert result.text == "filtered result"
        assert result.token_confidences == [0.7, 0.8]


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
            timestamps=[0.3, 0.7],
            logprobs=None,
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

    def test_recognizer_extracts_logprobs_as_token_confidences(self, mock_model_with_confidence):
        """Recognizer should convert logprobs to per-token confidences via exp(logprob)."""
        mock_model_with_confidence.recognize.return_value = TimestampedResult(
            text="hello world",
            tokens=[" hello", " world"],
            timestamps=[0.3, 0.7],
            logprobs=[-0.1, -0.3],
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model_with_confidence, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(16000, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0],
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        assert result.token_confidences[0] == pytest.approx(np.exp(-0.1), rel=1e-6)
        assert result.token_confidences[1] == pytest.approx(np.exp(-0.3), rel=1e-6)
        assert result.confidence == pytest.approx(np.mean([np.exp(-0.1), np.exp(-0.3)]), rel=1e-6)
        assert result.confidence_variance == pytest.approx(np.var([np.exp(-0.1), np.exp(-0.3)]), rel=1e-6)

    def test_recognizer_handles_none_logprobs(self, mock_model_with_confidence):
        """Recognizer should degrade to empty confidences when logprobs is None."""
        mock_model_with_confidence.recognize.return_value = TimestampedResult(
            text="hello world",
            tokens=[" hello", " world"],
            timestamps=[0.3, 0.7],
            logprobs=None,
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model_with_confidence, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(16000, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0],
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        assert result.token_confidences == []
        assert result.confidence == 0.0

    def test_recognizer_handles_misaligned_logprobs(self, mock_model_with_confidence):
        """Recognizer should degrade to empty confidences when logprobs length mismatches tokens."""
        mock_model_with_confidence.recognize.return_value = TimestampedResult(
            text="hello world",
            tokens=[" hello", " world"],
            timestamps=[0.3, 0.7],
            logprobs=[-0.1],
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model_with_confidence, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(16000, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0],
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        assert result.token_confidences == []
        assert result.confidence == 0.0

    def test_recognizer_handles_nonfinite_logprobs(self, mock_model_with_confidence):
        """Recognizer should degrade to empty confidences when any logprob is non-finite."""
        mock_model_with_confidence.recognize.return_value = TimestampedResult(
            text="hello world",
            tokens=[" hello", " world"],
            timestamps=[0.3, 0.7],
            logprobs=[float('nan'), -0.2],
        )

        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model_with_confidence, sample_rate=16000, app_state=Mock())

        segment = AudioSegment(
            type='incremental',
            data=np.full(16000, 0.1, dtype=np.float32),
            left_context=np.array([], dtype=np.float32),
            right_context=np.array([], dtype=np.float32),
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[0],
        )

        result = recognizer.recognize_window(segment)

        assert result is not None
        assert result.token_confidences == []

    def test_recognizer_handles_large_positive_logprob(self, mock_model_with_confidence):
        """_extract_token_confidences should clamp large positive logprobs to 1.0 without crash."""
        recognizer = Recognizer(input_queue=queue.Queue(), output_queue=queue.Queue(), model=mock_model_with_confidence, sample_rate=16000, app_state=Mock())

        mock_result = Mock()
        mock_result.logprobs = [1000.0, -0.1]
        mock_result.tokens = [" a", " b"]

        confidences = recognizer._extract_token_confidences(mock_result)

        assert confidences[0] == pytest.approx(1.0)
        assert confidences[1] == pytest.approx(np.exp(-0.1), rel=1e-6)


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
