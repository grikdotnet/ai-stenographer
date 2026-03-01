"""Tests for RemoteRecognitionPublisher (Phase 6).

Strategy: inject a mock RecognitionResultPublisher; call dispatch() with JSON strings
and verify the correct publisher method is called with correctly decoded fields.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.client.tk.RemoteRecognitionPublisher import RemoteRecognitionPublisher
from src.types import RecognitionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_json(status: str = "partial", **overrides) -> str:
    base = {
        "type": "recognition_result",
        "session_id": "sess-1",
        "status": status,
        "text": "hello world",
        "start_time": 1.0,
        "end_time": 2.5,
        "chunk_ids": [10, 11],
        "utterance_id": 3,
        "confidence": 0.88,
        "token_confidences": [0.9, 0.85],
        "audio_rms": 0.07,
        "confidence_variance": 0.002,
    }
    base.update(overrides)
    return json.dumps(base)


def _make_publisher() -> tuple[RemoteRecognitionPublisher, MagicMock]:
    publisher = MagicMock()
    remote = RemoteRecognitionPublisher(publisher)
    return remote, publisher


# ---------------------------------------------------------------------------
# Tests: partial result
# ---------------------------------------------------------------------------

class TestPartialResult:
    def test_partial_json_calls_on_partial_update(self) -> None:
        remote, publisher = _make_publisher()
        remote.dispatch(_result_json(status="partial"))
        publisher.publish_partial_update.assert_called_once()
        publisher.publish_finalization.assert_not_called()

    def test_partial_result_text_field(self) -> None:
        remote, publisher = _make_publisher()
        remote.dispatch(_result_json(status="partial", text="one two three"))
        result: RecognitionResult = publisher.publish_partial_update.call_args[0][0]
        assert result.text == "one two three"

    def test_partial_result_timing_fields(self) -> None:
        remote, publisher = _make_publisher()
        remote.dispatch(_result_json(status="partial", start_time=3.0, end_time=5.5))
        result: RecognitionResult = publisher.publish_partial_update.call_args[0][0]
        assert result.start_time == 3.0
        assert result.end_time == 5.5

    def test_partial_result_utterance_id(self) -> None:
        remote, publisher = _make_publisher()
        remote.dispatch(_result_json(status="partial", utterance_id=7))
        result: RecognitionResult = publisher.publish_partial_update.call_args[0][0]
        assert result.utterance_id == 7

    def test_partial_result_confidence_fields(self) -> None:
        remote, publisher = _make_publisher()
        remote.dispatch(_result_json(
            status="partial",
            confidence=0.91,
            token_confidences=[0.95, 0.87],
            audio_rms=0.05,
            confidence_variance=0.001,
        ))
        result: RecognitionResult = publisher.publish_partial_update.call_args[0][0]
        assert result.confidence == 0.91
        assert result.token_confidences == [0.95, 0.87]
        assert result.audio_rms == 0.05
        assert result.confidence_variance == 0.001

    def test_partial_result_chunk_ids(self) -> None:
        remote, publisher = _make_publisher()
        remote.dispatch(_result_json(status="partial", chunk_ids=[20, 21, 22]))
        result: RecognitionResult = publisher.publish_partial_update.call_args[0][0]
        assert result.chunk_ids == [20, 21, 22]


# ---------------------------------------------------------------------------
# Tests: final result
# ---------------------------------------------------------------------------

class TestFinalResult:
    def test_final_json_calls_on_finalization(self) -> None:
        remote, publisher = _make_publisher()
        remote.dispatch(_result_json(status="final"))
        publisher.publish_finalization.assert_called_once()
        publisher.publish_partial_update.assert_not_called()

    def test_final_result_text_field(self) -> None:
        remote, publisher = _make_publisher()
        remote.dispatch(_result_json(status="final", text="confirmed"))
        result: RecognitionResult = publisher.publish_finalization.call_args[0][0]
        assert result.text == "confirmed"

    def test_final_result_is_recognition_result_instance(self) -> None:
        remote, publisher = _make_publisher()
        remote.dispatch(_result_json(status="final"))
        result = publisher.publish_finalization.call_args[0][0]
        assert isinstance(result, RecognitionResult)


# ---------------------------------------------------------------------------
# Tests: unknown message type / status
# ---------------------------------------------------------------------------

class TestUnknownMessageType:
    def test_session_created_type_does_not_call_subscriber(self) -> None:
        remote, publisher = _make_publisher()
        remote.dispatch(json.dumps({"type": "session_created", "session_id": "s1"}))
        publisher.publish_partial_update.assert_not_called()
        publisher.publish_finalization.assert_not_called()

    def test_session_created_type_does_not_raise(self) -> None:
        remote, _ = _make_publisher()
        remote.dispatch(json.dumps({"type": "session_created", "session_id": "s1"}))

    def test_unknown_status_does_not_call_subscriber(self) -> None:
        remote, publisher = _make_publisher()
        remote.dispatch(_result_json(status="unknown_status"))
        publisher.publish_partial_update.assert_not_called()
        publisher.publish_finalization.assert_not_called()

    def test_unknown_status_does_not_raise(self) -> None:
        remote, _ = _make_publisher()
        remote.dispatch(_result_json(status="unknown_status"))

    def test_error_type_does_not_call_subscriber(self) -> None:
        remote, publisher = _make_publisher()
        error_json = json.dumps({
            "type": "error",
            "session_id": "s1",
            "error_code": "INTERNAL_ERROR",
            "message": "something broke",
            "fatal": False,
        })
        remote.dispatch(error_json)
        publisher.publish_partial_update.assert_not_called()
        publisher.publish_finalization.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: malformed input
# ---------------------------------------------------------------------------

class TestMalformedInput:
    def test_non_json_string_does_not_raise(self) -> None:
        remote, _ = _make_publisher()
        remote.dispatch("this is not json {{{")

    def test_non_json_string_does_not_call_subscriber(self) -> None:
        remote, publisher = _make_publisher()
        remote.dispatch("this is not json {{{")
        publisher.publish_partial_update.assert_not_called()
        publisher.publish_finalization.assert_not_called()

    def test_empty_string_does_not_raise(self) -> None:
        remote, _ = _make_publisher()
        remote.dispatch("")

    def test_missing_type_field_does_not_raise(self) -> None:
        remote, _ = _make_publisher()
        remote.dispatch(json.dumps({"session_id": "s1", "status": "partial"}))

    def test_missing_text_field_does_not_raise(self) -> None:
        """Missing required recognition_result fields should not propagate KeyError."""
        remote, _ = _make_publisher()
        incomplete = json.dumps({
            "type": "recognition_result",
            "session_id": "s1",
            "status": "partial",
        })
        remote.dispatch(incomplete)
