"""Tests for the server-side WebSocket wire protocol codec (src.network.codec).

Covers: decode_audio_frame validation errors, JSON server message encoding for
all types, client message decoding (control_command), unknown type handling.

Note: encode_audio_frame tests live in tests/client/tk/test_ws_codec.py.
"""

import json
import struct
import time

import numpy as np
import pytest

from src.network.types import (
    WsAudioFrame,
    WsControlCommand,
    WsDownloadProgress,
    WsError,
    WsModelInfo,
    WsModelList,
    WsModelStatus,
    WsRecognitionResult,
    WsServerState,
    WsSessionClosed,
    WsSessionCreated,
)
from src.client.tk.network.codec import encode_audio_frame
from src.network.codec import (
    SessionIdMismatchError,
    decode_audio_frame,
    decode_client_message,
    encode_server_message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(
    session_id: str = "sess-1",
    chunk_id: int = 1,
    num_samples: int = 512,
) -> WsAudioFrame:
    audio = np.zeros(num_samples, dtype=np.float32)
    return WsAudioFrame(
        session_id=session_id,
        chunk_id=chunk_id,
        timestamp=1_000.0,
        audio=audio,
    )


# ---------------------------------------------------------------------------
# decode_audio_frame — validation errors
# ---------------------------------------------------------------------------

class TestAudioFrameValidation:
    def test_rejects_truncated_header_length(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            decode_audio_frame(b"\x00\x01", expected_session_id="x")

    def test_rejects_invalid_header_json(self) -> None:
        bad_json = b"not-json"
        header_len = len(bad_json)
        raw = struct.pack("<I", header_len) + bad_json + b"\x00" * 4
        with pytest.raises(ValueError, match="header"):
            decode_audio_frame(raw, expected_session_id="x")

    def test_rejects_wrong_message_type(self) -> None:
        header = json.dumps({"type": "control_command", "session_id": "s"}).encode()
        raw = struct.pack("<I", len(header)) + header + b"\x00" * 4
        with pytest.raises(ValueError, match="type"):
            decode_audio_frame(raw, expected_session_id="s")

    def test_rejects_session_id_mismatch(self) -> None:
        frame = _frame(session_id="session-A")
        raw = encode_audio_frame(frame)
        with pytest.raises(SessionIdMismatchError, match="session_id"):
            decode_audio_frame(raw, expected_session_id="session-B")


# ---------------------------------------------------------------------------
# encode_server_message — JSON encoding for all server message types
# ---------------------------------------------------------------------------

class TestEncodeServerMessage:
    def test_encodes_session_created(self) -> None:
        msg = WsSessionCreated(
            session_id="s1",
            protocol_version="v1",
            server_time=1_000.0,
        )
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["type"] == "session_created"
        assert obj["session_id"] == "s1"
        assert obj["protocol_version"] == "v1"

    def test_encodes_recognition_result_partial(self) -> None:
        msg = WsRecognitionResult(
            session_id="s1",
            status="partial",
            text="hello",
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[1, 2],
            utterance_id=3,
            token_confidences=[0.9, 0.9],
        )
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["type"] == "recognition_result"
        assert obj["status"] == "partial"
        assert obj["text"] == "hello"
        assert obj["utterance_id"] == 3

    def test_encodes_recognition_result_final(self) -> None:
        msg = WsRecognitionResult(
            session_id="s1",
            status="final",
            text="world",
            start_time=2.0,
            end_time=3.0,
            chunk_ids=[3],
            utterance_id=4,
        )
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["status"] == "final"

    def test_encodes_session_closed(self) -> None:
        msg = WsSessionClosed(session_id="s1", reason="close_session")
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["type"] == "session_closed"
        assert obj["reason"] == "close_session"

    def test_encodes_session_closed_with_message(self) -> None:
        msg = WsSessionClosed(session_id="s1", reason="error", message="boom")
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["message"] == "boom"

    def test_encodes_error_non_fatal(self) -> None:
        msg = WsError(
            session_id="s1",
            error_code="INVALID_AUDIO_FRAME",
            message="bad payload",
            fatal=False,
        )
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["type"] == "error"
        assert obj["error_code"] == "INVALID_AUDIO_FRAME"
        assert obj["fatal"] is False

    def test_encodes_error_fatal(self) -> None:
        msg = WsError(
            session_id="s1",
            error_code="INTERNAL_ERROR",
            message="crash",
            fatal=True,
        )
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["fatal"] is True

    def test_encodes_error_with_request_id(self) -> None:
        msg = WsError(
            session_id="s1",
            error_code="DOWNLOAD_IN_PROGRESS",
            message="busy",
            fatal=False,
            request_id="req-1",
        )
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["request_id"] == "req-1"

    def test_encodes_model_list(self) -> None:
        msg = WsModelList(
            request_id="req-2",
            models=[
                WsModelInfo(
                    name="parakeet",
                    display_name="Parakeet TDT 0.6B v3",
                    size_description="1.25 GB",
                    status="downloaded",
                )
            ],
        )
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["type"] == "model_list"
        assert obj["request_id"] == "req-2"
        assert obj["models"][0]["name"] == "parakeet"

    def test_encodes_model_status(self) -> None:
        msg = WsModelStatus(status="downloading", request_id="req-3")
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["type"] == "model_status"
        assert obj["status"] == "downloading"
        assert obj["request_id"] == "req-3"

    def test_encodes_server_state(self) -> None:
        msg = WsServerState(state="waiting_for_model")
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["type"] == "server_state"
        assert obj["state"] == "waiting_for_model"

    def test_encodes_download_progress(self) -> None:
        msg = WsDownloadProgress(
            model_name="parakeet",
            status="downloading",
            progress=0.25,
            downloaded_bytes=256,
            total_bytes=1024,
        )
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["type"] == "download_progress"
        assert obj["model_name"] == "parakeet"
        assert obj["status"] == "downloading"
        assert obj["progress"] == 0.25
        assert obj["downloaded_bytes"] == 256
        assert obj["total_bytes"] == 1024

    def test_encodes_download_progress_error(self) -> None:
        msg = WsDownloadProgress(
            model_name="parakeet",
            status="error",
            error_message="disk full",
        )
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["type"] == "download_progress"
        assert obj["status"] == "error"
        assert obj["error_message"] == "disk full"

    def test_result_includes_token_confidences(self) -> None:
        """token_confidences should always be present in encoded recognition_result."""
        msg = WsRecognitionResult(
            session_id="s1",
            status="final",
            text="hi",
            start_time=0.0,
            end_time=1.0,
            chunk_ids=[1],
            utterance_id=1,
        )
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert "token_confidences" in obj


# ---------------------------------------------------------------------------
# decode_client_message — JSON decoding for client-sent messages
# ---------------------------------------------------------------------------

class TestDecodeClientMessage:
    def test_decodes_close_session_command(self) -> None:
        payload = json.dumps({
            "type": "control_command",
            "session_id": "s1",
            "command": "close_session",
            "timestamp": 1_000.0,
        })
        msg = decode_client_message(payload)
        assert isinstance(msg, WsControlCommand)
        assert msg.command == "close_session"
        assert msg.session_id == "s1"

    def test_decodes_list_models_command_with_request_id(self) -> None:
        payload = json.dumps({
            "type": "control_command",
            "session_id": "s2",
            "command": "list_models",
            "request_id": "req-42",
            "timestamp": 2_000.0,
        })
        msg = decode_client_message(payload)
        assert isinstance(msg, WsControlCommand)
        assert msg.command == "list_models"
        assert msg.request_id == "req-42"

    def test_decodes_download_model_command_with_model_name(self) -> None:
        payload = json.dumps({
            "type": "control_command",
            "session_id": "s3",
            "command": "download_model",
            "model_name": "parakeet",
            "request_id": "req-43",
            "timestamp": 3_000.0,
        })
        msg = decode_client_message(payload)
        assert isinstance(msg, WsControlCommand)
        assert msg.command == "download_model"
        assert msg.model_name == "parakeet"

    def test_unknown_type_raises_value_error(self) -> None:
        payload = json.dumps({"type": "ping", "session_id": "s1"})
        with pytest.raises(ValueError, match="unknown.*type"):
            decode_client_message(payload)

    def test_missing_type_field_raises_value_error(self) -> None:
        payload = json.dumps({"session_id": "s1", "command": "close_session"})
        with pytest.raises(ValueError, match="type"):
            decode_client_message(payload)

    def test_invalid_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="JSON"):
            decode_client_message("not-json{{{")

    def test_invalid_command_value_raises_value_error(self) -> None:
        payload = json.dumps({
            "type": "control_command",
            "session_id": "s1",
            "command": "explode",
            "timestamp": 1.0,
        })
        with pytest.raises(ValueError, match="command"):
            decode_client_message(payload)

    def test_shutdown_command_is_rejected(self) -> None:
        payload = json.dumps({
            "type": "control_command",
            "session_id": "s1",
            "command": "shutdown",
            "timestamp": 1.0,
        })
        with pytest.raises(ValueError, match="command"):
            decode_client_message(payload)

    def test_download_model_requires_model_name(self) -> None:
        payload = json.dumps({
            "type": "control_command",
            "session_id": "s1",
            "command": "download_model",
            "timestamp": 1.0,
        })
        with pytest.raises(ValueError, match="model_name"):
            decode_client_message(payload)
