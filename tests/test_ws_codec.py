"""Tests for WebSocket wire protocol codec (Phase 1).

Covers: binary audio frame roundtrip, header integrity, validation errors,
JSON server message encode/decode for all types, unknown type handling.
"""

import json
import struct
import time

import numpy as np
import pytest

from src.network.types import (
    WsAudioFrame,
    WsControlCommand,
    WsError,
    WsRecognitionResult,
    WsSessionClosed,
    WsSessionCreated,
)
from src.network.codec import (
    decode_audio_frame,
    decode_client_message,
    encode_audio_frame,
    encode_server_message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(
    session_id: str = "sess-1",
    chunk_id: int = 1,
    num_samples: int = 512,
    sample_rate: int = 16000,
) -> WsAudioFrame:
    audio = np.zeros(num_samples, dtype=np.float32)
    return WsAudioFrame(
        session_id=session_id,
        chunk_id=chunk_id,
        timestamp=1_000.0,
        sample_rate=sample_rate,
        num_samples=num_samples,
        dtype="float32",
        channels=1,
        audio=audio,
    )


def _raw_binary(frame: WsAudioFrame) -> bytes:
    """Encode a frame to raw bytes using the codec."""
    return encode_audio_frame(frame)


# ---------------------------------------------------------------------------
# encode_audio_frame / decode_audio_frame — binary roundtrip
# ---------------------------------------------------------------------------

class TestAudioFrameRoundtrip:
    def test_roundtrip_preserves_audio(self) -> None:
        audio = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
        frame = WsAudioFrame(
            session_id="abc",
            chunk_id=7,
            timestamp=42.5,
            sample_rate=16000,
            num_samples=512,
            dtype="float32",
            channels=1,
            audio=audio,
        )
        raw = encode_audio_frame(frame)
        decoded = decode_audio_frame(raw, expected_session_id="abc")
        np.testing.assert_array_equal(decoded.audio, audio)

    def test_roundtrip_preserves_metadata(self) -> None:
        frame = _frame(session_id="s99", chunk_id=42)
        decoded = decode_audio_frame(_raw_binary(frame), expected_session_id="s99")
        assert decoded.session_id == "s99"
        assert decoded.chunk_id == 42
        assert decoded.sample_rate == 16000
        assert decoded.num_samples == 512
        assert decoded.dtype == "float32"
        assert decoded.channels == 1

    def test_header_length_prefix_is_correct(self) -> None:
        frame = _frame()
        raw = encode_audio_frame(frame)
        header_len = struct.unpack_from("<I", raw, 0)[0]
        header_bytes = raw[4 : 4 + header_len]
        header = json.loads(header_bytes.decode("utf-8"))
        assert header["type"] == "audio_chunk"
        assert len(raw) == 4 + header_len + frame.num_samples * 4

    def test_payload_length_matches_num_samples(self) -> None:
        frame = _frame(num_samples=256)
        raw = encode_audio_frame(frame)
        header_len = struct.unpack_from("<I", raw, 0)[0]
        payload = raw[4 + header_len :]
        assert len(payload) == 256 * 4


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
        with pytest.raises(ValueError, match="session_id"):
            decode_audio_frame(raw, expected_session_id="session-B")

    def test_rejects_wrong_sample_rate(self) -> None:
        frame = _frame(sample_rate=8000)
        raw = encode_audio_frame(frame)
        with pytest.raises(ValueError, match="sample_rate"):
            decode_audio_frame(raw, expected_session_id=frame.session_id)

    def test_rejects_num_samples_payload_mismatch(self) -> None:
        frame = _frame(num_samples=512)
        raw = encode_audio_frame(frame)
        # Corrupt by truncating payload by 4 bytes (one float32 missing)
        truncated = raw[:-4]
        with pytest.raises(ValueError, match="payload"):
            decode_audio_frame(truncated, expected_session_id=frame.session_id)

    def test_rejects_wrong_dtype(self) -> None:
        frame = _frame()
        raw = encode_audio_frame(frame)
        header_len = struct.unpack_from("<I", raw, 0)[0]
        header = json.loads(raw[4 : 4 + header_len])
        header["dtype"] = "int16"
        bad_header = json.dumps(header).encode()
        rebuilt = struct.pack("<I", len(bad_header)) + bad_header + raw[4 + header_len :]
        with pytest.raises(ValueError, match="dtype"):
            decode_audio_frame(rebuilt, expected_session_id=frame.session_id)

    def test_rejects_wrong_channels(self) -> None:
        frame = _frame()
        raw = encode_audio_frame(frame)
        header_len = struct.unpack_from("<I", raw, 0)[0]
        header = json.loads(raw[4 : 4 + header_len])
        header["channels"] = 2
        bad_header = json.dumps(header).encode()
        rebuilt = struct.pack("<I", len(bad_header)) + bad_header + raw[4 + header_len :]
        with pytest.raises(ValueError, match="channels"):
            decode_audio_frame(rebuilt, expected_session_id=frame.session_id)


# ---------------------------------------------------------------------------
# encode_server_message — JSON encoding for all server message types
# ---------------------------------------------------------------------------

class TestEncodeServerMessage:
    def test_encodes_session_created(self) -> None:
        msg = WsSessionCreated(
            session_id="s1",
            protocol_version="v1",
            server_time=1_000.0,
            server_config={
                "sample_rate": 16000,
                "chunk_duration_sec": 0.032,
                "audio_dtype": "float32",
                "channels": 1,
            },
        )
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["type"] == "session_created"
        assert obj["session_id"] == "s1"
        assert obj["protocol_version"] == "v1"
        assert obj["server_config"]["sample_rate"] == 16000

    def test_encodes_recognition_result_partial(self) -> None:
        msg = WsRecognitionResult(
            session_id="s1",
            status="partial",
            text="hello",
            start_time=1.0,
            end_time=2.0,
            chunk_ids=[1, 2],
            utterance_id=3,
            confidence=0.9,
            token_confidences=[0.9, 0.9],
            audio_rms=0.05,
            confidence_variance=0.001,
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
        msg = WsSessionClosed(session_id="s1", reason="shutdown")
        raw = encode_server_message(msg)
        obj = json.loads(raw)
        assert obj["type"] == "session_closed"
        assert obj["reason"] == "shutdown"

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

    def test_result_omits_optional_confidence_fields_when_zero(self) -> None:
        """Optional confidence fields should be present in all cases (codec always encodes them)."""
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
        assert "confidence" in obj
        assert "token_confidences" in obj


# ---------------------------------------------------------------------------
# decode_client_message — JSON decoding for client-sent messages
# ---------------------------------------------------------------------------

class TestDecodeClientMessage:
    def test_decodes_shutdown_command(self) -> None:
        payload = json.dumps({
            "type": "control_command",
            "session_id": "s1",
            "command": "shutdown",
            "timestamp": 1_000.0,
        })
        msg = decode_client_message(payload)
        assert isinstance(msg, WsControlCommand)
        assert msg.command == "shutdown"
        assert msg.session_id == "s1"

    def test_decodes_shutdown_command_with_request_id(self) -> None:
        payload = json.dumps({
            "type": "control_command",
            "session_id": "s2",
            "command": "shutdown",
            "request_id": "req-42",
            "timestamp": 2_000.0,
        })
        msg = decode_client_message(payload)
        assert isinstance(msg, WsControlCommand)
        assert msg.request_id == "req-42"

    def test_unknown_type_raises_value_error(self) -> None:
        payload = json.dumps({"type": "ping", "session_id": "s1"})
        with pytest.raises(ValueError, match="unknown.*type"):
            decode_client_message(payload)

    def test_missing_type_field_raises_value_error(self) -> None:
        payload = json.dumps({"session_id": "s1", "command": "shutdown"})
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
