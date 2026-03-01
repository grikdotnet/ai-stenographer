"""Encode and decode WebSocket wire protocol frames (v1).

Binary frame layout (audio_chunk):
  bytes [0..3]          : uint32 little-endian header_len
  bytes [4..4+header_len): UTF-8 JSON header
  bytes [4+header_len..] : raw float32 PCM payload

All other messages are UTF-8 JSON text frames.
"""

import json
import struct
from typing import Union

import numpy as np

from src.network.types import (
    ClientTextMessage,
    ServerMessage,
    WsAudioFrame,
    WsControlCommand,
    WsError,
    WsRecognitionResult,
    WsSessionClosed,
    WsSessionCreated,
)

_HEADER_PREFIX_LEN = 4  # bytes for the uint32 header_len field
_EXPECTED_SAMPLE_RATE = 16000
_EXPECTED_DTYPE = "float32"
_EXPECTED_CHANNELS = 1


# ---------------------------------------------------------------------------
# Binary frame: audio_chunk
# ---------------------------------------------------------------------------

def encode_audio_frame(frame: WsAudioFrame) -> bytes:
    """Encode a WsAudioFrame to a binary WebSocket frame.

    Args:
        frame: Audio frame to encode.

    Returns:
        Binary bytes: 4-byte header_len prefix + JSON header + PCM payload.
    """
    header = {
        "type": "audio_chunk",
        "session_id": frame.session_id,
        "chunk_id": frame.chunk_id,
        "timestamp": frame.timestamp,
        "sample_rate": frame.sample_rate,
        "num_samples": frame.num_samples,
        "dtype": frame.dtype,
        "channels": frame.channels,
    }
    header_bytes = json.dumps(header).encode("utf-8")
    payload = frame.audio.astype(np.float32).tobytes()
    return struct.pack("<I", len(header_bytes)) + header_bytes + payload


def decode_audio_frame(raw: bytes, expected_session_id: str) -> WsAudioFrame:
    """Decode a binary WebSocket frame into a WsAudioFrame.

    Algorithm:
        1. Read 4-byte header_len prefix.
        2. Parse JSON header from [4 .. 4+header_len].
        3. Validate type, session_id, sample_rate, dtype, channels.
        4. Reconstruct float32 numpy array from remaining payload bytes.
        5. Verify payload length matches num_samples.

    Args:
        raw: Raw binary frame bytes.
        expected_session_id: Session ID the frame must carry.

    Returns:
        Decoded WsAudioFrame.

    Raises:
        ValueError: On any validation failure with a descriptive message.
    """
    if len(raw) < _HEADER_PREFIX_LEN:
        raise ValueError(f"Frame too short: {len(raw)} bytes, need at least {_HEADER_PREFIX_LEN}")

    header_len = struct.unpack_from("<I", raw, 0)[0]
    header_end = _HEADER_PREFIX_LEN + header_len

    if len(raw) < header_end:
        raise ValueError(f"Frame too short to contain header: need {header_end}, got {len(raw)}")

    try:
        header = json.loads(raw[_HEADER_PREFIX_LEN:header_end].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid header JSON: {exc}") from exc

    if header.get("type") != "audio_chunk":
        raise ValueError(f"Expected type 'audio_chunk', got {header.get('type')!r}")

    if header.get("session_id") != expected_session_id:
        raise ValueError(
            f"session_id mismatch: expected {expected_session_id!r}, got {header.get('session_id')!r}"
        )

    if header.get("sample_rate") != _EXPECTED_SAMPLE_RATE:
        raise ValueError(f"Invalid sample_rate {header.get('sample_rate')}: must be {_EXPECTED_SAMPLE_RATE}")

    if header.get("dtype") != _EXPECTED_DTYPE:
        raise ValueError(f"Invalid dtype {header.get('dtype')!r}: must be '{_EXPECTED_DTYPE}'")

    if header.get("channels") != _EXPECTED_CHANNELS:
        raise ValueError(f"Invalid channels {header.get('channels')}: must be {_EXPECTED_CHANNELS}")

    num_samples: int = header["num_samples"]
    payload = raw[header_end:]
    expected_payload_len = num_samples * 4

    if len(payload) != expected_payload_len:
        raise ValueError(
            f"payload length {len(payload)} does not match num_samples={num_samples} "
            f"(expected {expected_payload_len} bytes)"
        )

    audio = np.frombuffer(payload, dtype=np.float32).copy()

    return WsAudioFrame(
        session_id=header["session_id"],
        chunk_id=header["chunk_id"],
        timestamp=header["timestamp"],
        sample_rate=header["sample_rate"],
        num_samples=num_samples,
        dtype=header["dtype"],
        channels=header["channels"],
        audio=audio,
    )


# ---------------------------------------------------------------------------
# JSON text frames: server → client
# ---------------------------------------------------------------------------

def encode_server_message(msg: ServerMessage) -> str:
    """Encode a server-side message dataclass to a UTF-8 JSON string.

    Args:
        msg: One of WsSessionCreated, WsRecognitionResult, WsSessionClosed, WsError.

    Returns:
        JSON string suitable for sending as a WebSocket text frame.

    Raises:
        TypeError: If msg is not a recognised server message type.
    """
    if isinstance(msg, WsSessionCreated):
        obj = {
            "type": "session_created",
            "session_id": msg.session_id,
            "protocol_version": msg.protocol_version,
            "server_time": msg.server_time,
            "server_config": msg.server_config,
        }
    elif isinstance(msg, WsRecognitionResult):
        obj = {
            "type": "recognition_result",
            "session_id": msg.session_id,
            "status": msg.status,
            "text": msg.text,
            "start_time": msg.start_time,
            "end_time": msg.end_time,
            "chunk_ids": msg.chunk_ids,
            "utterance_id": msg.utterance_id,
            "confidence": msg.confidence,
            "token_confidences": msg.token_confidences,
            "audio_rms": msg.audio_rms,
            "confidence_variance": msg.confidence_variance,
        }
    elif isinstance(msg, WsSessionClosed):
        obj: dict = {
            "type": "session_closed",
            "session_id": msg.session_id,
            "reason": msg.reason,
        }
        if msg.message is not None:
            obj["message"] = msg.message
    elif isinstance(msg, WsError):
        obj = {
            "type": "error",
            "session_id": msg.session_id,
            "error_code": msg.error_code,
            "message": msg.message,
            "fatal": msg.fatal,
        }
    else:
        raise TypeError(f"Unknown server message type: {type(msg)}")

    return json.dumps(obj)


# ---------------------------------------------------------------------------
# JSON text frames: client → server
# ---------------------------------------------------------------------------

def decode_client_message(text: str) -> ClientTextMessage:
    """Decode a UTF-8 JSON text frame from the client into a typed dataclass.

    Args:
        text: Raw JSON string from a WebSocket text frame.

    Returns:
        WsControlCommand (only client text message type in v1).

    Raises:
        ValueError: On invalid JSON, missing/unknown type, or invalid field values.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in client message: {exc}") from exc

    msg_type = obj.get("type")
    if msg_type is None:
        raise ValueError("Client message missing 'type' field")

    if msg_type == "control_command":
        command = obj.get("command")
        if command != "shutdown":
            raise ValueError(f"Invalid command value: {command!r} (must be 'shutdown')")
        return WsControlCommand(
            session_id=obj["session_id"],
            command="shutdown",
            timestamp=obj["timestamp"],
            request_id=obj.get("request_id"),
        )

    raise ValueError(f"unknown message type: {msg_type!r}")
