"""Encode and decode WebSocket wire protocol frames for the STT server (v1).

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


# ---------------------------------------------------------------------------
# Binary frame: audio_chunk (receive)
# ---------------------------------------------------------------------------

def decode_audio_frame(raw: bytes, expected_session_id: str) -> WsAudioFrame:
    """Decode a binary WebSocket frame into a WsAudioFrame.

    Algorithm:
        1. Read 4-byte header_len prefix.
        2. Parse JSON header from [4 .. 4+header_len].
        3. Validate type and session_id.
        4. Reconstruct float32 numpy array from remaining payload bytes.

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

    audio = np.frombuffer(raw[header_end:], dtype=np.float32).copy()

    return WsAudioFrame(
        session_id=header["session_id"],
        chunk_id=header["chunk_id"],
        timestamp=header["timestamp"],
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
            "token_confidences": msg.token_confidences,
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
        session_id = obj.get("session_id")
        if session_id is None:
            raise ValueError("control_command missing 'session_id' field")
        timestamp = obj.get("timestamp")
        if timestamp is None:
            raise ValueError("control_command missing 'timestamp' field")
        return WsControlCommand(
            session_id=session_id,
            command="shutdown",
            timestamp=timestamp,
            request_id=obj.get("request_id"),
        )

    raise ValueError(f"unknown message type: {msg_type!r}")
