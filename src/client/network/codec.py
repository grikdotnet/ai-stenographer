"""Encode WebSocket wire protocol frames sent by the STT client (v1).

Binary frame layout (audio_chunk):
  bytes [0..3]          : uint32 little-endian header_len
  bytes [4..4+header_len): UTF-8 JSON header
  bytes [4+header_len..] : raw float32 PCM payload
"""

import json
import struct

import numpy as np

from src.network.types import WsAudioFrame

_HEADER_PREFIX_LEN = 4  # bytes for the uint32 header_len field


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
    }
    header_bytes = json.dumps(header).encode("utf-8")
    payload = frame.audio.astype(np.float32).tobytes()
    return struct.pack("<I", len(header_bytes)) + header_bytes + payload
