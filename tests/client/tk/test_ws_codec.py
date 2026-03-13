"""Tests for the client-side WebSocket codec (src.client.network.codec).

Covers: binary audio frame encoding — wire format correctness, payload length,
header prefix integrity. Round-trip verification uses the server-side
decode_audio_frame to confirm the on-wire format is compatible.
"""

import json
import struct

import numpy as np

from src.client.tk.network.codec import encode_audio_frame
from src.network.codec import decode_audio_frame
from src.network.types import WsAudioFrame


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


class TestEncodeAudioFrame:
    def test_roundtrip_preserves_audio(self) -> None:
        audio = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
        frame = WsAudioFrame(
            session_id="abc",
            chunk_id=7,
            timestamp=42.5,
            audio=audio,
        )
        raw = encode_audio_frame(frame)
        decoded = decode_audio_frame(raw, expected_session_id="abc")
        np.testing.assert_array_equal(decoded.audio, audio)

    def test_roundtrip_preserves_metadata(self) -> None:
        frame = _frame(session_id="s99", chunk_id=42)
        raw = encode_audio_frame(frame)
        decoded = decode_audio_frame(raw, expected_session_id="s99")
        assert decoded.session_id == "s99"
        assert decoded.chunk_id == 42

    def test_header_length_prefix_is_correct(self) -> None:
        frame = _frame()
        raw = encode_audio_frame(frame)
        header_len = struct.unpack_from("<I", raw, 0)[0]
        header_bytes = raw[4 : 4 + header_len]
        header = json.loads(header_bytes.decode("utf-8"))
        assert header["type"] == "audio_chunk"
        assert len(raw) == 4 + header_len + len(frame.audio) * 4

    def test_payload_length_matches_num_samples(self) -> None:
        frame = _frame(num_samples=256)
        raw = encode_audio_frame(frame)
        header_len = struct.unpack_from("<I", raw, 0)[0]
        payload = raw[4 + header_len :]
        assert len(payload) == 256 * 4
