"""Tests for the audio-only WebSocket receiver helper."""

import json
import queue

import numpy as np

from src.client.tk.network.codec import encode_audio_frame
from src.network.codec import SessionIdMismatchError
from src.network.types import WsAudioFrame
from src.server.WsAudioReceiver import handle_audio_frame

_SESSION_ID = "test-session"


def _make_audio_frame(
    session_id: str = _SESSION_ID,
    chunk_id: int = 1,
    num_samples: int = 512,
) -> bytes:
    """Build a valid encoded binary audio frame."""
    audio = np.zeros(num_samples, dtype=np.float32)
    frame = WsAudioFrame(
        session_id=session_id,
        chunk_id=chunk_id,
        timestamp=1000.0,
        audio=audio,
    )
    return encode_audio_frame(frame)


class TestAudioFrameHandling:
    """Audio frames are decoded, validated, and enqueued."""

    def test_valid_frame_puts_chunk_dict_to_queue(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()

        error = handle_audio_frame(
            raw=_make_audio_frame(),
            session_id=_SESSION_ID,
            chunk_queue=chunk_queue,
            server_state="running",
        )

        assert error is None
        chunk = chunk_queue.get_nowait()
        assert set(chunk.keys()) == {"audio", "timestamp"}
        assert chunk["audio"].dtype == np.float32
        assert isinstance(chunk["timestamp"], float)

    def test_malformed_frame_returns_invalid_audio_error(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()

        error = handle_audio_frame(
            raw=b"\x00\x00\x00\x05not-json-at-all-padding",
            session_id=_SESSION_ID,
            chunk_queue=chunk_queue,
            server_state="running",
        )

        assert error is not None
        assert error.error_code == "INVALID_AUDIO_FRAME"
        assert error.fatal is False

    def test_full_queue_returns_backpressure_drop_error(self) -> None:
        chunk_queue: queue.Queue = queue.Queue(maxsize=1)
        chunk_queue.put_nowait({"audio": np.zeros(1, dtype=np.float32), "timestamp": 0.0})

        error = handle_audio_frame(
            raw=_make_audio_frame(chunk_id=10),
            session_id=_SESSION_ID,
            chunk_queue=chunk_queue,
            server_state="running",
        )

        assert error is not None
        assert error.error_code == "BACKPRESSURE_DROP"

    def test_session_id_mismatch_returns_session_id_mismatch_error(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()

        error = handle_audio_frame(
            raw=_make_audio_frame(session_id="wrong-session"),
            session_id=_SESSION_ID,
            chunk_queue=chunk_queue,
            server_state="running",
        )

        assert error is not None
        assert error.error_code == "SESSION_ID_MISMATCH"
        assert chunk_queue.empty()

    def test_session_id_mismatch_class_maps_to_session_id_mismatch_error_code(
        self,
        monkeypatch,
    ) -> None:
        chunk_queue: queue.Queue = queue.Queue()

        def _raise_mismatch(raw: bytes, expected_session_id: str) -> WsAudioFrame:
            raise SessionIdMismatchError("mismatch details changed")

        monkeypatch.setattr("src.server.WsAudioReceiver.decode_audio_frame", _raise_mismatch)

        error = handle_audio_frame(
            raw=b"ignored",
            session_id=_SESSION_ID,
            chunk_queue=chunk_queue,
            server_state="running",
        )

        assert error is not None
        assert error.error_code == "SESSION_ID_MISMATCH"
        assert error.message == "mismatch details changed"

    def test_audio_when_model_not_ready_returns_model_not_ready_error(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()

        error = handle_audio_frame(
            raw=_make_audio_frame(),
            session_id=_SESSION_ID,
            chunk_queue=chunk_queue,
            server_state="waiting_for_model",
        )

        assert error is not None
        assert error.error_code == "MODEL_NOT_READY"
        assert chunk_queue.empty()

    def test_model_not_ready_error_encodes_requestless_server_error_shape(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()

        error = handle_audio_frame(
            raw=_make_audio_frame(),
            session_id=_SESSION_ID,
            chunk_queue=chunk_queue,
            server_state="starting",
        )

        assert error is not None
        encoded = json.loads(
            json.dumps(
                {
                    "type": "error",
                    "session_id": error.session_id,
                    "error_code": error.error_code,
                    "message": error.message,
                    "fatal": error.fatal,
                }
            )
        )
        assert encoded["error_code"] == "MODEL_NOT_READY"
