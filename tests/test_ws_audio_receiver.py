"""Tests for WsAudioReceiver (Phase 3).

Strategy: AsyncMock for websocket, asyncio.run() to drive the coroutine.
The chunk_queue is a real queue.Queue (thread-safe, same type SoundPreProcessor reads).
"""

import asyncio
import json
import queue
import struct
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.network.codec import encode_audio_frame
from src.network.types import WsAudioFrame
from src.server.WsAudioReceiver import receive_audio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        sample_rate=16000,
        num_samples=num_samples,
        dtype="float32",
        channels=1,
        audio=audio,
    )
    return encode_audio_frame(frame)


def _shutdown_command(session_id: str = _SESSION_ID) -> str:
    return json.dumps({
        "type": "control_command",
        "session_id": session_id,
        "command": "shutdown",
        "timestamp": 1000.0,
    })


class _WebSocketSequence:
    """Fake websocket that serves a fixed sequence of messages then raises StopAsyncIteration."""

    def __init__(self, messages: list) -> None:
        self._messages = iter(messages)
        self.sent: list[str] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._messages)
        except StopIteration:
            raise StopAsyncIteration

    async def send(self, message: str) -> None:
        self.sent.append(message)


# ---------------------------------------------------------------------------
# Tests: binary audio frame → chunk_queue
# ---------------------------------------------------------------------------

class TestAudioFrameToQueue:
    def test_valid_frame_puts_chunk_dict_to_queue(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()
        ws = _WebSocketSequence([_make_audio_frame()])

        asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))

        assert not chunk_queue.empty()
        chunk = chunk_queue.get_nowait()
        assert "audio" in chunk
        assert "timestamp" in chunk

    def test_chunk_dict_audio_is_float32_array(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()
        ws = _WebSocketSequence([_make_audio_frame(num_samples=256)])

        asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))

        chunk = chunk_queue.get_nowait()
        assert chunk["audio"].dtype == np.float32
        assert len(chunk["audio"]) == 256

    def test_chunk_dict_matches_audiosource_format(self) -> None:
        """Chunk dict must have 'audio' and 'timestamp' keys — same as AudioSource emits."""
        chunk_queue: queue.Queue = queue.Queue()
        ws = _WebSocketSequence([_make_audio_frame()])

        asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))

        chunk = chunk_queue.get_nowait()
        assert set(chunk.keys()) == {"audio", "timestamp"}

    def test_multiple_frames_all_queued(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()
        frames = [_make_audio_frame(chunk_id=i) for i in range(3)]
        ws = _WebSocketSequence(frames)

        asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))

        assert chunk_queue.qsize() == 3

    def test_timestamp_is_float(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()
        ws = _WebSocketSequence([_make_audio_frame()])

        asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))

        chunk = chunk_queue.get_nowait()
        assert isinstance(chunk["timestamp"], float)


# ---------------------------------------------------------------------------
# Tests: decode error sends WsError and continues
# ---------------------------------------------------------------------------

class TestDecodeErrorHandling:
    def test_malformed_frame_sends_error_response(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()
        bad_frame = b"\x00\x00\x00\x05not-json-at-all-padding"
        ws = _WebSocketSequence([bad_frame])

        asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))

        assert len(ws.sent) == 1
        obj = json.loads(ws.sent[0])
        assert obj["type"] == "error"
        assert obj["error_code"] == "INVALID_AUDIO_FRAME"

    def test_malformed_frame_does_not_stop_processing(self) -> None:
        """After a bad frame the receiver continues and processes the next valid frame."""
        chunk_queue: queue.Queue = queue.Queue()
        bad_frame = b"\x00\x00\x00\x05not-json-at-all-padding"
        good_frame = _make_audio_frame(chunk_id=2)
        ws = _WebSocketSequence([bad_frame, good_frame])

        asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))

        assert chunk_queue.qsize() == 1

    def test_error_message_is_non_fatal(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()
        bad_frame = b"\x00\x00\x00\x05not-json-at-all-padding"
        ws = _WebSocketSequence([bad_frame])

        asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))

        obj = json.loads(ws.sent[0])
        assert obj["fatal"] is False


# ---------------------------------------------------------------------------
# Tests: shutdown command text frame triggers clean exit
# ---------------------------------------------------------------------------

class TestShutdownCommand:
    def test_shutdown_command_stops_receive_loop(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()
        # Frame after shutdown should not be processed
        ws = _WebSocketSequence([_shutdown_command(), _make_audio_frame(chunk_id=99)])

        asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))

        assert chunk_queue.empty()

    def test_shutdown_command_returns_shutdown_reason(self) -> None:
        chunk_queue: queue.Queue = queue.Queue()
        ws = _WebSocketSequence([_shutdown_command()])

        reason = asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))

        assert reason == "shutdown"

    def test_connection_closed_returns_connection_lost_reason(self) -> None:
        from websockets.exceptions import ConnectionClosed
        from websockets.frames import Close

        chunk_queue: queue.Queue = queue.Queue()

        async def _raising_iter(ws):
            raise ConnectionClosed(
                rcvd=Close(code=1001, reason="going away"),
                sent=None,
            )

        class _ClosingWs:
            sent: list[str] = []
            def __aiter__(self): return self
            async def __anext__(self):
                raise ConnectionClosed(
                    rcvd=Close(code=1001, reason="going away"),
                    sent=None,
                )
            async def send(self, msg: str) -> None:
                self.sent.append(msg)

        ws = _ClosingWs()
        reason = asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))
        assert reason == "connection_lost"


# ---------------------------------------------------------------------------
# Tests: full chunk_queue drops frame and sends BACKPRESSURE_DROP error
# ---------------------------------------------------------------------------

class TestBackpressure:
    def test_full_queue_drops_frame_and_sends_error(self) -> None:
        chunk_queue: queue.Queue = queue.Queue(maxsize=1)
        chunk_queue.put_nowait({"audio": np.zeros(1, dtype=np.float32), "timestamp": 0.0})

        frame1 = _make_audio_frame(chunk_id=10)
        ws = _WebSocketSequence([frame1])

        asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))

        assert len(ws.sent) == 1
        obj = json.loads(ws.sent[0])
        assert obj["error_code"] == "BACKPRESSURE_DROP"

    def test_full_queue_does_not_block_receiver(self) -> None:
        """The receive coroutine must return even when the queue is full."""
        import time
        chunk_queue: queue.Queue = queue.Queue(maxsize=1)
        chunk_queue.put_nowait({"audio": np.zeros(1, dtype=np.float32), "timestamp": 0.0})
        ws = _WebSocketSequence([_make_audio_frame(chunk_id=5)])

        start = time.monotonic()
        asyncio.run(receive_audio(ws, session_id=_SESSION_ID, chunk_queue=chunk_queue))
        assert time.monotonic() - start < 2.0
