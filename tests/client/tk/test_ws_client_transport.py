"""Tests for WsClientTransport (Phase 6).

Strategy: real asyncio event loop on a background daemon thread (same pattern as
test_ws_result_sender.py).  The websocket is replaced by _FakeWebSocket which
exposes an async-iterable incoming message queue and an outgoing sent queue.
send_audio_chunk() is called from the test thread (sync boundary).
"""

import asyncio
import json
import queue
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.client.tk.RemoteRecognitionPublisher import RemoteRecognitionPublisher
from src.client.tk.WsClientTransport import WsClientTransport
from src.ApplicationState import ApplicationState
from src.network.codec import decode_audio_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION_ID = "test-session"
_SERVER_URL = "ws://127.0.0.1:9999"


def _make_app_state() -> ApplicationState:
    state = ApplicationState(config={})
    state.set_state("running")
    return state


def _make_result_json(status: str = "partial", text: str = "hello") -> str:
    return json.dumps({
        "type": "recognition_result",
        "session_id": _SESSION_ID,
        "status": status,
        "text": text,
        "start_time": 0.0,
        "end_time": 1.0,
        "chunk_ids": [1],
        "utterance_id": 1,
        "confidence": 0.9,
        "token_confidences": [0.9],
        "audio_rms": 0.05,
        "confidence_variance": 0.001,
    })


def _make_audio_chunk(num_samples: int = 512) -> dict:
    return {
        "audio": np.zeros(num_samples, dtype=np.float32),
        "timestamp": 1.5,
    }


class _FakeWebSocket:
    """Fake websocket: serves messages fed from test thread; captures outgoing frames.

    Uses a threading.Queue internally so put_message() is safe to call from
    the test (sync) thread while __anext__ runs in the asyncio event loop thread.
    """

    def __init__(self) -> None:
        self._incoming: "queue.Queue[object]" = queue.Queue()
        self.sent: asyncio.Queue = asyncio.Queue()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        deadline = asyncio.get_event_loop().time() + 0.8
        while asyncio.get_event_loop().time() < deadline:
            try:
                return self._incoming.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
        raise StopAsyncIteration

    async def send(self, message) -> None:
        if self.closed:
            raise ConnectionError("closed")
        await self.sent.put(message)

    async def close(self) -> None:
        self.closed = True

    def put_message(self, msg) -> None:
        """Thread-safe: enqueue a message for the receive loop to pick up."""
        self._incoming.put_nowait(msg)


def _start_transport(
    ws: _FakeWebSocket,
    app_state: ApplicationState | None = None,
    subscriber: MagicMock | None = None,
) -> tuple[WsClientTransport, asyncio.AbstractEventLoop, threading.Thread]:
    """Start a WsClientTransport on a daemon asyncio loop thread, injecting a fake websocket."""
    if app_state is None:
        app_state = _make_app_state()
    if subscriber is None:
        subscriber = MagicMock()

    remote_publisher = RemoteRecognitionPublisher(subscriber)
    ready = threading.Event()
    loop_holder: list[asyncio.AbstractEventLoop] = []

    def run_loop() -> None:
        loop = asyncio.new_event_loop()
        loop_holder.append(loop)
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    ready.wait(timeout=2.0)

    loop = loop_holder[0]
    transport = WsClientTransport(
        server_url=_SERVER_URL,
        session_id=_SESSION_ID,
        app_state=app_state,
        publisher=remote_publisher,
        loop=loop,
    )
    asyncio.run_coroutine_threadsafe(transport.start(ws), loop).result(timeout=2.0)
    return transport, loop, t


def _stop_transport(transport: WsClientTransport, loop: asyncio.AbstractEventLoop) -> None:
    asyncio.run_coroutine_threadsafe(transport.stop(), loop).result(timeout=5.0)


# ---------------------------------------------------------------------------
# Tests: audio send
# ---------------------------------------------------------------------------

class TestAudioSend:
    def test_send_audio_chunk_puts_binary_frame_to_websocket(self) -> None:
        ws = _FakeWebSocket()
        transport, loop, _ = _start_transport(ws)
        try:
            transport.send_audio_chunk(_make_audio_chunk())
            time.sleep(0.2)
            frame = asyncio.run_coroutine_threadsafe(ws.sent.get(), loop).result(timeout=2.0)
            assert isinstance(frame, bytes)
        finally:
            _stop_transport(transport, loop)

    def test_encoded_frame_decodes_back_to_correct_audio(self) -> None:
        ws = _FakeWebSocket()
        transport, loop, _ = _start_transport(ws)
        try:
            audio = np.ones(256, dtype=np.float32) * 0.5
            transport.send_audio_chunk({"audio": audio, "timestamp": 2.0})
            time.sleep(0.2)
            frame_bytes = asyncio.run_coroutine_threadsafe(ws.sent.get(), loop).result(timeout=2.0)
            decoded = decode_audio_frame(frame_bytes, expected_session_id=_SESSION_ID)
            assert decoded.num_samples == 256
            np.testing.assert_array_almost_equal(decoded.audio, audio)
        finally:
            _stop_transport(transport, loop)

    def test_timestamp_preserved_in_encoded_frame(self) -> None:
        ws = _FakeWebSocket()
        transport, loop, _ = _start_transport(ws)
        try:
            transport.send_audio_chunk({"audio": np.zeros(64, dtype=np.float32), "timestamp": 9.99})
            time.sleep(0.2)
            frame_bytes = asyncio.run_coroutine_threadsafe(ws.sent.get(), loop).result(timeout=2.0)
            decoded = decode_audio_frame(frame_bytes, expected_session_id=_SESSION_ID)
            assert decoded.timestamp == pytest.approx(9.99)
        finally:
            _stop_transport(transport, loop)

    def test_chunk_ids_are_monotonically_increasing(self) -> None:
        ws = _FakeWebSocket()
        transport, loop, _ = _start_transport(ws)
        try:
            for _ in range(3):
                transport.send_audio_chunk(_make_audio_chunk())
            time.sleep(0.3)
            chunk_ids = []
            for _ in range(3):
                frame_bytes = asyncio.run_coroutine_threadsafe(ws.sent.get(), loop).result(timeout=2.0)
                decoded = decode_audio_frame(frame_bytes, expected_session_id=_SESSION_ID)
                chunk_ids.append(decoded.chunk_id)
            assert chunk_ids == sorted(chunk_ids)
            assert len(set(chunk_ids)) == 3
        finally:
            _stop_transport(transport, loop)


# ---------------------------------------------------------------------------
# Tests: result receive
# ---------------------------------------------------------------------------

class TestResultReceive:
    def test_partial_result_json_dispatches_to_on_partial_update(self) -> None:
        ws = _FakeWebSocket()
        subscriber = MagicMock()
        transport, loop, _ = _start_transport(ws, subscriber=subscriber)
        try:
            ws.put_message(_make_result_json(status="partial", text="testing"))
            time.sleep(0.3)
            subscriber.on_partial_update.assert_called_once()
            result = subscriber.on_partial_update.call_args[0][0]
            assert result.text == "testing"
        finally:
            _stop_transport(transport, loop)

    def test_final_result_json_dispatches_to_on_finalization(self) -> None:
        ws = _FakeWebSocket()
        subscriber = MagicMock()
        transport, loop, _ = _start_transport(ws, subscriber=subscriber)
        try:
            ws.put_message(_make_result_json(status="final", text="done"))
            time.sleep(0.3)
            subscriber.on_finalization.assert_called_once()
            result = subscriber.on_finalization.call_args[0][0]
            assert result.text == "done"
        finally:
            _stop_transport(transport, loop)

    def test_session_created_json_does_not_raise(self) -> None:
        ws = _FakeWebSocket()
        subscriber = MagicMock()
        transport, loop, _ = _start_transport(ws, subscriber=subscriber)
        try:
            ws.put_message(json.dumps({"type": "session_created", "session_id": _SESSION_ID}))
            time.sleep(0.2)
            subscriber.on_partial_update.assert_not_called()
            subscriber.on_finalization.assert_not_called()
        finally:
            _stop_transport(transport, loop)

    def test_unknown_message_type_is_logged_and_skipped(self) -> None:
        ws = _FakeWebSocket()
        subscriber = MagicMock()
        transport, loop, _ = _start_transport(ws, subscriber=subscriber)
        try:
            ws.put_message(json.dumps({"type": "unknown_type"}))
            time.sleep(0.2)
            subscriber.on_partial_update.assert_not_called()
            subscriber.on_finalization.assert_not_called()
        finally:
            _stop_transport(transport, loop)


# ---------------------------------------------------------------------------
# Tests: disconnect
# ---------------------------------------------------------------------------

class TestDisconnect:
    def test_stop_completes_cleanly(self) -> None:
        ws = _FakeWebSocket()
        transport, loop, _ = _start_transport(ws)
        _stop_transport(transport, loop)

    def test_connection_closed_transitions_app_state_to_shutdown(self) -> None:
        from websockets.exceptions import ConnectionClosed
        from websockets.frames import Close

        app_state = _make_app_state()
        subscriber = MagicMock()
        remote_publisher = RemoteRecognitionPublisher(subscriber)
        ready = threading.Event()
        loop_holder: list[asyncio.AbstractEventLoop] = []

        def run_loop() -> None:
            loop = asyncio.new_event_loop()
            loop_holder.append(loop)
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        t = threading.Thread(target=run_loop, daemon=True)
        t.start()
        ready.wait(timeout=2.0)
        loop = loop_holder[0]

        class _ClosingWs:
            """WebSocket that immediately raises ConnectionClosed on iteration."""
            sent: list = []
            closed = False

            def __aiter__(self): return self

            async def __anext__(self):
                raise ConnectionClosed(
                    rcvd=Close(code=1001, reason="going away"),
                    sent=None,
                )

            async def send(self, msg) -> None:
                self.sent.append(msg)

            async def close(self) -> None:
                self.closed = True

        closing_ws = _ClosingWs()
        transport = WsClientTransport(
            server_url=_SERVER_URL,
            session_id=_SESSION_ID,
            app_state=app_state,
            publisher=remote_publisher,
            loop=loop,
        )
        asyncio.run_coroutine_threadsafe(transport.start(closing_ws), loop).result(timeout=2.0)
        time.sleep(0.5)
        assert app_state.get_state() == "shutdown"


# ---------------------------------------------------------------------------
# Tests: backpressure
# ---------------------------------------------------------------------------

class TestBackpressure:
    def test_full_send_queue_does_not_block_caller(self) -> None:
        class _SlowWebSocket(_FakeWebSocket):
            async def send(self, message) -> None:
                await asyncio.sleep(60)

        ws = _SlowWebSocket()
        transport, loop, _ = _start_transport(ws)
        try:
            start = time.monotonic()
            for _ in range(30):
                transport.send_audio_chunk(_make_audio_chunk())
            elapsed = time.monotonic() - start
            assert elapsed < 1.0
        finally:
            _stop_transport(transport, loop)
