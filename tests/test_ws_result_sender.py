"""Tests for WsResultSender (Phase 2).

Strategy: real asyncio event loop on a background thread, a plain asyncio.Queue
as the websocket send sink, publish_* called from a separate threading.Thread
to exercise the sync→async bridge.
"""

import asyncio
import json
import threading
import time

import pytest

from src.network.codec import decode_audio_frame
from src.server.WsResultSender import WsResultSender
from src.types import RecognitionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(text: str = "hello", utterance_id: int = 1) -> RecognitionResult:
    return RecognitionResult(
        text=text,
        start_time=0.0,
        end_time=1.0,
        chunk_ids=[1, 2],
        confidence=0.9,
        token_confidences=[0.9, 0.9],
        audio_rms=0.05,
        confidence_variance=0.001,
        utterance_id=utterance_id,
    )


class _FakeWebSocket:
    """Async websocket stand-in: collects sent messages into a queue."""

    def __init__(self) -> None:
        self.sent: asyncio.Queue[str] = asyncio.Queue()
        self.closed = False

    async def send(self, message: str) -> None:
        if self.closed:
            raise ConnectionError("closed")
        await self.sent.put(message)

    async def drain(self, timeout: float = 2.0) -> list[str]:
        """Collect all messages sent within timeout seconds."""
        messages = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                msg = self.sent.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.02)
        return messages


def _start_sender(session_id: str, ws: _FakeWebSocket) -> tuple[WsResultSender, asyncio.AbstractEventLoop, threading.Thread]:
    """Start a WsResultSender on a daemon asyncio loop thread."""
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
    sender = WsResultSender(session_id=session_id, websocket=ws, loop=loop)
    asyncio.run_coroutine_threadsafe(sender.start(), loop).result(timeout=2.0)
    return sender, loop, t


def _stop_sender(sender: WsResultSender, loop: asyncio.AbstractEventLoop) -> None:
    asyncio.run_coroutine_threadsafe(sender.stop(), loop).result(timeout=5.0)


# ---------------------------------------------------------------------------
# Tests: publish_partial_update
# ---------------------------------------------------------------------------

class TestPublishPartialUpdate:
    def test_enqueues_message_as_json(self) -> None:
        ws = _FakeWebSocket()
        sender, loop, _ = _start_sender("s1", ws)
        try:
            sender.publish_partial_update(_result("hi", utterance_id=2))
            time.sleep(0.2)
            msg = asyncio.run_coroutine_threadsafe(ws.sent.get(), loop).result(timeout=2.0)
            obj = json.loads(msg)
            assert obj["type"] == "recognition_result"
            assert obj["status"] == "partial"
            assert obj["text"] == "hi"
        finally:
            _stop_sender(sender, loop)

    def test_partial_result_contains_session_id(self) -> None:
        ws = _FakeWebSocket()
        sender, loop, _ = _start_sender("my-session", ws)
        try:
            sender.publish_partial_update(_result())
            time.sleep(0.2)
            msg = asyncio.run_coroutine_threadsafe(ws.sent.get(), loop).result(timeout=2.0)
            obj = json.loads(msg)
            assert obj["session_id"] == "my-session"
        finally:
            _stop_sender(sender, loop)

    def test_utterance_id_propagated(self) -> None:
        ws = _FakeWebSocket()
        sender, loop, _ = _start_sender("s1", ws)
        try:
            sender.publish_partial_update(_result(utterance_id=7))
            time.sleep(0.2)
            msg = asyncio.run_coroutine_threadsafe(ws.sent.get(), loop).result(timeout=2.0)
            obj = json.loads(msg)
            assert obj["utterance_id"] == 7
        finally:
            _stop_sender(sender, loop)


# ---------------------------------------------------------------------------
# Tests: publish_finalization
# ---------------------------------------------------------------------------

class TestPublishFinalization:
    def test_enqueues_final_status(self) -> None:
        ws = _FakeWebSocket()
        sender, loop, _ = _start_sender("s1", ws)
        try:
            sender.publish_finalization(_result("done"))
            time.sleep(0.2)
            msg = asyncio.run_coroutine_threadsafe(ws.sent.get(), loop).result(timeout=2.0)
            obj = json.loads(msg)
            assert obj["status"] == "final"
            assert obj["text"] == "done"
        finally:
            _stop_sender(sender, loop)

    def test_final_result_contains_session_id(self) -> None:
        ws = _FakeWebSocket()
        sender, loop, _ = _start_sender("sess-99", ws)
        try:
            sender.publish_finalization(_result())
            time.sleep(0.2)
            msg = asyncio.run_coroutine_threadsafe(ws.sent.get(), loop).result(timeout=2.0)
            obj = json.loads(msg)
            assert obj["session_id"] == "sess-99"
        finally:
            _stop_sender(sender, loop)


# ---------------------------------------------------------------------------
# Tests: sender task drains queue and calls websocket.send
# ---------------------------------------------------------------------------

class TestSenderTaskDrains:
    def test_multiple_messages_all_delivered(self) -> None:
        ws = _FakeWebSocket()
        sender, loop, _ = _start_sender("s1", ws)
        try:
            for i in range(5):
                sender.publish_partial_update(_result(f"word{i}"))
            time.sleep(0.4)
            messages = []
            while True:
                try:
                    msg = ws.sent.get_nowait()
                    messages.append(json.loads(msg))
                except asyncio.QueueEmpty:
                    break
            texts = [m["text"] for m in messages]
            for i in range(5):
                assert f"word{i}" in texts
        finally:
            _stop_sender(sender, loop)

    def test_partial_and_final_interleaved(self) -> None:
        ws = _FakeWebSocket()
        sender, loop, _ = _start_sender("s1", ws)
        try:
            sender.publish_partial_update(_result("partial"))
            sender.publish_finalization(_result("final"))
            time.sleep(0.3)
            statuses = []
            while True:
                try:
                    msg = ws.sent.get_nowait()
                    statuses.append(json.loads(msg)["status"])
                except asyncio.QueueEmpty:
                    break
            assert "partial" in statuses
            assert "final" in statuses
        finally:
            _stop_sender(sender, loop)


# ---------------------------------------------------------------------------
# Tests: backpressure — full queue drops item without blocking sync caller
# ---------------------------------------------------------------------------

class TestBackpressure:
    def test_full_queue_does_not_block_caller(self) -> None:
        """Filling beyond maxsize must return quickly from the sync thread."""

        class _SlowWebSocket:
            """Websocket that never completes sends — keeps queue full."""
            async def send(self, message: str) -> None:
                await asyncio.sleep(60)

        ws = _SlowWebSocket()
        sender, loop, _ = _start_sender("s1", ws)  # type: ignore[arg-type]
        try:
            start = time.monotonic()
            # Flood beyond the bounded queue capacity (maxsize=20)
            for _ in range(30):
                sender.publish_partial_update(_result())
            elapsed = time.monotonic() - start
            # All 30 calls must return within 1 second total
            assert elapsed < 1.0
        finally:
            _stop_sender(sender, loop)


# ---------------------------------------------------------------------------
# Tests: closed connection does not raise to caller
# ---------------------------------------------------------------------------

class TestClosedConnection:
    def test_send_on_closed_websocket_does_not_raise(self) -> None:
        ws = _FakeWebSocket()
        sender, loop, _ = _start_sender("s1", ws)
        ws.closed = True
        try:
            # Should not raise — error is logged and swallowed
            sender.publish_partial_update(_result())
            time.sleep(0.2)
        finally:
            _stop_sender(sender, loop)
