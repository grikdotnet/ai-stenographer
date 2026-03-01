"""Tests for ClientSession (Phase 4).

Strategy: patch VoiceActivityDetector and GrowingWindowAssembler to avoid
loading the real ONNX model. Real queues and threading.Event for coordination.
"""

import asyncio
import queue
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.ServerApplicationState import ServerApplicationState
from src.server.ClientSession import ClientSession
from src.server.RecognizerService import RecognizerService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONFIG = {
    "audio": {
        "sample_rate": 16000,
        "chunk_duration": 0.032,
        "silence_energy_threshold": 1.5,
        "rms_normalization": {
            "target_rms": 0.05,
            "silence_threshold": 0.001,
            "gain_smoothing": 0.9,
        },
    },
    "vad": {
        "frame_duration_ms": 32,
        "threshold": 0.5,
    },
    "windowing": {
        "max_speech_duration_ms": 3000,
        "max_window_duration": 7.0,
    },
}

_VAD_MODEL_PATH = Path("/fake/silero_vad.onnx")


def _make_mock_vad() -> MagicMock:
    vad = MagicMock()
    vad.process_frame.return_value = {"is_speech": False, "speech_probability": 0.1}
    return vad


def _make_mock_windower() -> MagicMock:
    windower = MagicMock()
    windower.process_segment = MagicMock()
    windower.flush = MagicMock()
    return windower


def _make_mock_websocket() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


def _make_recognizer_service(app_state: ServerApplicationState) -> RecognizerService:
    recognizer = MagicMock()
    recognizer.recognize_window.return_value = None
    service = RecognizerService(recognizer=recognizer, app_state=app_state)
    service.start()
    return service


def _make_session(
    session_id: str = "test-session",
    session_index: int = 1,
    app_state: ServerApplicationState | None = None,
    recognizer_service: RecognizerService | None = None,
) -> tuple[ClientSession, ServerApplicationState, RecognizerService, asyncio.AbstractEventLoop]:
    if app_state is None:
        app_state = ServerApplicationState()
        app_state.set_state("running")

    loop = asyncio.new_event_loop()
    ws = _make_mock_websocket()

    if recognizer_service is None:
        recognizer_service = _make_recognizer_service(app_state)

    with (
        patch("src.server.ClientSession.VoiceActivityDetector", return_value=_make_mock_vad()),
        patch("src.server.ClientSession.GrowingWindowAssembler", return_value=_make_mock_windower()),
    ):
        session = ClientSession(
            session_id=session_id,
            session_index=session_index,
            websocket=ws,
            loop=loop,
            recognizer_service=recognizer_service,
            app_state=app_state,
            config=_CONFIG,
            vad_model_path=_VAD_MODEL_PATH,
        )

    return session, app_state, recognizer_service, loop


# ---------------------------------------------------------------------------
# Session lifecycle: start() and close()
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    def test_start_starts_all_workers(self) -> None:
        session, app_state, recognizer_service, loop = _make_session()

        loop.run_until_complete(session.start())

        assert session._sound_preprocessor.is_running
        assert session._speech_end_router.is_running
        assert session._matcher.is_running

        loop.run_until_complete(session.close())
        recognizer_service.stop()
        recognizer_service.join()
        loop.close()

    def test_close_stops_all_workers(self) -> None:
        session, app_state, recognizer_service, loop = _make_session()

        loop.run_until_complete(session.start())
        loop.run_until_complete(session.close())

        assert not session._sound_preprocessor.is_running
        assert not session._speech_end_router.is_running
        assert not session._matcher.is_running

        recognizer_service.stop()
        recognizer_service.join()
        loop.close()

    def test_close_unregisters_session_from_recognizer_service(self) -> None:
        session, app_state, recognizer_service, loop = _make_session(session_index=1)

        loop.run_until_complete(session.start())
        assert 1 in recognizer_service._session_output_queues

        loop.run_until_complete(session.close())
        assert 1 not in recognizer_service._session_output_queues

        recognizer_service.stop()
        recognizer_service.join()
        loop.close()

    def test_start_registers_session_with_recognizer_service(self) -> None:
        session, app_state, recognizer_service, loop = _make_session(session_index=3)

        loop.run_until_complete(session.start())
        assert 3 in recognizer_service._session_output_queues

        loop.run_until_complete(session.close())
        recognizer_service.stop()
        recognizer_service.join()
        loop.close()


# ---------------------------------------------------------------------------
# chunk_queue is accessible and connected to SoundPreProcessor
# ---------------------------------------------------------------------------

class TestChunkQueue:
    def test_chunk_queue_is_the_same_queue_sound_preprocessor_reads(self) -> None:
        session, app_state, recognizer_service, loop = _make_session()

        assert session.chunk_queue is session._sound_preprocessor.chunk_queue

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()

    def test_chunk_queue_accepts_audio_dicts(self) -> None:
        session, app_state, recognizer_service, loop = _make_session()

        chunk = {"audio": np.zeros(512, dtype=np.float32), "timestamp": 1.0}
        session.chunk_queue.put_nowait(chunk)

        retrieved = session.chunk_queue.get_nowait()
        assert retrieved["audio"].dtype == np.float32

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()


# ---------------------------------------------------------------------------
# Session message ID partitioning
# ---------------------------------------------------------------------------

class TestMessageIdPartitioning:
    def test_session1_first_message_id_is_10_000_001(self) -> None:
        session, _, recognizer_service, loop = _make_session(session_index=1)

        assert session._speech_end_router._next_message_id == 10_000_001

        loop.close()
        recognizer_service.stop()
        recognizer_service.join()

    def test_session2_first_message_id_is_20_000_001(self) -> None:
        app_state = ServerApplicationState()
        app_state.set_state("running")
        recognizer_service = _make_recognizer_service(app_state)

        session1, _, _, loop1 = _make_session(
            session_index=1, app_state=app_state, recognizer_service=recognizer_service
        )
        session2, _, _, loop2 = _make_session(
            session_index=2, app_state=app_state, recognizer_service=recognizer_service
        )

        assert session1._speech_end_router._next_message_id == 10_000_001
        assert session2._speech_end_router._next_message_id == 20_000_001

        loop1.close()
        loop2.close()
        recognizer_service.stop()
        recognizer_service.join()
