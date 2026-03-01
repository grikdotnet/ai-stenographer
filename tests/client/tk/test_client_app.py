"""Tests for ClientApp (Phase 8).

Strategy: mock AudioSource, WsClientTransport, ApplicationWindow, and
RecognitionResultPublisher to avoid loading real models or opening real windows.

Tests verify:
- client connects to server and receives session_created
- AudioSource is wired to WsClientTransport (audio chunks forwarded to transport)
- ClientApp.stop() stops AudioSource and transport
"""

import asyncio
import json
import threading
import time
import queue
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np
import pytest

from src.ApplicationState import ApplicationState
from src.client.tk.ClientApp import ClientApp


_SERVER_URL = "ws://127.0.0.1:19876"
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
}

_SESSION_CREATED_MSG = json.dumps({
    "type": "session_created",
    "session_id": "test-session-uuid",
    "protocol_version": "v1",
    "server_time": 1000.0,
    "server_config": {
        "sample_rate": 16000,
        "chunk_duration_sec": 0.032,
        "audio_dtype": "float32",
        "channels": 1,
    },
})


def _make_mock_audio_source(chunk_queue: queue.Queue) -> MagicMock:
    source = MagicMock()
    source.chunk_queue = chunk_queue
    source.start = MagicMock()
    source.stop = MagicMock()
    return source


def _make_mock_transport() -> MagicMock:
    transport = MagicMock()
    transport.send_audio_chunk = MagicMock()
    transport.stop = AsyncMock()
    transport.start = AsyncMock()
    return transport


def _make_mock_app_window() -> MagicMock:
    window = MagicMock()
    window.get_root.return_value = MagicMock()
    window.get_formatter.return_value = MagicMock()
    return window


class TestClientAppWiring:
    def test_audio_source_wired_to_transport(self) -> None:
        """Audio chunks placed in AudioSource.chunk_queue reach transport.send_audio_chunk."""
        chunk_q: queue.Queue = queue.Queue()
        mock_source = _make_mock_audio_source(chunk_q)
        mock_transport = _make_mock_transport()

        received_event = threading.Event()
        captured = []

        def capture_chunk(chunk):
            captured.append(chunk)
            received_event.set()

        mock_transport.send_audio_chunk.side_effect = capture_chunk

        audio_chunk = {"audio": np.zeros(512, dtype=np.float32), "timestamp": 1.0}

        with (
            patch("src.client.tk.ClientApp.AudioSource", return_value=mock_source),
            patch("src.client.tk.ClientApp.WsClientTransport", return_value=mock_transport),
            patch("src.client.tk.ClientApp.ApplicationWindow", return_value=_make_mock_app_window()),
            patch("src.client.tk.ClientApp.RecognitionResultPublisher"),
            patch("src.client.tk.ClientApp.TextInsertionService"),
            patch("src.client.tk.ClientApp.PauseController"),
        ):
            app = ClientApp(
                server_url=_SERVER_URL,
                session_id="test-session-uuid",
                config=_CONFIG,
            )
            app._start_audio_bridge()

            chunk_q.put(audio_chunk)
            received_event.wait(timeout=1.0)

            app._stop_audio_bridge()

        assert len(captured) == 1
        assert captured[0] is audio_chunk

    def test_audio_source_is_started(self) -> None:
        """ClientApp.start() calls AudioSource.start()."""
        chunk_q: queue.Queue = queue.Queue()
        mock_source = _make_mock_audio_source(chunk_q)

        with (
            patch("src.client.tk.ClientApp.AudioSource", return_value=mock_source),
            patch("src.client.tk.ClientApp.WsClientTransport", return_value=_make_mock_transport()),
            patch("src.client.tk.ClientApp.ApplicationWindow", return_value=_make_mock_app_window()),
            patch("src.client.tk.ClientApp.RecognitionResultPublisher"),
            patch("src.client.tk.ClientApp.TextInsertionService"),
            patch("src.client.tk.ClientApp.PauseController"),
        ):
            app = ClientApp(
                server_url=_SERVER_URL,
                session_id="test-session-uuid",
                config=_CONFIG,
            )
            app.start()

        mock_source.start.assert_called_once()

    def test_stop_stops_audio_source(self) -> None:
        """ClientApp.stop() calls AudioSource.stop()."""
        chunk_q: queue.Queue = queue.Queue()
        mock_source = _make_mock_audio_source(chunk_q)

        with (
            patch("src.client.tk.ClientApp.AudioSource", return_value=mock_source),
            patch("src.client.tk.ClientApp.WsClientTransport", return_value=_make_mock_transport()),
            patch("src.client.tk.ClientApp.ApplicationWindow", return_value=_make_mock_app_window()),
            patch("src.client.tk.ClientApp.RecognitionResultPublisher"),
            patch("src.client.tk.ClientApp.TextInsertionService"),
            patch("src.client.tk.ClientApp.PauseController"),
        ):
            app = ClientApp(
                server_url=_SERVER_URL,
                session_id="test-session-uuid",
                config=_CONFIG,
            )
            app.start()
            app.stop()

        mock_source.stop.assert_called_once()


class TestClientAppSessionCreated:
    def test_session_created_message_sets_session_id(self) -> None:
        """ClientApp stores the session_id parsed from session_created message."""
        chunk_q: queue.Queue = queue.Queue()

        with (
            patch("src.client.tk.ClientApp.AudioSource", return_value=_make_mock_audio_source(chunk_q)),
            patch("src.client.tk.ClientApp.WsClientTransport", return_value=_make_mock_transport()),
            patch("src.client.tk.ClientApp.ApplicationWindow", return_value=_make_mock_app_window()),
            patch("src.client.tk.ClientApp.RecognitionResultPublisher"),
            patch("src.client.tk.ClientApp.TextInsertionService"),
            patch("src.client.tk.ClientApp.PauseController"),
        ):
            app = ClientApp(
                server_url=_SERVER_URL,
                session_id="test-session-uuid",
                config=_CONFIG,
            )

        assert app.session_id == "test-session-uuid"

    def test_create_from_session_created_msg_parses_session_id(self) -> None:
        """ClientApp.from_session_created() extracts session_id from JSON."""
        chunk_q: queue.Queue = queue.Queue()

        with (
            patch("src.client.tk.ClientApp.AudioSource", return_value=_make_mock_audio_source(chunk_q)),
            patch("src.client.tk.ClientApp.WsClientTransport", return_value=_make_mock_transport()),
            patch("src.client.tk.ClientApp.ApplicationWindow", return_value=_make_mock_app_window()),
            patch("src.client.tk.ClientApp.RecognitionResultPublisher"),
            patch("src.client.tk.ClientApp.TextInsertionService"),
            patch("src.client.tk.ClientApp.PauseController"),
        ):
            app = ClientApp.from_session_created(
                server_url=_SERVER_URL,
                session_created_json=_SESSION_CREATED_MSG,
                config=_CONFIG,
            )

        assert app.session_id == "test-session-uuid"


class TestClientAppPublisherWiring:
    def test_formatter_subscribed_to_publisher(self) -> None:
        """TextFormatter returned by ApplicationWindow is subscribed to the publisher."""
        chunk_q: queue.Queue = queue.Queue()
        mock_formatter = MagicMock()
        mock_window = _make_mock_app_window()
        mock_window.get_formatter.return_value = mock_formatter

        mock_publisher = MagicMock()

        with (
            patch("src.client.tk.ClientApp.AudioSource", return_value=_make_mock_audio_source(chunk_q)),
            patch("src.client.tk.ClientApp.WsClientTransport", return_value=_make_mock_transport()),
            patch("src.client.tk.ClientApp.ApplicationWindow", return_value=mock_window),
            patch("src.client.tk.ClientApp.RecognitionResultPublisher", return_value=mock_publisher),
            patch("src.client.tk.ClientApp.TextInsertionService"),
            patch("src.client.tk.ClientApp.PauseController"),
        ):
            ClientApp(
                server_url=_SERVER_URL,
                session_id="test-session-uuid",
                config=_CONFIG,
            )

        mock_publisher.subscribe.assert_called_with(mock_formatter)

    def test_remote_publisher_receives_recognition_results(self) -> None:
        """RemoteRecognitionPublisher is constructed with the local publisher."""
        chunk_q: queue.Queue = queue.Queue()
        mock_publisher = MagicMock()

        with (
            patch("src.client.tk.ClientApp.AudioSource", return_value=_make_mock_audio_source(chunk_q)),
            patch("src.client.tk.ClientApp.WsClientTransport") as MockTransport,
            patch("src.client.tk.ClientApp.ApplicationWindow", return_value=_make_mock_app_window()),
            patch("src.client.tk.ClientApp.RecognitionResultPublisher", return_value=mock_publisher),
            patch("src.client.tk.ClientApp.TextInsertionService"),
            patch("src.client.tk.ClientApp.PauseController"),
            patch("src.client.tk.ClientApp.RemoteRecognitionPublisher") as MockRemote,
        ):
            ClientApp(
                server_url=_SERVER_URL,
                session_id="test-session-uuid",
                config=_CONFIG,
            )

        MockRemote.assert_called_once_with(publisher=mock_publisher)
