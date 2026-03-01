"""Tk client orchestrator: AudioSource + WsClientTransport + GUI.

ClientApp wires the audio capture, WebSocket transport, and Tk GUI together
for one client session.  It receives a session_id (already negotiated with
the server) and the server WebSocket URL.

Audio bridge: a daemon thread drains AudioSource.chunk_queue and forwards
each chunk to WsClientTransport.send_audio_chunk() so the sync audio callback
is fully decoupled from the async transport internals.
"""

import json
import logging
import queue
import threading
from typing import Any

from src.ApplicationState import ApplicationState
from src.RecognitionResultPublisher import RecognitionResultPublisher
from src.client.tk.RemoteRecognitionPublisher import RemoteRecognitionPublisher
from src.client.tk.WsClientTransport import WsClientTransport
from src.client.tk.gui.ApplicationWindow import ApplicationWindow
from src.client.tk.gui.TextInsertionService import TextInsertionService
from src.client.tk.controllers.PauseController import PauseController
from src.client.tk.sound.AudioSource import AudioSource

logger = logging.getLogger(__name__)


class ClientApp:
    """Orchestrates AudioSource, WsClientTransport, and the Tk GUI.

    Responsibilities:
    - Create ApplicationState (client-side with paused support).
    - Create RecognitionResultPublisher and subscribe TextFormatter and TextInserter.
    - Create RemoteRecognitionPublisher that forwards WS results to the local publisher.
    - Create WsClientTransport carrying audio and receiving results.
    - Create AudioSource and wire its chunks to the transport via a drain thread.
    - Create ApplicationWindow (Tk GUI).

    Args:
        server_url: WebSocket URL of the server (e.g. ``ws://127.0.0.1:9000``).
        session_id: Session UUID assigned by the server in ``session_created``.
        config: Application configuration dict.
        verbose: Enable verbose logging.
    """

    def __init__(
        self,
        server_url: str,
        session_id: str,
        config: dict,
        verbose: bool = False,
    ) -> None:
        self._server_url = server_url
        self._session_id = session_id
        self._config = config
        self._verbose = verbose

        self._bridge_thread: threading.Thread | None = None
        self._bridge_running = False

        self.app_state = ApplicationState(config=config)

        self._publisher = RecognitionResultPublisher(verbose=verbose)

        self._text_insertion_service = TextInsertionService(
            publisher=self._publisher,
            verbose=verbose,
        )

        self._pause_controller = PauseController(self.app_state)

        self._window = ApplicationWindow(
            app_state=self.app_state,
            config=config,
            verbose=verbose,
            insertion_controller=self._text_insertion_service.controller,
            pause_controller=self._pause_controller,
        )

        formatter = self._window.get_formatter()
        self._publisher.subscribe(formatter)

        self._remote_publisher = RemoteRecognitionPublisher(publisher=self._publisher)

        self._transport = WsClientTransport(
            server_url=server_url,
            session_id=session_id,
            app_state=self.app_state,
            publisher=self._remote_publisher,
            loop=None,
        )

        self._audio_source = AudioSource(
            chunk_queue=queue.Queue(maxsize=200),
            config=config,
            app_state=self.app_state,
            verbose=verbose,
        )

    @classmethod
    def from_session_created(
        cls,
        server_url: str,
        session_created_json: str,
        config: dict,
        verbose: bool = False,
    ) -> "ClientApp":
        """Construct from a ``session_created`` JSON frame received from the server.

        Args:
            server_url: WebSocket server URL.
            session_created_json: Raw JSON string of the ``session_created`` frame.
            config: Application config dict.
            verbose: Enable verbose logging.

        Returns:
            Configured ClientApp instance ready for start().

        Raises:
            ValueError: If the JSON is malformed or missing required fields.
        """
        try:
            obj = json.loads(session_created_json)
            session_id: str = obj["session_id"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"Invalid session_created frame: {exc}") from exc

        return cls(
            server_url=server_url,
            session_id=session_id,
            config=config,
            verbose=verbose,
        )

    @property
    def session_id(self) -> str:
        """Session UUID assigned by the server.

        Returns:
            The session identifier string.
        """
        return self._session_id

    def start(self) -> None:
        """Start the audio bridge thread and the AudioSource.

        Algorithm:
            1. Transition app_state to running.
            2. Start audio bridge drain thread.
            3. Start AudioSource (mic capture).
        """
        self.app_state.set_state("running")
        self._start_audio_bridge()
        self._audio_source.start()
        logger.info("ClientApp[%s]: started", self._session_id)

    def stop(self) -> None:
        """Stop AudioSource and audio bridge thread.

        Algorithm:
            1. Stop AudioSource.
            2. Stop audio bridge drain thread.
            3. Transition app_state to shutdown if not already.
        """
        self._audio_source.stop()
        self._stop_audio_bridge()

        try:
            self.app_state.set_state("shutdown")
        except ValueError:
            pass

        logger.info("ClientApp[%s]: stopped", self._session_id)

    def get_root(self) -> Any:
        """Return the tkinter root window.

        Returns:
            tk.Tk root instance.
        """
        return self._window.get_root()

    # ------------------------------------------------------------------
    # Internal: audio bridge
    # ------------------------------------------------------------------

    def _start_audio_bridge(self) -> None:
        """Start the daemon thread that drains AudioSource.chunk_queue → transport."""
        self._bridge_running = True
        self._bridge_thread = threading.Thread(
            target=self._drain_audio,
            daemon=True,
            name="ClientApp-AudioBridge",
        )
        self._bridge_thread.start()

    def _stop_audio_bridge(self) -> None:
        """Signal the bridge thread to stop and wait for it to exit."""
        self._bridge_running = False
        if self._bridge_thread is not None:
            self._bridge_thread.join(timeout=2.0)
            self._bridge_thread = None

    def _drain_audio(self) -> None:
        """Bridge thread: drain AudioSource.chunk_queue and forward to transport.

        Algorithm:
            1. Block on chunk_queue.get(timeout=0.05).
            2. Call transport.send_audio_chunk(chunk).
            3. Repeat until _bridge_running is False.
        """
        chunk_queue: queue.Queue = self._audio_source.chunk_queue

        while self._bridge_running:
            try:
                chunk = chunk_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                self._transport.send_audio_chunk(chunk)
            except Exception:
                logger.exception("ClientApp[%s]: error forwarding audio chunk", self._session_id)
