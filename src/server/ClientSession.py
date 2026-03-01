"""Per-WebSocket-connection session: owns all stateful pipeline components.

Each ClientSession isolates one connected client's audio processing pipeline.
The session is created on WebSocket connect and destroyed on disconnect.
"""

import logging
import queue
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.asr.VoiceActivityDetector import VoiceActivityDetector
from src.postprocessing.IncrementalTextMatcher import IncrementalTextMatcher
from src.server.WsResultSender import WsResultSender
from src.sound.GrowingWindowAssembler import GrowingWindowAssembler
from src.sound.SoundPreProcessor import SoundPreProcessor
from src.SpeechEndRouter import SpeechEndRouter

if TYPE_CHECKING:
    from src.server.RecognizerService import RecognizerService
    from src.ServerApplicationState import ServerApplicationState

logger = logging.getLogger(__name__)

_SESSION_PREFIX = 10_000_000
_QUEUE_MAXSIZE = 100


class ClientSession:
    """Owns and orchestrates all per-session pipeline components.

    Responsibilities:
    - Allocate per-session queues and pipeline workers on construction.
    - Wire workers together: chunk_queue → SoundPreProcessor → speech_queue →
      SpeechEndRouter → IncrementalTextMatcher → WsResultSender → websocket.
    - Share RecognizerService.input_queue as the recognizer sink.
    - Start and stop all workers in order.

    Args:
        session_id: UUID string assigned to this session.
        session_index: 1-based index used to partition message IDs.
        websocket: Active WebSocket connection (must support async send).
        loop: asyncio event loop running the websocket handler.
        recognizer_service: Shared inference service; this session registers/unregisters.
        app_state: Server lifecycle state used by pipeline workers for shutdown signaling.
        config: Application configuration dict (audio/vad/windowing sections required).
        vad_model_path: Absolute path to the Silero VAD ONNX model file.
    """

    def __init__(
        self,
        session_id: str,
        session_index: int,
        websocket: Any,
        loop: Any,
        recognizer_service: "RecognizerService",
        app_state: "ServerApplicationState",
        config: dict,
        vad_model_path: Path,
    ) -> None:
        self._session_id = session_id
        self._session_index = session_index
        self._recognizer_service = recognizer_service
        self._app_state = app_state

        self._chunk_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._speech_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._recognizer_output_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._matcher_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._control_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)

        vad = VoiceActivityDetector(config=config, model_path=vad_model_path)
        windower = GrowingWindowAssembler(speech_queue=self._speech_queue, config=config)

        self._sound_preprocessor = SoundPreProcessor(
            chunk_queue=self._chunk_queue,
            speech_queue=self._speech_queue,
            vad=vad,
            windower=windower,
            config=config,
            control_queue=self._control_queue,
            app_state=app_state,
        )

        first_message_id = session_index * _SESSION_PREFIX + 1
        self._speech_end_router = SpeechEndRouter(
            speech_queue=self._speech_queue,
            recognizer_queue=recognizer_service.input_queue,
            recognizer_output_queue=self._recognizer_output_queue,
            matcher_queue=self._matcher_queue,
            app_state=app_state,
            control_queue=self._control_queue,
            first_message_id=first_message_id,
        )

        self._result_sender = WsResultSender(
            session_id=session_id,
            websocket=websocket,
            loop=loop,
        )

        self._matcher = IncrementalTextMatcher(
            text_queue=self._matcher_queue,
            publisher=self._result_sender,
            app_state=app_state,
        )

    @property
    def chunk_queue(self) -> queue.Queue:
        """The audio chunk queue consumed by SoundPreProcessor.

        Returns:
            Thread-safe queue accepting ``{"audio": ndarray, "timestamp": float}`` dicts.
        """
        return self._chunk_queue

    async def start(self) -> None:
        """Register with RecognizerService and start all pipeline workers.

        Must be called from the asyncio event loop thread so WsResultSender
        can create its asyncio task on the running loop.
        """
        self._recognizer_service.register_session(
            self._session_index, self._recognizer_output_queue
        )
        await self._result_sender.start()
        self._sound_preprocessor.start()
        self._speech_end_router.start()
        self._matcher.start()
        logger.info("ClientSession[%s]: started (index=%s)", self._session_id, self._session_index)

    async def close(self) -> None:
        """Stop all pipeline workers in order and unregister from RecognizerService.

        Algorithm:
            1. Stop SoundPreProcessor — stop() already joins its thread internally.
            2. Stop SpeechEndRouter and join its thread.
            3. Stop IncrementalTextMatcher and join its thread.
            4. Stop WsResultSender — cancels the async drain task.
            5. Unregister session output queue from RecognizerService.
        """
        self._sound_preprocessor.stop()

        self._speech_end_router.stop()
        if self._speech_end_router.thread is not None:
            self._speech_end_router.thread.join(timeout=2.0)

        self._matcher.stop()
        if self._matcher.thread is not None:
            self._matcher.thread.join(timeout=2.0)

        await self._result_sender.stop()

        self._recognizer_service.unregister_session(self._session_index)
        logger.info("ClientSession[%s]: closed", self._session_id)
