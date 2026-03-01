"""WebSocket audio receiver: async-to-sync bridge for incoming audio chunks.

receive_audio() is an async coroutine that runs for the lifetime of one
WebSocket session.  It reads frames from the websocket, validates them,
and puts raw audio dicts onto chunk_queue for SoundPreProcessor to consume.
"""

import json
import logging
import queue
import time
from typing import Any

from websockets.exceptions import ConnectionClosed

from src.network.codec import decode_audio_frame, decode_client_message, encode_server_message
from src.network.types import WsControlCommand, WsError

logger = logging.getLogger(__name__)

_RETURN_SHUTDOWN = "shutdown"
_RETURN_EXHAUSTED = "exhausted"
_RETURN_CONNECTION_LOST = "connection_lost"


async def receive_audio(
    websocket: Any,
    session_id: str,
    chunk_queue: queue.Queue,
) -> str:
    """Receive audio frames from the WebSocket and forward them to chunk_queue.

    Runs until one of: shutdown command received, websocket closes, or the
    async iterator is exhausted (used in tests to supply a fixed sequence).

    Algorithm:
        1. Iterate over websocket messages.
        2. Binary message → decode as audio_chunk; put dict onto chunk_queue.
           - Decode error → send non-fatal INVALID_AUDIO_FRAME error, continue.
           - Queue full → send non-fatal BACKPRESSURE_DROP error, continue.
        3. Text message → decode as client control message.
           - shutdown command → return "shutdown".
           - Unknown/invalid → send non-fatal UNKNOWN_MESSAGE_TYPE error, continue.
        4. ConnectionClosed → return "connection_lost".
        5. Async iterator exhausted (StopAsyncIteration) → return "exhausted".

    Args:
        websocket: WebSocket connection object (must support async iteration and send).
        session_id: Active session identifier used for frame validation and error messages.
        chunk_queue: Thread-safe queue consumed by SoundPreProcessor. Items are dicts
            with keys ``{"audio": np.ndarray[float32], "timestamp": float}``.

    Returns:
        Reason string: ``"shutdown"``, ``"connection_lost"``, or ``"exhausted"``.
    """
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                result = _handle_binary(message, session_id, chunk_queue)
                if result is not None:
                    await websocket.send(encode_server_message(result))
            else:
                stop, error = _handle_text(message, session_id)
                if error is not None:
                    await websocket.send(encode_server_message(error))
                if stop:
                    return _RETURN_SHUTDOWN
    except ConnectionClosed:
        logger.info("WsAudioReceiver[%s]: connection closed", session_id)
        return _RETURN_CONNECTION_LOST

    return _RETURN_EXHAUSTED


def _handle_binary(
    raw: bytes,
    session_id: str,
    chunk_queue: queue.Queue,
) -> WsError | None:
    """Decode a binary frame and push its audio onto chunk_queue.

    Args:
        raw: Raw binary websocket frame.
        session_id: Expected session identifier for validation.
        chunk_queue: Destination queue for the decoded audio chunk.

    Returns:
        A WsError to send back to the client, or None on success.
    """
    try:
        frame = decode_audio_frame(raw, expected_session_id=session_id)
    except ValueError as exc:
        logger.warning("WsAudioReceiver[%s]: invalid audio frame: %s", session_id, exc)
        return WsError(
            session_id=session_id,
            error_code="INVALID_AUDIO_FRAME",
            message=str(exc),
            fatal=False,
        )

    chunk = {"audio": frame.audio, "timestamp": frame.timestamp}

    try:
        chunk_queue.put_nowait(chunk)
    except queue.Full:
        logger.warning("WsAudioReceiver[%s]: chunk_queue full, dropping frame", session_id)
        return WsError(
            session_id=session_id,
            error_code="BACKPRESSURE_DROP",
            message="audio chunk dropped: ingress queue full",
            fatal=False,
        )

    return None


def _handle_text(
    text: str,
    session_id: str,
) -> tuple[bool, WsError | None]:
    """Decode a JSON text frame from the client.

    Args:
        text: Raw JSON string from a websocket text frame.
        session_id: Session identifier for error messages.

    Returns:
        Tuple of (should_stop, optional_error_to_send).
        should_stop is True only when a valid shutdown command is received.
    """
    try:
        msg = decode_client_message(text)
    except ValueError as exc:
        logger.warning("WsAudioReceiver[%s]: invalid client message: %s", session_id, exc)
        return False, WsError(
            session_id=session_id,
            error_code="UNKNOWN_MESSAGE_TYPE",
            message=str(exc),
            fatal=False,
        )

    if isinstance(msg, WsControlCommand) and msg.command == "shutdown":
        logger.info("WsAudioReceiver[%s]: shutdown command received", session_id)
        return True, None

    return False, None
