"""Audio-only WebSocket frame handling for server-side ingestion."""

import logging
import queue

from src.network.codec import SessionIdMismatchError, decode_audio_frame
from src.network.types import WsError

logger = logging.getLogger(__name__)


def handle_audio_frame(
    raw: bytes,
    session_id: str,
    chunk_queue: queue.Queue,
    server_state: str,
) -> WsError | None:
    """Decode and enqueue one binary audio frame.

    Args:
        raw: Raw binary websocket frame.
        session_id: Expected session identifier for validation.
        chunk_queue: Destination queue for the decoded audio chunk.
        server_state: Current server lifecycle state.

    Returns:
        A ``WsError`` response to send back to the client, or ``None`` on success.
    """
    if server_state != "running":
        return WsError(
            session_id=session_id,
            error_code="MODEL_NOT_READY",
            message=f"Audio is not accepted while server_state={server_state!r}",
            fatal=False,
        )

    try:
        frame = decode_audio_frame(raw, expected_session_id=session_id)
    except ValueError as exc:
        logger.warning("WsAudioReceiver[%s]: invalid audio frame: %s", session_id, exc)
        error_code = (
            "SESSION_ID_MISMATCH"
            if isinstance(exc, SessionIdMismatchError)
            else "INVALID_AUDIO_FRAME"
        )
        return WsError(
            session_id=session_id,
            error_code=error_code,
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
