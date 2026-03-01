"""WebSocket wire protocol message types for the STT client-server protocol (v1)."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Client → Server
# ---------------------------------------------------------------------------

@dataclass
class WsAudioFrame:
    """Binary audio_chunk frame sent from client to server.

    Carries one chunk of mono float32 PCM audio at 16 kHz.
    """

    session_id: str
    chunk_id: int
    timestamp: float
    sample_rate: int
    num_samples: int
    dtype: str
    channels: int
    audio: npt.NDArray[np.float32]


@dataclass
class WsControlCommand:
    """JSON control_command frame sent from client to server.

    Args:
        session_id: Active session identifier.
        command: Command name; only ``"shutdown"`` is valid in v1.
        timestamp: Client wall-clock time when the command was issued.
        request_id: Optional correlation identifier for tracing.
    """

    session_id: str
    command: Literal["shutdown"]
    timestamp: float
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Server → Client
# ---------------------------------------------------------------------------

@dataclass
class WsSessionCreated:
    """JSON session_created frame sent from server to client after connect.

    Args:
        session_id: UUID assigned to this session.
        protocol_version: Wire protocol version string (``"v1"``).
        server_time: Server wall-clock time at session creation.
        server_config: Audio format parameters the client must honour.
    """

    session_id: str
    protocol_version: str
    server_time: float
    server_config: dict


@dataclass
class WsRecognitionResult:
    """JSON recognition_result frame streamed from server to client.

    Args:
        session_id: Session this result belongs to.
        status: ``"partial"`` for incremental updates; ``"final"`` for finalized utterances.
        text: Recognised text.
        start_time: Utterance audio start offset in seconds.
        end_time: Utterance audio end offset in seconds.
        chunk_ids: Audio chunk IDs that contributed to this result.
        utterance_id: Monotonically increasing utterance counter.
        confidence: Mean token confidence in ``[0.0, 1.0]``.
        token_confidences: Per-token confidence scores.
        audio_rms: RMS amplitude of the utterance audio.
        confidence_variance: Variance of token confidences.
    """

    session_id: str
    status: Literal["partial", "final"]
    text: str
    start_time: float
    end_time: float
    chunk_ids: list[int] = field(default_factory=list)
    utterance_id: int = 0
    confidence: float = 0.0
    token_confidences: list[float] = field(default_factory=list)
    audio_rms: float = 0.0
    confidence_variance: float = 0.0


@dataclass
class WsSessionClosed:
    """JSON session_closed frame sent by server before graceful disconnect.

    Args:
        session_id: Session being closed.
        reason: Closure reason: ``"shutdown"``, ``"timeout"``, or ``"error"``.
        message: Optional human-readable detail.
    """

    session_id: str
    reason: Literal["shutdown", "timeout", "error"]
    message: str | None = None


@dataclass
class WsError:
    """JSON error frame sent from server to client.

    Args:
        session_id: Session this error relates to.
        error_code: Machine-readable error code (v1 enum).
        message: Human-readable description.
        fatal: If ``True``, the server closes the connection immediately after sending.
    """

    session_id: str
    error_code: Literal[
        "INVALID_AUDIO_FRAME",
        "UNKNOWN_MESSAGE_TYPE",
        "SESSION_NOT_FOUND",
        "BACKPRESSURE_DROP",
        "PROTOCOL_VIOLATION",
        "INTERNAL_ERROR",
    ]
    message: str
    fatal: bool = False


# ---------------------------------------------------------------------------
# Union types
# ---------------------------------------------------------------------------

ServerMessage = WsSessionCreated | WsRecognitionResult | WsSessionClosed | WsError
ClientTextMessage = WsControlCommand
