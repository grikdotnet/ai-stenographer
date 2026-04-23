"""WebSocket wire protocol message types for the STT client-server protocol."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Client -> Server
# ---------------------------------------------------------------------------


@dataclass
class WsAudioFrame:
    """Binary audio_chunk frame sent from client to server."""

    session_id: str
    chunk_id: int
    timestamp: float
    audio: npt.NDArray[np.float32]


@dataclass
class WsControlCommand:
    """JSON control_command frame sent from client to server.

    Args:
        session_id: Active session identifier.
        command: Command name defined by the protocol.
        timestamp: Client wall-clock time when the command was issued.
        model_name: Target model name for ``download_model`` commands.
        request_id: Optional correlation identifier echoed by server responses.
    """

    session_id: str
    command: Literal["close_session", "list_models", "download_model"]
    timestamp: float
    model_name: str | None = None
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Server -> Client
# ---------------------------------------------------------------------------


@dataclass
class WsSessionCreated:
    """JSON session_created frame sent from server to client after connect."""

    session_id: str
    protocol_version: str
    server_time: float


@dataclass
class WsRecognitionResult:
    """JSON recognition_result frame streamed from server to client."""

    session_id: str
    status: Literal["partial", "final"]
    text: str
    start_time: float
    end_time: float
    chunk_ids: list[int] = field(default_factory=list)
    utterance_id: int = 0
    token_confidences: list[float] = field(default_factory=list)


@dataclass
class WsSessionClosed:
    """JSON session_closed frame sent by server before graceful disconnect."""

    session_id: str
    reason: Literal["close_session", "timeout", "error"]
    message: str | None = None


@dataclass
class WsServerState:
    """JSON server_state frame broadcast by the server."""

    state: Literal["starting", "waiting_for_model", "running", "shutdown"]


@dataclass
class WsError:
    """JSON error frame sent from server to client."""

    session_id: str
    error_code: Literal[
        "INVALID_AUDIO_FRAME",
        "UNKNOWN_MESSAGE_TYPE",
        "SESSION_ID_MISMATCH",
        "BACKPRESSURE_DROP",
        "PROTOCOL_VIOLATION",
        "INTERNAL_ERROR",
        "MODEL_NOT_READY",
        "DOWNLOAD_IN_PROGRESS",
        "INVALID_MODEL_NAME",
    ]
    message: str
    fatal: bool = False
    request_id: str | None = None


@dataclass
class WsModelInfo:
    """Model metadata for download/status surfaces."""

    name: str
    display_name: str
    size_description: str
    status: Literal["downloaded", "missing", "downloading"]


@dataclass
class WsModelList:
    """JSON model_list response sent to one client."""

    models: list[WsModelInfo]
    request_id: str | None = None


@dataclass
class WsModelStatus:
    """JSON model_status response for download_model commands."""

    status: Literal["ready", "downloading"]
    request_id: str | None = None


@dataclass
class WsDownloadProgress:
    """JSON download_progress frame broadcast during model downloads."""

    model_name: str
    status: Literal["downloading", "complete", "error"]
    progress: float | None = None
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Union types
# ---------------------------------------------------------------------------


ServerMessage = (
    WsSessionCreated
    | WsRecognitionResult
    | WsSessionClosed
    | WsServerState
    | WsError
    | WsModelList
    | WsModelStatus
    | WsDownloadProgress
)
ClientTextMessage = WsControlCommand
