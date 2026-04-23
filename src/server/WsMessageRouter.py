"""Thin classifier for incoming WebSocket frames."""

from dataclasses import dataclass

from src.network.types import WsError


@dataclass(frozen=True)
class RoutedAudioMessage:
    """A binary websocket frame that should be treated as audio."""

    raw: bytes


@dataclass(frozen=True)
class RoutedCommandMessage:
    """A text websocket frame that should be treated as a control command."""

    text: str


@dataclass(frozen=True)
class RoutedProtocolError:
    """A websocket frame that is invalid at the transport/message-kind layer."""

    error: WsError


def route_incoming_message(
    message: object,
    session_id: str,
) -> RoutedAudioMessage | RoutedCommandMessage | RoutedProtocolError:
    """Classify an incoming websocket frame without executing it.

    Args:
        message: Frame received from the websocket client.
        session_id: Active session identifier used for error responses.

    Returns:
        Routed frame classification.
    """
    if isinstance(message, bytes):
        return RoutedAudioMessage(raw=message)

    if isinstance(message, str):
        return RoutedCommandMessage(text=message)

    return RoutedProtocolError(
        error=WsError(
            session_id=session_id,
            error_code="UNKNOWN_MESSAGE_TYPE",
            message=f"Unsupported websocket frame type: {type(message).__name__}",
            fatal=False,
        )
    )
