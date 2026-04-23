"""Tests for incoming WebSocket frame routing."""

from src.server.WsMessageRouter import (
    RoutedAudioMessage,
    RoutedCommandMessage,
    RoutedProtocolError,
    route_incoming_message,
)


class TestWsMessageRouter:
    """Incoming websocket frames are classified without executing them."""

    def test_binary_frame_routes_to_audio(self) -> None:
        routed = route_incoming_message(b"audio-bytes", session_id="session-1")

        assert isinstance(routed, RoutedAudioMessage)
        assert routed.raw == b"audio-bytes"

    def test_text_frame_routes_to_command(self) -> None:
        routed = route_incoming_message('{"type":"control_command"}', session_id="session-1")

        assert isinstance(routed, RoutedCommandMessage)
        assert routed.text == '{"type":"control_command"}'

    def test_unsupported_frame_type_routes_to_protocol_error(self) -> None:
        routed = route_incoming_message(42, session_id="session-1")

        assert isinstance(routed, RoutedProtocolError)
        assert routed.error.error_code == "UNKNOWN_MESSAGE_TYPE"
