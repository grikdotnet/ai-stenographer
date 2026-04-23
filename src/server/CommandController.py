"""Command controller for client-issued WebSocket control commands."""

import logging
from typing import TYPE_CHECKING

from src.network.codec import decode_client_message
from src.network.types import ServerMessage, WsControlCommand, WsError, WsModelList, WsModelStatus

if TYPE_CHECKING:
    from src.server.protocols import IModelCommandHandler

logger = logging.getLogger(__name__)


class CommandController:
    """Execute decoded client control commands against injected model services.

    Args:
        model_commands: Handler for model listing and background download actions.
    """

    def __init__(self, model_commands: "IModelCommandHandler") -> None:
        self._model_commands = model_commands

    def handle(self, text: str, session_id: str) -> tuple[bool, ServerMessage | None]:
        """Decode and execute a client command.

        Args:
            text: Raw JSON text frame.
            session_id: Active session identifier expected on the frame.

        Returns:
            Tuple of ``(should_close_session, response_message)``.
        """
        try:
            msg = decode_client_message(text)
        except ValueError as exc:
            logger.warning("CommandController[%s]: invalid client message: %s", session_id, exc)
            return False, WsError(
                session_id=session_id,
                error_code="UNKNOWN_MESSAGE_TYPE",
                message=str(exc),
                fatal=False,
            )

        if msg.session_id != session_id:
            logger.warning(
                "CommandController[%s]: session_id mismatch: got %r",
                session_id,
                msg.session_id,
            )
            return False, WsError(
                session_id=session_id,
                error_code="SESSION_ID_MISMATCH",
                message=f"session_id mismatch: expected {session_id!r}, got {msg.session_id!r}",
                fatal=False,
                request_id=msg.request_id,
            )

        if msg.command == "close_session":
            logger.info("CommandController[%s]: close_session command received", session_id)
            return True, None

        if msg.command == "list_models":
            return False, WsModelList(
                models=self._model_commands.get_model_list(),
                request_id=msg.request_id,
            )

        if msg.command == "download_model":
            return False, self._handle_download_model(msg, session_id)

        return False, WsError(
            session_id=session_id,
            error_code="UNKNOWN_MESSAGE_TYPE",
            message=f"Unsupported control command: {msg.command!r}",
            fatal=False,
            request_id=msg.request_id,
        )

    def _handle_download_model(
        self,
        msg: WsControlCommand,
        session_id: str,
    ) -> ServerMessage:
        """Process a ``download_model`` command.

        Args:
            msg: Decoded command frame.
            session_id: Active session identifier used for error responses.

        Returns:
            Immediate command response.
        """
        assert msg.model_name, "download_model requires model_name after codec validation"
        model_name = msg.model_name

        try:
            if self._model_commands.is_model_ready(model_name):
                self._model_commands.ensure_model_ready(model_name)
                return WsModelStatus(status="ready", request_id=msg.request_id)

            started = self._model_commands.start_download(model_name)
        except ValueError as exc:
            return WsError(
                session_id=session_id,
                error_code="INVALID_MODEL_NAME",
                message=str(exc),
                fatal=False,
                request_id=msg.request_id,
            )
        except Exception as exc:
            logger.exception(
                "CommandController[%s]: download_model failed for %s",
                session_id,
                model_name,
            )
            return WsError(
                session_id=session_id,
                error_code="INTERNAL_ERROR",
                message=str(exc),
                fatal=False,
                request_id=msg.request_id,
            )

        if not started:
            return WsError(
                session_id=session_id,
                error_code="DOWNLOAD_IN_PROGRESS",
                message="A download is already in progress",
                fatal=False,
                request_id=msg.request_id,
            )

        return WsModelStatus(status="downloading", request_id=msg.request_id)
