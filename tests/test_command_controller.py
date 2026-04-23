"""Tests for server-side control command execution."""

import json

from src.network.types import WsError, WsModelInfo, WsModelList, WsModelStatus
from src.server.CommandController import CommandController


class _StubModelCommandHandler:
    """Controllable fake for CommandController tests."""

    def __init__(self) -> None:
        self.models = [
            WsModelInfo(
                name="parakeet",
                display_name="Parakeet TDT 0.6B v3",
                size_description="1.25 GB",
                status="missing",
            )
        ]
        self.ready = False
        self.ensure_model_ready_calls: list[str] = []
        self.ensure_model_ready_error: Exception | None = None
        self.start_download_result = True
        self.start_download_calls: list[str] = []
        self.start_download_error: Exception | None = None

    def get_model_list(self) -> list[WsModelInfo]:
        return self.models

    def is_model_ready(self, model_name: str) -> bool:
        return self.ready

    def start_download(self, model_name: str) -> bool:
        self.start_download_calls.append(model_name)
        if self.start_download_error is not None:
            raise self.start_download_error
        return self.start_download_result

    def ensure_model_ready(self, model_name: str) -> None:
        self.ensure_model_ready_calls.append(model_name)
        if self.ensure_model_ready_error is not None:
            raise self.ensure_model_ready_error


def _command_payload(
    command: str,
    session_id: str = "session-1",
    request_id: str | None = None,
    model_name: str | None = None,
) -> str:
    payload: dict[str, object] = {
        "type": "control_command",
        "session_id": session_id,
        "command": command,
        "timestamp": 1000.0,
    }
    if request_id is not None:
        payload["request_id"] = request_id
    if model_name is not None:
        payload["model_name"] = model_name
    return json.dumps(payload)


class TestCommandController:
    """CommandController executes protocol commands and reports typed responses."""

    def test_invalid_json_returns_unknown_message_type_error(self) -> None:
        _, response = CommandController(_StubModelCommandHandler()).handle(
            "not json",
            session_id="session-1",
        )

        assert isinstance(response, WsError)
        assert response.error_code == "UNKNOWN_MESSAGE_TYPE"

    def test_close_session_returns_stop_signal(self) -> None:
        handler = _StubModelCommandHandler()
        controller = CommandController(handler)

        should_close, response = controller.handle(
            _command_payload("close_session"),
            session_id="session-1",
        )

        assert should_close is True
        assert response is None

    def test_list_models_returns_model_list_and_echoes_request_id(self) -> None:
        handler = _StubModelCommandHandler()
        controller = CommandController(handler)

        should_close, response = controller.handle(
            _command_payload("list_models", request_id="req-1"),
            session_id="session-1",
        )

        assert should_close is False
        assert isinstance(response, WsModelList)
        assert response.request_id == "req-1"
        assert response.models == handler.models

    def test_download_model_returns_ready_when_model_already_present(self) -> None:
        handler = _StubModelCommandHandler()
        handler.ready = True
        controller = CommandController(handler)

        should_close, response = controller.handle(
            _command_payload("download_model", request_id="req-2", model_name="parakeet"),
            session_id="session-1",
        )

        assert should_close is False
        assert isinstance(response, WsModelStatus)
        assert response.status == "ready"
        assert response.request_id == "req-2"
        assert handler.ensure_model_ready_calls == ["parakeet"]

    def test_download_model_returns_internal_error_when_ready_model_attach_fails(self) -> None:
        handler = _StubModelCommandHandler()
        handler.ready = True
        handler.ensure_model_ready_error = RuntimeError("attach failed")
        controller = CommandController(handler)

        should_close, response = controller.handle(
            _command_payload("download_model", request_id="req-2b", model_name="parakeet"),
            session_id="session-1",
        )

        assert should_close is False
        assert isinstance(response, WsError)
        assert response.error_code == "INTERNAL_ERROR"
        assert response.request_id == "req-2b"

    def test_download_model_returns_downloading_when_worker_started(self) -> None:
        handler = _StubModelCommandHandler()
        controller = CommandController(handler)

        should_close, response = controller.handle(
            _command_payload("download_model", request_id="req-3", model_name="parakeet"),
            session_id="session-1",
        )

        assert should_close is False
        assert isinstance(response, WsModelStatus)
        assert response.status == "downloading"
        assert response.request_id == "req-3"
        assert handler.start_download_calls == ["parakeet"]

    def test_download_model_returns_invalid_model_error(self) -> None:
        handler = _StubModelCommandHandler()
        handler.start_download_error = ValueError("bad model")
        controller = CommandController(handler)

        should_close, response = controller.handle(
            _command_payload("download_model", request_id="req-4", model_name="bad"),
            session_id="session-1",
        )

        assert should_close is False
        assert isinstance(response, WsError)
        assert response.error_code == "INVALID_MODEL_NAME"
        assert response.request_id == "req-4"

    def test_download_model_returns_download_in_progress_error(self) -> None:
        handler = _StubModelCommandHandler()
        handler.start_download_result = False
        controller = CommandController(handler)

        should_close, response = controller.handle(
            _command_payload("download_model", request_id="req-5", model_name="parakeet"),
            session_id="session-1",
        )

        assert should_close is False
        assert isinstance(response, WsError)
        assert response.error_code == "DOWNLOAD_IN_PROGRESS"
        assert response.request_id == "req-5"

    def test_session_mismatch_returns_session_id_mismatch(self) -> None:
        handler = _StubModelCommandHandler()
        controller = CommandController(handler)

        should_close, response = controller.handle(
            _command_payload("close_session", session_id="wrong-session"),
            session_id="session-1",
        )

        assert should_close is False
        assert isinstance(response, WsError)
        assert response.error_code == "SESSION_ID_MISMATCH"
