"""Tests for recognizer readiness coordination."""

from unittest.mock import MagicMock

from src.ApplicationState import ApplicationState
from src.server.RecognizerReadinessCoordinator import RecognizerReadinessCoordinator


class TestRecognizerReadinessCoordinator:
    """RecognizerReadinessCoordinator attaches models and publishes outcomes."""

    def test_ensure_model_ready_uses_factory_once_and_transitions_to_running(self) -> None:
        app_state = ApplicationState()
        app_state.set_state("waiting_for_model")
        recognizer = MagicMock()
        recognizer_factory = MagicMock()
        recognizer_factory.create_recognizer.return_value = recognizer
        recognizer_service = MagicMock()
        download_events = MagicMock()

        coordinator = RecognizerReadinessCoordinator(
            app_state=app_state,
            recognizer_service=recognizer_service,
            recognizer_factory=recognizer_factory,
            download_events=download_events,
        )

        coordinator.ensure_model_ready("parakeet")
        coordinator.ensure_model_ready("parakeet")

        recognizer_factory.create_recognizer.assert_called_once_with()
        recognizer_service.attach_recognizer.assert_called_once_with(recognizer)
        assert app_state.get_state() == "running"

    def test_attach_recognizer_marks_existing_recognizer_ready(self) -> None:
        app_state = ApplicationState()
        app_state.set_state("waiting_for_model")
        recognizer = MagicMock()
        recognizer_factory = MagicMock()
        recognizer_service = MagicMock()

        coordinator = RecognizerReadinessCoordinator(
            app_state=app_state,
            recognizer_service=recognizer_service,
            recognizer_factory=recognizer_factory,
            download_events=MagicMock(),
        )

        coordinator.attach_recognizer(recognizer)
        coordinator.ensure_model_ready("parakeet")

        recognizer_factory.create_recognizer.assert_not_called()
        recognizer_service.attach_recognizer.assert_called_once_with(recognizer)
        assert app_state.get_state() == "running"

    def test_download_success_attaches_recognizer_before_broadcast_complete(self) -> None:
        call_order: list[str] = []
        recognizer = MagicMock()
        recognizer_factory = MagicMock()
        recognizer_factory.create_recognizer.return_value = recognizer
        recognizer_service = MagicMock()
        recognizer_service.attach_recognizer.side_effect = (
            lambda _recognizer: call_order.append("attach")
        )
        download_events = MagicMock()
        download_events.on_complete.side_effect = lambda _model_name: call_order.append("complete")

        coordinator = RecognizerReadinessCoordinator(
            app_state=ApplicationState(),
            recognizer_service=recognizer_service,
            recognizer_factory=recognizer_factory,
            download_events=download_events,
        )

        coordinator.on_download_success("parakeet")

        assert call_order == ["attach", "complete"]
        download_events.on_error.assert_not_called()

    def test_download_success_broadcasts_error_when_attach_fails(self) -> None:
        error = RuntimeError("ort load failed")
        recognizer_factory = MagicMock()
        recognizer_factory.create_recognizer.side_effect = error
        download_events = MagicMock()

        coordinator = RecognizerReadinessCoordinator(
            app_state=ApplicationState(),
            recognizer_service=MagicMock(),
            recognizer_factory=recognizer_factory,
            download_events=download_events,
        )

        coordinator.on_download_success("parakeet")

        download_events.on_complete.assert_not_called()
        download_events.on_error.assert_called_once_with("parakeet", error)
