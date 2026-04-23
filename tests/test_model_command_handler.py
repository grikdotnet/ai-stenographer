"""Tests for server-side model command handling."""

from unittest.mock import MagicMock, patch
import pytest

from src.server.ModelCommandHandler import ModelCommandHandler


class _Readiness:
    """Fake model readiness coordinator for command handler tests."""

    def __init__(self) -> None:
        self.ensure_calls: list[str] = []
        self.download_success_calls: list[str] = []

    def ensure_model_ready(self, model_name: str) -> None:
        self.ensure_calls.append(model_name)

    def on_download_success(self, model_name: str) -> None:
        self.download_success_calls.append(model_name)


class _DownloadEvents:
    """Fake download event publisher for command handler tests."""

    def __init__(self) -> None:
        self.progress_calls: list[tuple[str, float, int, int]] = []
        self.error_calls: list[tuple[str, Exception]] = []

    def on_progress(
        self,
        model_name: str,
        progress: float,
        downloaded_bytes: int,
        total_bytes: int,
    ) -> None:
        self.progress_calls.append((model_name, progress, downloaded_bytes, total_bytes))

    def on_complete(self, model_name: str) -> None:
        del model_name

    def on_error(self, model_name: str, exc: Exception) -> None:
        self.error_calls.append((model_name, exc))


def _make_handler(registry) -> tuple[ModelCommandHandler, _Readiness, _DownloadEvents]:
    readiness = _Readiness()
    download_events = _DownloadEvents()
    return (
        ModelCommandHandler(
            model_registry=registry,
            model_readiness=readiness,
            download_events=download_events,
        ),
        readiness,
        download_events,
    )


class TestModelCommandHandler:
    def test_constructor_rejects_positional_model_registry(self) -> None:
        with pytest.raises(TypeError):
            ModelCommandHandler(MagicMock())

    def test_get_model_list_marks_current_download_generic_model(self) -> None:
        downloadable_model = MagicMock()
        downloadable_model.name = "custom_model"
        downloadable_model.is_ready.return_value = False
        downloadable_model.to_ws_model_info.return_value = MagicMock(name="ws-info")

        registry = MagicMock()
        registry.get_downloadable_models.return_value = [downloadable_model]

        with patch("src.server.ModelCommandHandler.DownloadWorker") as worker_cls:
            worker = worker_cls.return_value
            worker.current_model_name.return_value = "custom_model"

            handler, _, _ = _make_handler(registry)
            result = handler.get_model_list()

        assert result == [downloadable_model.to_ws_model_info.return_value]
        downloadable_model.to_ws_model_info.assert_called_once_with(
            status_override="downloading"
        )

    def test_get_model_list_does_not_override_completed_or_other_downloads(self) -> None:
        downloadable_model = MagicMock()
        downloadable_model.name = "parakeet"
        downloadable_model.is_ready.return_value = True
        downloadable_model.to_ws_model_info.return_value = MagicMock(name="ws-info")

        registry = MagicMock()
        registry.get_downloadable_models.return_value = [downloadable_model]

        with patch("src.server.ModelCommandHandler.DownloadWorker") as worker_cls:
            worker = worker_cls.return_value
            worker.current_model_name.return_value = "some_other_model"

            handler, _, _ = _make_handler(registry)
            handler.get_model_list()

        downloadable_model.to_ws_model_info.assert_called_once_with(status_override=None)

    def test_ensure_model_ready_delegates_to_readiness_when_model_exists(self) -> None:
        model = MagicMock()
        model.is_ready.return_value = True
        registry = MagicMock()
        registry.get_model.return_value = model
        handler, readiness, _ = _make_handler(registry)

        handler.ensure_model_ready("parakeet")

        assert readiness.ensure_calls == ["parakeet"]

    def test_start_download_wires_worker_callbacks_to_services(self) -> None:
        model = MagicMock()
        model.is_downloadable.return_value = True
        registry = MagicMock()
        registry.get_model.return_value = model

        with patch("src.server.ModelCommandHandler.DownloadWorker") as worker_cls:
            worker = worker_cls.return_value
            worker.start.return_value = True
            handler, readiness, download_events = _make_handler(registry)

            assert handler.start_download("parakeet") is True

        kwargs = worker.start.call_args.kwargs
        kwargs["progress_callback"](0.5, 100, 200)
        kwargs["on_success"]("parakeet")
        error = RuntimeError("network")
        kwargs["on_error"]("parakeet", error)

        assert download_events.progress_calls == [("parakeet", 0.5, 100, 200)]
        assert readiness.download_success_calls == ["parakeet"]
        assert download_events.error_calls == [("parakeet", error)]
