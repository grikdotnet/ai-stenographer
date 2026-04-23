"""Tests for ModelManager model lookup delegation."""

from pathlib import Path
from unittest.mock import MagicMock

from src.asr.ModelManager import ModelManager
from src.PathResolver import ResolvedPaths
from src.asr.ModelRegistry import ModelRegistry


_MODELS_DIR = Path("/fake/models")
_PATHS = ResolvedPaths(
    app_dir=Path("/fake"),
    internal_dir=Path("/fake"),
    root_dir=Path("/fake"),
    models_dir=_MODELS_DIR,
    config_dir=Path("/fake/config"),
    assets_dir=Path("/fake"),
    logs_dir=Path("/fake/logs"),
    environment="development",
)


class TestModelExists:
    def test_returns_true_when_asr_model_is_ready(self) -> None:
        registry = MagicMock(spec=ModelRegistry)
        registry.get_asr_model.return_value.is_ready.return_value = True

        assert ModelManager(registry).model_exists() is True

    def test_returns_false_when_asr_model_is_not_ready(self) -> None:
        registry = MagicMock(spec=ModelRegistry)
        registry.get_asr_model.return_value.is_ready.return_value = False

        assert ModelManager(registry).model_exists() is False

    def test_model_exists_delegates_to_asr_model(self) -> None:
        registry = MagicMock(spec=ModelRegistry)
        model = registry.get_asr_model.return_value
        model.is_ready.return_value = True

        ModelManager(registry).model_exists()

        registry.get_asr_model.assert_called_once_with()
        model.is_ready.assert_called_once_with()

    def test_validate_model_delegates_to_registry_model(self) -> None:
        registry = ModelRegistry(_PATHS)
        manager = ModelManager(registry)
        model = MagicMock()
        model.validate.return_value = True
        registry.get_model = MagicMock(return_value=model)  # type: ignore[method-assign]

        assert manager.validate_model("parakeet") is True

        registry.get_model.assert_called_once_with("parakeet")
        model.validate.assert_called_once_with()
