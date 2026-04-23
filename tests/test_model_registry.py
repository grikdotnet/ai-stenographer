"""Tests for shared model definitions and registry behavior."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.PathResolver import ResolvedPaths
from src.asr.ModelDefinitions import ParakeetAsrModel, SileroVadModel
from src.downloader.ModelDownloader import ModelDownloader
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


class TestModelDefinitions:
    def test_parakeet_model_path_is_relative_to_models_dir(self) -> None:
        assert ParakeetAsrModel(_MODELS_DIR).get_model_path() == _MODELS_DIR / "parakeet"

    def test_silero_model_path_is_relative_to_models_dir(self) -> None:
        expected = _MODELS_DIR / "silero_vad" / "silero_vad.onnx"
        assert SileroVadModel(_MODELS_DIR).get_model_path() == expected

    def test_parakeet_validate_delegates_to_shared_validator(self) -> None:
        with patch("src.asr.ModelDefinitions.validate_parakeet", return_value=True) as mock_validate:
            assert ParakeetAsrModel(_MODELS_DIR).validate() is True

        mock_validate.assert_called_once_with(_MODELS_DIR)

    def test_silero_validate_checks_expected_file_contract(self) -> None:
        model = SileroVadModel(_MODELS_DIR)
        model_path = MagicMock()
        model_path.exists.return_value = True
        model_path.stat.return_value.st_size = 1

        with patch.object(model, "get_model_path", return_value=model_path):
            assert model.validate() is True

    def test_to_ws_model_info_uses_status_override_when_present(self) -> None:
        model = ParakeetAsrModel(_MODELS_DIR)

        info = model.to_ws_model_info(status_override="downloading")

        assert info.name == "parakeet"
        assert info.status == "downloading"


class TestModelRegistry:
    def test_get_asr_model_returns_parakeet_model(self) -> None:
        assert isinstance(ModelRegistry(_PATHS).get_asr_model(), ParakeetAsrModel)

    def test_get_vad_model_returns_silero_model(self) -> None:
        assert isinstance(ModelRegistry(_PATHS).get_vad_model(), SileroVadModel)

    def test_get_model_returns_concrete_model_by_name(self) -> None:
        registry = ModelRegistry(_PATHS)

        assert isinstance(registry.get_model("parakeet"), ParakeetAsrModel)
        assert isinstance(registry.get_model("silero_vad"), SileroVadModel)

    def test_get_model_list_is_generated_from_registered_models(self) -> None:
        registry = ModelRegistry(_PATHS)

        models = registry.get_model_list()

        assert [model.name for model in models] == ["parakeet"]

    def test_get_downloadable_models_returns_model_definitions(self) -> None:
        registry = ModelRegistry(_PATHS)

        models = registry.get_downloadable_models()

        assert [model.name for model in models] == ["parakeet"]
        assert isinstance(models[0], ParakeetAsrModel)

    def test_get_missing_models_returns_missing_downloadable_models(self) -> None:
        with patch("src.asr.ModelDefinitions.validate_parakeet", return_value=False):
            assert ModelRegistry(_PATHS).get_missing_models() == ["parakeet"]

    def test_compat_get_missing_models_preserves_legacy_behavior(self) -> None:
        with patch("src.asr.ModelDefinitions.validate_parakeet", return_value=False):
            assert ModelRegistry.get_missing_models_for_dir(_MODELS_DIR) == ["parakeet"]

    def test_compat_validate_model_returns_true_for_valid_silero(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "silero_vad" / "silero_vad.onnx"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("vad")

        assert ModelRegistry.validate_model("silero_vad", model_dir) is True


class TestModelDownloaderFacade:
    def test_validate_parakeet_delegates_to_module_function(self) -> None:
        downloader = ModelDownloader(_MODELS_DIR)
        with patch("src.downloader.ModelDownloader.validate_parakeet", return_value=True) as mock_validate:
            assert downloader.validate_parakeet() is True
        mock_validate.assert_called_once_with(_MODELS_DIR)

    def test_cleanup_partial_files_delegates_to_module_function(self) -> None:
        with patch("src.downloader.ModelDownloader.cleanup_partial_files") as mock_cleanup:
            ModelDownloader.cleanup_partial_files(_MODELS_DIR)
        mock_cleanup.assert_called_once_with(_MODELS_DIR)
