"""Top-level shared registry for model definitions and lookup."""

from pathlib import Path

from src.PathResolver import ResolvedPaths
from src.asr.ModelDefinitions import IModelDefinition, ParakeetAsrModel, SileroVadModel
from src.network.types import WsModelInfo


class ModelRegistry:
    """Registry of concrete shared models resolved for one application root.

    Responsibilities:
    - Construct model-specific objects from ``ResolvedPaths``.
    - Provide lookup by semantic role (ASR/VAD) or model name.
    - Generate model-list and missing-model views from registered models.
    - Offer temporary compatibility wrappers for legacy path-based callers.
    """

    def __init__(self, paths: ResolvedPaths) -> None:
        self._models_dir = paths.models_dir
        self._models = self._build_models(paths.models_dir)

    def get_asr_model(self) -> ParakeetAsrModel:
        """Return the concrete ASR model definition."""
        return self.get_model("parakeet")  # type: ignore[return-value]

    def get_vad_model(self) -> SileroVadModel:
        """Return the concrete VAD model definition."""
        return self.get_model("silero_vad")  # type: ignore[return-value]

    def get_model(self, model_name: str) -> IModelDefinition:
        """Return the registered model definition by name.

        Args:
            model_name: Stable model identifier.

        Returns:
            Matching model definition.

        Raises:
            ValueError: If the model name is not registered.
        """
        try:
            return self._models[model_name]
        except KeyError as exc:
            raise ValueError(f"Unknown model name: {model_name}") from exc

    def is_known(self, model_name: str) -> bool:
        """Return whether the given model name is registered."""
        return model_name in self._models

    def is_ready(self, model_name: str) -> bool:
        """Return whether the given model validates successfully."""
        return self.get_model(model_name).is_ready()

    def get_downloadable_models(self) -> list[IModelDefinition]:
        """Return registered models that support shared download flows."""
        return [
            model
            for model in self._models.values()
            if model.is_downloadable()
        ]

    def get_model_list(self) -> list[WsModelInfo]:
        """Return wire metadata for registered downloadable models."""
        return [
            model.to_ws_model_info()
            for model in self.get_downloadable_models()
        ]

    def get_missing_models(self) -> list[str]:
        """Return names of registered downloadable models that are not ready."""
        return [
            model.name
            for model in self.get_downloadable_models()
            if not model.is_ready()
        ]

    @staticmethod
    def get_missing_models_for_dir(models_dir: Path) -> list[str]:
        """Compatibility wrapper for legacy callers that only know models_dir."""
        return ModelRegistry._from_models_dir(models_dir).get_missing_models()

    @staticmethod
    def validate_model(model_name: str, model_dir: Path) -> bool:
        """Compatibility wrapper for legacy model validation callers."""
        return ModelRegistry._from_models_dir(model_dir).get_model(model_name).validate()

    @staticmethod
    def _from_models_dir(models_dir: Path) -> "ModelRegistry":
        """Create a compatibility registry from a raw models directory only.

        Uses ``__new__`` to bypass ``__init__`` so legacy callers do not need a
        synthetic ``ResolvedPaths``. This helper must stay in sync with
        ``__init__`` if that initializer ever starts populating additional
        required instance state.
        """
        registry = ModelRegistry.__new__(ModelRegistry)
        registry._models_dir = models_dir
        registry._models = registry._build_models(models_dir)
        return registry

    @staticmethod
    def _build_models(models_dir: Path) -> dict[str, IModelDefinition]:
        """Construct the concrete model set for the given models root."""
        return {
            "parakeet": ParakeetAsrModel(models_dir),
            "silero_vad": SileroVadModel(models_dir),
        }
