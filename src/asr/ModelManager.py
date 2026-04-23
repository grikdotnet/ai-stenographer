"""Model manager facade for shared registry-backed model access."""

from src.asr.ModelDefinitions import IModelDefinition, ParakeetAsrModel, SileroVadModel
from src.asr.ModelRegistry import ModelRegistry


class ModelManager:
    """Expose the server startup model queries needed by higher layers.

    Responsibilities:
    - Ask the registry whether the primary ASR model is ready.
    - Delegate per-model validation to the registry.
    - Expose the shared registry and well-known model objects to callers
      that need richer model behavior.
    """

    def __init__(self, model_registry: ModelRegistry) -> None:
        self._model_registry = model_registry

    @property
    def model_registry(self) -> ModelRegistry:
        """Return the backing shared model registry."""
        return self._model_registry

    def get_asr_model(self) -> ParakeetAsrModel:
        """Return the concrete ASR model definition."""
        return self._model_registry.get_asr_model()

    def get_vad_model(self) -> SileroVadModel:
        """Return the concrete VAD model definition."""
        return self._model_registry.get_vad_model()

    def get_model(self, model_name: str) -> IModelDefinition:
        """Return a model definition by stable name."""
        return self._model_registry.get_model(model_name)

    def model_exists(self) -> bool:
        """Return whether the primary ASR model is ready for inference."""
        return self.get_asr_model().is_ready()

    def validate_model(self, model_name: str) -> bool:
        """Validate the requested model through the shared registry."""
        return self.get_model(model_name).validate()
