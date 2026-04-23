"""Dedicated factory for constructing ONNX-backed recognizers."""

import queue as _queue
from typing import Any, Protocol

import onnxruntime as rt

from src.ApplicationState import ApplicationState
from src.asr.ExecutionProviderManager import ExecutionProviderManager
from src.asr.ModelDefinitions import IModelDefinition
from src.asr.ModelLoader import load_model as load_asr_model
from src.asr.Recognizer import Recognizer
from src.asr.SessionOptionsFactory import SessionOptionsFactory


class IRecognizerFactory(Protocol):
    """Interface for creating recognizer instances on demand.

    Responsibilities:
    - Hide ONNX session construction details from startup and server orchestration.
    - Provide a single operation for building a ready-to-attach recognizer.
    """

    def create_recognizer(self) -> Recognizer:
        """Create and return a recognizer instance."""


class RecognizerFactory:
    """Build ``Recognizer`` instances from application config and model metadata.

    Responsibilities:
    - Resolve execution providers and session options.
    - Load the ASR model through the shared model loader.
    - Construct a recognizer with fresh input/output queues.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        asr_model: IModelDefinition,
        app_state: ApplicationState,
        verbose: bool = False,
    ) -> None:
        self._config = config
        self._asr_model = asr_model
        self._app_state = app_state
        self._verbose = verbose

    def create_recognizer(self) -> Recognizer:
        """Build the ONNX session and return a recognizer instance.

        Algorithm:
            1. Resolve execution providers from the configured environment.
            2. Build and configure ONNX session options for the detected GPU type.
            3. Load the ASR model with the shared loader.
            4. Construct a recognizer with fresh queues bound to the shared app state.

        Returns:
            A constructed recognizer ready to attach to ``RecognizerService``.
        """
        exec_mgr = ExecutionProviderManager(self._config)
        providers = exec_mgr.build_provider_list()

        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        gpu_type = exec_mgr.detect_gpu_type()
        factory = SessionOptionsFactory(self._config)
        factory.get_strategy(gpu_type).configure_session_options(sess_options)

        model = load_asr_model(self._asr_model, providers, sess_options)

        return Recognizer(
            model=model,
            input_queue=_queue.Queue(),
            output_queue=_queue.Queue(),
            sample_rate=self._config["audio"]["sample_rate"],
            app_state=self._app_state,
            verbose=self._verbose,
        )
