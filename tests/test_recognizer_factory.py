"""Tests for dedicated recognizer construction."""

import queue
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.asr.ModelDefinitions import IModelDefinition
from src.asr.RecognizerFactory import RecognizerFactory


_CONFIG = {
    "audio": {
        "sample_rate": 16000,
    },
}


class TestRecognizerFactory:
    def test_create_recognizer_builds_runtime_objects_from_constructor_dependencies(self) -> None:
        model_definition = MagicMock(spec=IModelDefinition)
        model_definition.get_model_path.return_value = Path("/fake/models/parakeet")
        app_state = MagicMock()
        recognizer = MagicMock()
        exec_manager = MagicMock()
        exec_manager.build_provider_list.return_value = ["CPUExecutionProvider"]
        exec_manager.detect_gpu_type.return_value = "cpu"
        session_options = MagicMock()
        strategy = MagicMock()
        session_options_factory = MagicMock()
        session_options_factory.get_strategy.return_value = strategy

        factory = RecognizerFactory(
            config=_CONFIG,
            asr_model=model_definition,
            app_state=app_state,
            verbose=True,
        )

        with (
            patch(
                "src.asr.RecognizerFactory.ExecutionProviderManager",
                return_value=exec_manager,
            ),
            patch(
                "src.asr.RecognizerFactory.rt.SessionOptions",
                return_value=session_options,
            ),
            patch("src.asr.RecognizerFactory.rt.GraphOptimizationLevel"),
            patch(
                "src.asr.RecognizerFactory.SessionOptionsFactory",
                return_value=session_options_factory,
            ),
            patch(
                "src.asr.RecognizerFactory.load_asr_model",
                return_value=MagicMock(),
            ) as load_model,
            patch(
                "src.asr.RecognizerFactory.Recognizer",
                return_value=recognizer,
            ) as recognizer_cls,
        ):
            created = factory.create_recognizer()

        assert created is recognizer
        load_model.assert_called_once_with(
            model_definition,
            ["CPUExecutionProvider"],
            session_options,
        )
        recognizer_cls.assert_called_once()
        _, kwargs = recognizer_cls.call_args
        assert isinstance(kwargs["input_queue"], queue.Queue)
        assert isinstance(kwargs["output_queue"], queue.Queue)
        assert kwargs["sample_rate"] == 16000
        assert kwargs["app_state"] is app_state
        assert kwargs["verbose"] is True
        strategy.configure_session_options.assert_called_once_with(session_options)
