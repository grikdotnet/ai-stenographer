import queue
import time
from unittest.mock import Mock

import numpy as np

from src.asr.Recognizer import Recognizer
from src.types import AudioSegment, RecognitionTextMessage, RecognizerAck


class _FakeAppState:
    def register_component_observer(self, _observer) -> None:
        pass


def _segment(message_id: int = 1) -> AudioSegment:
    return AudioSegment(
        type="incremental",
        data=np.ones(320, dtype=np.float32),
        left_context=np.array([], dtype=np.float32),
        right_context=np.array([], dtype=np.float32),
        start_time=0.0,
        end_time=0.02,
        utterance_id=1,
        chunk_ids=[1],
        message_id=message_id,
    )


def _recognizer(model: Mock) -> tuple[Recognizer, queue.Queue, queue.Queue]:
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    recognizer = Recognizer(
        input_queue=input_queue,
        output_queue=output_queue,
        model=model,
        app_state=_FakeAppState(),
    )
    return recognizer, input_queue, output_queue


def test_emits_text_then_terminal_ack() -> None:
    model = Mock()
    model.recognize.return_value = Mock(
        text="hello",
        tokens=[" hello"],
        timestamps=[0.0],
        logprobs=None,
    )
    recognizer, input_queue, output_queue = _recognizer(model)
    input_queue.put(_segment(message_id=7))

    recognizer.start()
    time.sleep(0.15)
    recognizer.stop()

    first = output_queue.get_nowait()
    second = output_queue.get_nowait()
    assert isinstance(first, RecognitionTextMessage)
    assert first.message_id == 7
    assert isinstance(second, RecognizerAck)
    assert second.message_id == 7
    assert second.ok is True


def test_silence_emits_ack_only() -> None:
    model = Mock()
    model.recognize.return_value = Mock(
        text="   ",
        tokens=[],
        timestamps=[],
    )
    recognizer, input_queue, output_queue = _recognizer(model)
    input_queue.put(_segment(message_id=3))

    recognizer.start()
    time.sleep(0.15)
    recognizer.stop()

    msg = output_queue.get_nowait()
    assert isinstance(msg, RecognizerAck)
    assert msg.message_id == 3
    assert output_queue.empty()


def test_emits_text_with_populated_confidence() -> None:
    model = Mock()
    model.recognize.return_value = Mock(
        text="hello world",
        tokens=[" hello", " world"],
        timestamps=[0.0, 0.3],
        logprobs=[-0.1, -0.2],
    )
    recognizer, input_queue, output_queue = _recognizer(model)
    input_queue.put(_segment(message_id=9))

    recognizer.start()
    time.sleep(0.15)
    recognizer.stop()

    first = output_queue.get_nowait()
    second = output_queue.get_nowait()
    assert isinstance(first, RecognitionTextMessage)
    assert first.result.token_confidences != []
    assert first.result.confidence > 0.0
    assert isinstance(second, RecognizerAck)
    assert second.ok is True
    assert second.message_id == first.message_id


def test_error_emits_failed_ack() -> None:
    model = Mock()
    model.recognize.side_effect = RuntimeError("boom")
    recognizer, input_queue, output_queue = _recognizer(model)
    input_queue.put(_segment(message_id=5))

    recognizer.start()
    time.sleep(0.15)
    recognizer.stop()

    msg = output_queue.get_nowait()
    assert isinstance(msg, RecognizerAck)
    assert msg.message_id == 5
    assert msg.ok is False
    assert "boom" in (msg.error or "")
