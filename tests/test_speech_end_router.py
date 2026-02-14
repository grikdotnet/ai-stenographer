import queue
import time

import numpy as np

from src.SpeechEndRouter import SpeechEndRouter
from src.types import (
    AudioSegment,
    RecognitionResult,
    RecognitionTextMessage,
    RecognizerAck,
    SpeechEndSignal,
)


class _FakeAppState:
    def __init__(self) -> None:
        self._observers = []

    def register_component_observer(self, observer) -> None:
        self._observers.append(observer)


def _segment(chunk_id: int, utterance_id: int = 1) -> AudioSegment:
    return AudioSegment(
        type="incremental",
        data=np.ones(160, dtype=np.float32),
        left_context=np.array([], dtype=np.float32),
        right_context=np.array([], dtype=np.float32),
        start_time=0.0,
        end_time=0.01,
        utterance_id=utterance_id,
        chunk_ids=[chunk_id],
    )


def _result(text: str = "hi") -> RecognitionResult:
    return RecognitionResult(
        text=text,
        start_time=0.0,
        end_time=0.01,
        chunk_ids=[1],
    )


def _build_router(
    speech_maxsize: int = 20,
    recognizer_maxsize: int = 20,
    output_maxsize: int = 20,
    matcher_maxsize: int = 20,
) -> tuple[SpeechEndRouter, queue.Queue, queue.Queue, queue.Queue, queue.Queue]:
    speech_queue = queue.Queue(maxsize=speech_maxsize)
    recognizer_queue = queue.Queue(maxsize=recognizer_maxsize)
    output_queue = queue.Queue(maxsize=output_maxsize)
    matcher_queue = queue.Queue(maxsize=matcher_maxsize)

    router = SpeechEndRouter(
        speech_queue=speech_queue,
        recognizer_queue=recognizer_queue,
        recognizer_output_queue=output_queue,
        matcher_queue=matcher_queue,
        app_state=_FakeAppState(),
    )
    return router, speech_queue, recognizer_queue, output_queue, matcher_queue


def _drain(router: SpeechEndRouter) -> None:
    # Directly trigger scheduling logic in unit tests (without background thread).
    router._drain_progress()


def test_one_in_flight_enforcement() -> None:
    router, speech_queue, recognizer_queue, output_queue, _ = _build_router()

    s1 = _segment(1)
    s2 = _segment(2)
    speech_queue.put(s1)
    speech_queue.put(s2)

    router._handle_speech_item(speech_queue.get_nowait())
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)

    first = recognizer_queue.get_nowait()
    assert first.chunk_ids == [1]
    assert router._in_flight_message_id == first.message_id
    assert len(router._audio_fifo) == 1

    output_queue.put(RecognizerAck(message_id=first.message_id or -1))
    router._handle_recognizer_output(output_queue.get_nowait())
    _drain(router)

    second = recognizer_queue.get_nowait()
    assert second.chunk_ids == [2]
    assert router._in_flight_message_id == second.message_id


def test_audio_segment_message_id_assignment() -> None:
    router, speech_queue, recognizer_queue, _, _ = _build_router()
    s1 = _segment(1)
    speech_queue.put(s1)
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    out = recognizer_queue.get_nowait()
    assert out.message_id == 1


def test_text_forwarding() -> None:
    router, speech_queue, recognizer_queue, _, matcher_queue = _build_router()
    speech_queue.put(_segment(1))
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    in_flight = recognizer_queue.get_nowait()

    msg = RecognitionTextMessage(result=_result("hello"), message_id=in_flight.message_id or -1)
    router._handle_recognizer_output(msg)
    forwarded = matcher_queue.get_nowait()
    assert isinstance(forwarded, RecognitionResult)
    assert forwarded.text == "hello"


def test_ack_handling_and_unblock() -> None:
    router, speech_queue, recognizer_queue, _, _ = _build_router()
    speech_queue.put(_segment(1))
    speech_queue.put(_segment(2))
    router._handle_speech_item(speech_queue.get_nowait())
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    first = recognizer_queue.get_nowait()

    router._handle_recognizer_output(RecognizerAck(message_id=first.message_id or -1))
    _drain(router)
    second = recognizer_queue.get_nowait()
    assert second.chunk_ids == [2]


def test_boundary_immediate_release() -> None:
    router, speech_queue, _, _, matcher_queue = _build_router()
    boundary = SpeechEndSignal(utterance_id=1, end_time=1.0)
    speech_queue.put(boundary)
    router._handle_speech_item(speech_queue.get_nowait())
    released = matcher_queue.get_nowait()
    assert isinstance(released, SpeechEndSignal)
    assert released.message_id == 1


def test_boundary_held_until_ack_and_buffer_drain() -> None:
    router, speech_queue, recognizer_queue, _, matcher_queue = _build_router()
    speech_queue.put(_segment(1))
    speech_queue.put(_segment(2))
    speech_queue.put(SpeechEndSignal(utterance_id=1, end_time=2.0))
    router._handle_speech_item(speech_queue.get_nowait())
    router._handle_speech_item(speech_queue.get_nowait())
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    first = recognizer_queue.get_nowait()

    router._handle_recognizer_output(RecognizerAck(message_id=first.message_id or -1))
    _drain(router)
    second = recognizer_queue.get_nowait()

    # Boundary should still be held while second audio is in flight.
    assert matcher_queue.empty()

    router._handle_recognizer_output(RecognizerAck(message_id=second.message_id or -1))
    _drain(router)
    released = matcher_queue.get_nowait()
    assert isinstance(released, SpeechEndSignal)


def test_sequential_message_id_assignment() -> None:
    router, speech_queue, _, _, matcher_queue = _build_router()
    speech_queue.put(_segment(1))
    speech_queue.put(SpeechEndSignal(utterance_id=1, end_time=1.0))

    router._handle_speech_item(speech_queue.get_nowait())
    assert router._audio_fifo[0].message_id == 1
    router._handle_speech_item(speech_queue.get_nowait())
    # boundary held as message_id=2 until inflight clears
    assert router._held_boundary is not None
    assert router._held_boundary.message_id == 2


def test_silence_path_ack_only() -> None:
    router, speech_queue, recognizer_queue, _, matcher_queue = _build_router()
    speech_queue.put(_segment(1))
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    seg = recognizer_queue.get_nowait()
    router._handle_recognizer_output(RecognizerAck(message_id=seg.message_id or -1))
    assert matcher_queue.empty()


def test_error_ack_path() -> None:
    router, speech_queue, recognizer_queue, _, _ = _build_router()
    speech_queue.put(_segment(1))
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    seg = recognizer_queue.get_nowait()
    router._handle_recognizer_output(RecognizerAck(message_id=seg.message_id or -1, ok=False, error="x"))
    assert router._in_flight_message_id is None


def test_unexpected_message_id_dropped() -> None:
    router, speech_queue, recognizer_queue, _, matcher_queue = _build_router()
    speech_queue.put(_segment(1))
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    _ = recognizer_queue.get_nowait()
    router._handle_recognizer_output(RecognizerAck(message_id=999))
    assert router.dropped_protocol_messages == 1
    assert matcher_queue.empty()


def test_start_stop_lifecycle() -> None:
    router, speech_queue, recognizer_queue, output_queue, matcher_queue = _build_router()
    router.start()
    speech_queue.put(_segment(1))
    time.sleep(0.15)
    seg = recognizer_queue.get_nowait()
    output_queue.put(RecognizerAck(message_id=seg.message_id or -1))
    time.sleep(0.15)
    router.stop()
    assert router.is_running is False
    assert matcher_queue.empty()


def test_queue_full_handling() -> None:
    router, speech_queue, recognizer_queue, _, matcher_queue = _build_router(recognizer_maxsize=1, matcher_maxsize=1)
    recognizer_queue.put(_segment(99))
    speech_queue.put(_segment(1))
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    assert router.dropped_recognizer_messages == 1

    matcher_queue.put(_result("existing"))
    router._put_matcher(SpeechEndSignal(utterance_id=1, end_time=1.0, message_id=2))
    assert router.dropped_matcher_messages == 1


def test_text_queue_full_drops_text_and_ack_and_unblocks() -> None:
    router, speech_queue, recognizer_queue, _, matcher_queue = _build_router(matcher_maxsize=1)

    # Fill matcher queue so recognizer text delivery fails.
    matcher_queue.put(_result("occupied"))

    speech_queue.put(_segment(1))
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    in_flight = recognizer_queue.get_nowait()
    message_id = in_flight.message_id or -1

    router._handle_recognizer_output(
        RecognitionTextMessage(result=_result("hello"), message_id=message_id)
    )
    assert router._in_flight_message_id is None
    assert message_id in router._skipped_ack_ids

    # ACK is intentionally dropped because text was dropped.
    router._handle_recognizer_output(RecognizerAck(message_id=message_id, ok=True))
    assert message_id not in router._skipped_ack_ids

    # Ensure dropped text was not queued.
    assert matcher_queue.qsize() == 1
    _ = matcher_queue.get_nowait()  # drain placeholder

    # Pipeline should continue with next segment.
    speech_queue.put(_segment(2))
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    next_segment = recognizer_queue.get_nowait()
    assert next_segment.chunk_ids == [2]
