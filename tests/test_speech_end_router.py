import queue
import time

import numpy as np

from src.SpeechEndRouter import SpeechEndRouter
from src.types import (
    AudioSegment,
    RecognitionResult,
    RecognitionTextMessage,
    RecognizerAck,
    RecognizerFreeSignal,
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
    control_maxsize: int = 20,
) -> tuple[SpeechEndRouter, queue.Queue, queue.Queue, queue.Queue, queue.Queue, queue.Queue]:
    speech_queue = queue.Queue(maxsize=speech_maxsize)
    recognizer_queue = queue.Queue(maxsize=recognizer_maxsize)
    output_queue = queue.Queue(maxsize=output_maxsize)
    matcher_queue = queue.Queue(maxsize=matcher_maxsize)
    control_queue = queue.Queue(maxsize=control_maxsize)

    router = SpeechEndRouter(
        speech_queue=speech_queue,
        recognizer_queue=recognizer_queue,
        recognizer_output_queue=output_queue,
        matcher_queue=matcher_queue,
        app_state=_FakeAppState(),
        control_queue=control_queue,
    )
    return router, speech_queue, recognizer_queue, output_queue, matcher_queue, control_queue


def _drain(router: SpeechEndRouter) -> None:
    # Directly trigger scheduling logic in unit tests (without background thread).
    router._drain_progress()


def test_one_in_flight() -> None:
    """Only one audio segment is dispatched to the recognizer at a time."""
    router, speech_queue, recognizer_queue, output_queue, _, _ = _build_router()

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
    """First dispatched segment receives message_id=1."""
    router, speech_queue, recognizer_queue, _, _, _ = _build_router()
    s1 = _segment(1)
    speech_queue.put(s1)
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    out = recognizer_queue.get_nowait()
    assert out.message_id == 1


def test_text_forwarding() -> None:
    """RecognitionTextMessage is forwarded as RecognitionResult to the matcher queue."""
    router, speech_queue, recognizer_queue, _, matcher_queue, _ = _build_router()
    speech_queue.put(_segment(1))
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    in_flight = recognizer_queue.get_nowait()

    result = _result("hello")
    msg = RecognitionTextMessage(result, message_id=in_flight.message_id or -1)
    router._handle_recognizer_output(msg)
    forwarded = matcher_queue.get_nowait()
    assert forwarded == result


def test_ack_handling_and_unblock() -> None:
    """ACK from recognizer unblocks dispatch of the next queued segment."""
    router, speech_queue, recognizer_queue, _, _, _ = _build_router()
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
    """SpeechEndSignal with no pending audio is forwarded to matcher immediately."""
    router, speech_queue, _, _, matcher_queue, _ = _build_router()
    boundary = SpeechEndSignal(utterance_id=1, end_time=1.0)
    speech_queue.put(boundary)
    router._handle_speech_item(speech_queue.get_nowait())
    released = matcher_queue.get_nowait()
    assert isinstance(released, SpeechEndSignal)
    assert released.message_id == 1


def test_boundary_waits_for_same_utterance_fifo_audio() -> None:
    """SpeechEndSignal is held in Router until all same-utterance audio ACKs are received."""
    router, speech_queue, recognizer_queue, _, matcher_queue, _ = _build_router()
    speech_queue.put(_segment(1))
    speech_queue.put(_segment(2))
    speech_queue.put(SpeechEndSignal(utterance_id=1, end_time=2.0))
    router._handle_speech_item(speech_queue.get_nowait())
    router._handle_speech_item(speech_queue.get_nowait())
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)

    first = recognizer_queue.get_nowait()
    assert first.chunk_ids == [1]
    assert len(router._audio_fifo) == 1

    router._handle_recognizer_output(RecognizerAck(message_id=first.message_id))
    _drain(router)

    second = recognizer_queue.get_nowait()
    assert second.chunk_ids == [2]
    assert second.utterance_id == 1
    assert router._held_boundary is not None
    assert matcher_queue.empty()

    router._handle_recognizer_output(RecognizerAck(message_id=second.message_id))
    _drain(router)

    released = matcher_queue.get_nowait()
    assert isinstance(released, SpeechEndSignal)
    assert released.utterance_id == 1
    assert released.end_time == 2.0
    assert router._held_boundary is None
    assert len(router._audio_fifo) == 0
    assert recognizer_queue.empty()


def test_boundary_released_before_next_utterance_fifo_audio() -> None:
    """SpeechEndSignal is released before dispatching audio from the next utterance."""
    router, speech_queue, recognizer_queue, _, matcher_queue, _ = _build_router()

    speech_queue.put(_segment(1, utterance_id=1))
    speech_queue.put(SpeechEndSignal(utterance_id=1, end_time=1.0))
    speech_queue.put(_segment(2, utterance_id=2))

    router._handle_speech_item(speech_queue.get_nowait())
    router._handle_speech_item(speech_queue.get_nowait())
    router._handle_speech_item(speech_queue.get_nowait())

    # one drain is enough for “in-flight” state, stop before ACK.
    _drain(router)
    first = recognizer_queue.get_nowait()
    assert first.utterance_id == 1

    router._handle_recognizer_output(RecognizerAck(message_id=first.message_id or -1))
    _drain(router)

    released = matcher_queue.get_nowait()
    assert isinstance(released, SpeechEndSignal)
    assert released.utterance_id == 1

    _drain(router)
    second = recognizer_queue.get_nowait()
    assert second.utterance_id == 2


def test_should_not_release_boundary_while_ack_in_flight() -> None:
    """Boundary remains held while recognizer ACK for prior audio is pending."""
    router, _, _, _, _, _ = _build_router()
    router._held_boundary = SpeechEndSignal(utterance_id=1, end_time=1.0, message_id=2)
    router._in_flight_message_id = 1

    assert router._should_release_held_boundary() is False


def test_ack_only() -> None:
    """Test error case: ACK from Recognizer without a result does not affect the matcher queue."""
    router, speech_queue, recognizer_queue, _, queue_to_matcher, _ = _build_router()
    router._handle_speech_item(_segment(1))
    _drain(router)
    seg = recognizer_queue.get_nowait()
    router._handle_recognizer_output(RecognizerAck(message_id=seg.message_id or -1))
    assert queue_to_matcher.empty()


def test_error_ack_path() -> None:
    """Failed ACK (ok=False) clears in-flight state so the next segment can be dispatched."""
    router, speech_queue, recognizer_queue, _, _, _ = _build_router()
    speech_queue.put(_segment(1))
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    seg = recognizer_queue.get_nowait()
    router._handle_recognizer_output(RecognizerAck(message_id=seg.message_id or -1, ok=False, error="x"))
    assert router._in_flight_message_id is None


def test_unexpected_message_id_dropped() -> None:
    """ACK with an unknown message_id is counted as a dropped protocol message."""
    router, speech_queue, recognizer_queue, _, matcher_queue, _ = _build_router()
    speech_queue.put(_segment(1))
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    _ = recognizer_queue.get_nowait()
    router._handle_recognizer_output(RecognizerAck(message_id=999))
    assert router.dropped_protocol_messages == 1
    assert matcher_queue.empty()


def test_ack_emits_control_signal() -> None:
    """Terminal ACK emits signal with correct seq/message_id/utterance_id."""
    router, speech_queue, recognizer_queue, output_queue, _, control_queue = _build_router(control_maxsize=10)

    seg = _segment(chunk_id=1, utterance_id=42)
    speech_queue.put(seg)
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)
    dispatched = recognizer_queue.get_nowait()
    message_id = dispatched.message_id or -1

    output_queue.put(RecognizerAck(message_id=message_id, ok=True))
    router._handle_recognizer_output(output_queue.get_nowait())

    signal = control_queue.get_nowait()
    assert isinstance(signal, RecognizerFreeSignal)
    assert signal.seq == 0
    assert signal.message_id == message_id
    assert signal.utterance_id == 42


def test_seq_monotonic_across_acks() -> None:
    """Seq increments across multiple ACKs."""
    router, speech_queue, recognizer_queue, output_queue, _, control_queue = _build_router(control_maxsize=10)

    for i in range(3):
        seg = _segment(chunk_id=i, utterance_id=1)
        speech_queue.put(seg)
        router._handle_speech_item(speech_queue.get_nowait())
        _drain(router)
        dispatched = recognizer_queue.get_nowait()
        message_id = dispatched.message_id or -1
        output_queue.put(RecognizerAck(message_id=message_id, ok=True))
        router._handle_recognizer_output(output_queue.get_nowait())
        _drain(router)

    signals = []
    while not control_queue.empty():
        signals.append(control_queue.get_nowait())

    assert len(signals) == 3
    assert signals[0].seq == 0
    assert signals[1].seq == 1
    assert signals[2].seq == 2


def test_in_flight_utterance_id_hygiene() -> None:
    """Check that in-flight utterance_id and message_id are set when a segment 
        is sent to Recognizer, and cleared when ACK is received from Recognizer."""
    router, speech_queue, recognizer_queue, _, _, _ = _build_router()

    seg = _segment(chunk_id=1, utterance_id=99)
    speech_queue.put(seg)
    router._handle_speech_item(speech_queue.get_nowait())
    _drain(router)

    dispatched = recognizer_queue.get_nowait()
    message_id = dispatched.message_id or -1

    assert router._in_flight_utterance_id == 99
    assert router._in_flight_message_id == message_id

    router._handle_recognizer_output(RecognizerAck(message_id=message_id, ok=True))

    assert router._in_flight_utterance_id is None
    assert router._in_flight_message_id is None



def test_blocking_strategy_when_waiting_for_ack() -> None:
    """When recognizer is busy, ensure router blocks on recognizer queue, 
    and polls speech queue with nowait()."""
    router, speech_queue, _, output_queue, _, _ = _build_router()
    router._in_flight_message_id = 1

    recognizer_timeouts: list[float] = []
    speech_nowait_calls: list[bool] = []

    def _recognizer_get(timeout: float) -> RecognizerAck:
        recognizer_timeouts.append(timeout)
        raise queue.Empty

    def _speech_get_nowait() -> AudioSegment:
        speech_nowait_calls.append(True)
        router.is_running = False
        raise queue.Empty

    def _unexpected_speech_get(timeout: float) -> AudioSegment:
        raise AssertionError(f"speech_queue.get(timeout=...) should not be called, got timeout={timeout}")

    output_queue.get = _recognizer_get  # type: ignore[method-assign]
    speech_queue.get_nowait = _speech_get_nowait  # type: ignore[method-assign]
    speech_queue.get = _unexpected_speech_get  # type: ignore[method-assign]

    router.is_running = True
    router.process()

    assert recognizer_timeouts == [0.05]
    assert speech_nowait_calls == [True]


def test_process_wait_strategy_when_recognizer_is_free() -> None:
    """When recognizer is free, Router blocks on the speech queue, not on Recognizer."""
    router, speech_queue, _, output_queue, _, _ = _build_router()
    router._in_flight_message_id = None

    recognizer_nowait_calls: list[bool] = []
    speech_timeouts: list[float] = []

    def _recognizer_get_nowait() -> RecognizerAck:
        recognizer_nowait_calls.append(True)
        raise queue.Empty

    def _speech_get(timeout: float) -> AudioSegment:
        speech_timeouts.append(timeout)
        router.is_running = False
        raise queue.Empty

    def _unexpected_recognizer_get(timeout: float) -> RecognizerAck:
        raise AssertionError(f"recognizer_output_queue.get(timeout=...) should not be called, got timeout={timeout}")

    def _unexpected_speech_get_nowait() -> AudioSegment:
        raise AssertionError("speech_queue.get_nowait() should not be called when recognizer is free and no work happened")

    output_queue.get_nowait = _recognizer_get_nowait  # type: ignore[method-assign]
    output_queue.get = _unexpected_recognizer_get  # type: ignore[method-assign]
    speech_queue.get = _speech_get  # type: ignore[method-assign]
    speech_queue.get_nowait = _unexpected_speech_get_nowait  # type: ignore[method-assign]

    router.is_running = True
    router.process()

    assert recognizer_nowait_calls == [True]
    assert speech_timeouts == [0.05]
