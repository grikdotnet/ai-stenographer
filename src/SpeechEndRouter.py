import logging
import queue
import threading
from collections import deque
from typing import Optional, TYPE_CHECKING

from .types import (
    AudioSegment,
    MatcherQueueItem,
    RecognitionTextMessage,
    RecognizerAck,
    RecognizerFreeSignal,
    RecognizerOutputItem,
    SpeechEndSignal,
    SpeechQueueItem,
)

if TYPE_CHECKING:
    from src.ApplicationState import ApplicationState


class SpeechEndRouter:
    """Routes speech and recognizer protocol messages with strict ordering.

    Responsibilities:
    - Enforce exactly one in-flight AudioSegment to Recognizer.
    - Hold end-of-speech boundaries until in-flight audio completes.
    - Forward recognition text and final boundaries to matcher queue in order.

    Observer Pattern:
    - Subscribes to ApplicationState and stops on shutdown.
    """

    def __init__(
        self,
        speech_queue: queue.Queue,
        recognizer_queue: queue.Queue,
        recognizer_output_queue: queue.Queue,
        matcher_queue: queue.Queue,
        app_state: "ApplicationState",
        control_queue: queue.Queue,
        verbose: bool = False,
    ) -> None:
        self.speech_queue: queue.Queue[SpeechQueueItem] = speech_queue
        self.recognizer_queue: queue.Queue[AudioSegment] = recognizer_queue
        self.recognizer_output_queue: queue.Queue[RecognizerOutputItem] = recognizer_output_queue
        self.matcher_queue: queue.Queue[MatcherQueueItem] = matcher_queue
        self.app_state: "ApplicationState" = app_state
        self.verbose: bool = verbose

        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None

        self._in_flight_message_id: Optional[int] = None
        self._audio_fifo: deque[AudioSegment] = deque()
        self._held_boundary: Optional[SpeechEndSignal] = None
        self._skipped_ack_ids: set[int] = set()
        self._next_message_id: int = 1

        # An optimization channel to signal SoundPreProcessor that Recognizer is free.
        self.control_queue: queue.Queue[RecognizerFreeSignal] = control_queue
        self._control_seq: int = 0

        self._in_flight_utterance_id: Optional[int] = None
        self.dropped_control_signals: int = 0

        self.dropped_protocol_messages: int = 0
        self.dropped_matcher_messages: int = 0
        self.dropped_recognizer_messages: int = 0

        self.app_state.register_component_observer(self.on_state_change)

    def start(self) -> None:
        """Start router worker thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop router worker thread."""
        if not self.is_running:
            return
        self.is_running = False

    def on_state_change(self, old_state: str, new_state: str) -> None:
        """Stop when application transitions to shutdown."""
        if new_state == "shutdown":
            self.stop()

    def process(self) -> None:
        """Route messages from both inputs and maintain protocol invariants."""
        while self.is_running:
            did_work = False
            waiting_for_ack = self._in_flight_message_id is not None

            try:
                recognizer_output: RecognizerOutputItem = (
                    self.recognizer_output_queue.get(timeout=0.05) if waiting_for_ack
                    else self.recognizer_output_queue.get_nowait()
                )
                self._handle_recognizer_output(recognizer_output)
                did_work = True
            except queue.Empty:
                pass

            try:
                speech_item: SpeechQueueItem = (
                    self.speech_queue.get_nowait() if waiting_for_ack or did_work
                    else self.speech_queue.get(timeout=0.05)
                )
                self._handle_speech_item(speech_item)
                did_work = True
            except queue.Empty:
                pass

            if did_work:
                self._drain_progress()

    def _handle_speech_item(self, item: SpeechQueueItem) -> None:
        """Handle upstream item from SoundPreProcessor."""
        message_id = self._next_message_id
        self._next_message_id += 1

        if isinstance(item, AudioSegment):
            item.message_id = message_id
            self._audio_fifo.append(item)
            if self.verbose:
                logging.debug(f"SpeechEndRouter: queued AudioSegment message_id={message_id}")
            return

        item.message_id = message_id
        if self._in_flight_message_id is None and not self._audio_fifo:
            # Fast-path optimization for SpeechEndSignal when Recognizer is idle and no backlog,
            # happends at end of utterance. Immediate route to IncrementalTextMatcher.
            self._put_matcher(item)
            return

        if self._held_boundary is None:
            self._held_boundary = item
        else:
            # Unexpected double-boundary while one is already held.
            self.dropped_protocol_messages += 1
            logging.error(
                "SpeechEndRouter: dropped SpeechEndSignal message_id=%s, boundary already held message_id=%s",
                item.message_id,
                self._held_boundary.message_id,
            )

    def _handle_recognizer_output(self, item: RecognizerOutputItem) -> None:
        """Handle downstream recognizer output and ACK protocol."""
        if self._in_flight_message_id is None:
            if isinstance(item, RecognizerAck) and item.message_id in self._skipped_ack_ids:
                # This ACK corresponds to an audio segment dropped when matcher queue was full.
                self._skipped_ack_ids.remove(item.message_id)
                return
            self.dropped_protocol_messages += 1
            return

        if item.message_id != self._in_flight_message_id:
            self.dropped_protocol_messages += 1
            logging.error(
                "SpeechEndRouter: unexpected recognizer message_id=%s expected=%s",
                item.message_id,
                self._in_flight_message_id,
            )
            return

        if isinstance(item, RecognitionTextMessage):
            if not self._put_matcher(item.result):
                # matcher queue is full, drop text and ACK for this message id, avoid blocking pipeline
                self._skipped_ack_ids.add(item.message_id)
                self._in_flight_message_id = None
                self._in_flight_utterance_id = None
            return

        # RecognizerAck is terminal for in-flight item.
        if self.verbose and not item.ok:
            logging.warning(
                "SpeechEndRouter: recognizer ACK failure for message_id=%s error=%s",
                item.message_id,
                item.error,
            )

        # Send RecognizerFreeSignal (ACK-driven) to SoundPreProcessor.
        signal = RecognizerFreeSignal(
            seq=self._control_seq,
            message_id=item.message_id,
            utterance_id=self._in_flight_utterance_id,
        )
        self._control_seq += 1

        if self.verbose:
            logging.debug(f"SpeechEndRouter: sent RecognizerFreeSignal seq={signal.seq} message_id={signal.message_id} utterance_id={signal.utterance_id}")
        try:
            self.control_queue.put_nowait(signal)
        except queue.Full:
            self.dropped_control_signals += 1
            if self.verbose:
                logging.warning(
                    "SpeechEndRouter: control_queue full, RecognizerFreeSignal dropped (seq=%s, drops=%s)",
                    signal.seq,
                    self.dropped_control_signals
                )

        self._in_flight_utterance_id = None
        self._in_flight_message_id = None

    def _drain_progress(self) -> None:
        """Push next eligible items downstream."""
        if self._held_boundary is not None:
            self._drop_invalid_audio_before_held_boundary()

            if self._should_release_held_boundary():
                held = self._held_boundary
                if self._put_matcher(held):
                    self._held_boundary = None

        if self._in_flight_message_id is None and self._audio_fifo:
            next_audio = self._audio_fifo[0]
            if self._held_boundary is not None and next_audio.utterance_id > self._held_boundary.utterance_id:
                return

            next_audio = self._audio_fifo.popleft()
            if self._put_recognizer(next_audio):
                self._in_flight_message_id = next_audio.message_id
                self._in_flight_utterance_id = next_audio.utterance_id
            else:
                # Retry later.
                self._audio_fifo.appendleft(next_audio)

    def _drop_invalid_audio_before_held_boundary(self) -> None:
        """Drop stale AudioSegment from FIFO that violate ordering.

        Algorithm: If the left-most AudioSegment entry is from an older utterance 
        than current utterance boundary, drop it.
        This should never happen.
        """

        boundary_utterance_id = self._held_boundary.utterance_id
        while self._audio_fifo and self._audio_fifo[0].utterance_id < boundary_utterance_id:
            stale_segment = self._audio_fifo.popleft()
            self.dropped_protocol_messages += 1
            self.dropped_recognizer_messages += 1
            logging.error(
                "SpeechEndRouter: dropped stale audio segment; message_id=%s utterance_id=%s, current utterance_id=%s",
                stale_segment.message_id,
                stale_segment.utterance_id,
                boundary_utterance_id,
            )

    def _should_release_held_boundary(self) -> bool:
        """Return True when boundary can be forwarded before FIFO dispatch.

        Algorithm:
        1. Hold boundary while an audio segment ACK is in-flight.
        2. Release immediately when FIFO is empty.
        3. Release when FIFO head belongs to a newer utterance.
        4. Hold boundary when FIFO head belongs to the same utterance.
        """
        if self._held_boundary is None or self._in_flight_message_id is not None:
            return False

        if not self._audio_fifo:
            return True

        boundary_utterance_id = self._held_boundary.utterance_id
        return self._audio_fifo[0].utterance_id > boundary_utterance_id

    def _put_recognizer(self, segment: AudioSegment) -> bool:
        """Enqueue a segment to recognizer queue with drop accounting."""
        try:
            self.recognizer_queue.put_nowait(segment)
            return True
        except queue.Full:
            self.dropped_recognizer_messages += 1
            if self.verbose:
                logging.warning(
                    "SpeechEndRouter: recognizer_queue full, delayed message_id=%s (drops=%s)",
                    segment.message_id,
                    self.dropped_recognizer_messages,
                )
            return False

    def _put_matcher(self, item: MatcherQueueItem) -> bool:
        """Enqueue a matcher item with drop accounting."""
        try:
            self.matcher_queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped_matcher_messages += 1
            if self.verbose:
                logging.warning(
                    "SpeechEndRouter: matcher_queue full, delayed message_id=%s (drops=%s)",
                    getattr(item, "message_id", None),
                    self.dropped_matcher_messages,
                )
            return False
