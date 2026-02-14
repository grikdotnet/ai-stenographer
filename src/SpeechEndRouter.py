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
        verbose: bool = False,
    ) -> None:
        self.speech_queue = speech_queue
        self.recognizer_queue = recognizer_queue
        self.recognizer_output_queue = recognizer_output_queue
        self.matcher_queue = matcher_queue
        self.app_state = app_state
        self.verbose = verbose

        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None

        self._in_flight_message_id: Optional[int] = None
        self._audio_fifo: deque[AudioSegment] = deque()
        self._held_boundary: Optional[SpeechEndSignal] = None
        self._skipped_ack_ids: set[int] = set()
        self._next_message_id: int = 1

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

            try:
                recognizer_output: RecognizerOutputItem = self.recognizer_output_queue.get(timeout=0.05)
                self._handle_recognizer_output(recognizer_output)
                did_work = True
            except queue.Empty:
                pass

            try:
                speech_item: SpeechQueueItem
                if did_work:
                    speech_item = self.speech_queue.get_nowait()
                else:
                    speech_item = self.speech_queue.get(timeout=0.05)
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
            self._put_matcher(item)
            return

        if self._held_boundary is None:
            self._held_boundary = item
        else:
            # Unexpected double-boundary while one is already held.
            self.dropped_protocol_messages += 1
            logging.error(
                "SpeechEndRouter: dropped SpeechEndSignal message_id=%s because boundary already held message_id=%s",
                item.message_id,
                self._held_boundary.message_id,
            )

    def _handle_recognizer_output(self, item: RecognizerOutputItem) -> None:
        """Handle downstream recognizer output and ACK protocol."""
        if self._in_flight_message_id is None:
            if isinstance(item, RecognizerAck) and item.message_id in self._skipped_ack_ids:
                self._skipped_ack_ids.remove(item.message_id)
                return
            self.dropped_protocol_messages += 1
            logging.error(
                "SpeechEndRouter: received recognizer output message_id=%s with no in-flight message",
                getattr(item, "message_id", None),
            )
            return

        if item.message_id != self._in_flight_message_id:
            self.dropped_protocol_messages += 1
            logging.error(
                "SpeechEndRouter: unexpected recognizer message_id=%s expected=%s (dropped)",
                item.message_id,
                self._in_flight_message_id,
            )
            return

        if isinstance(item, RecognitionTextMessage):
            if not self._put_matcher(item.result):
                # Drop both text and terminal ACK handling for this message id.
                # Do not block pipeline progress on matcher backpressure.
                self._skipped_ack_ids.add(item.message_id)
                self._in_flight_message_id = None
            return

        # RecognizerAck is terminal for in-flight item.
        if self.verbose and not item.ok:
            logging.warning(
                "SpeechEndRouter: recognizer ACK failure for message_id=%s error=%s",
                item.message_id,
                item.error,
            )
        self._in_flight_message_id = None

    def _drain_progress(self) -> None:
        """Push next eligible items downstream."""
        if self._in_flight_message_id is None and self._audio_fifo:
            next_audio = self._audio_fifo.popleft()
            if self._put_recognizer(next_audio):
                self._in_flight_message_id = next_audio.message_id
            else:
                # Retry later.
                self._audio_fifo.appendleft(next_audio)

        if self._in_flight_message_id is None and not self._audio_fifo and self._held_boundary is not None:
            held = self._held_boundary
            if self._put_matcher(held):
                self._held_boundary = None

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
