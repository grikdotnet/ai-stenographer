"""Decodes WebSocket recognition result JSON frames and dispatches to a local subscriber."""

import json
import logging
from typing import TYPE_CHECKING

from src.types import RecognitionResult

if TYPE_CHECKING:
    from src.RecognitionResultPublisher import RecognitionResultPublisher

logger = logging.getLogger(__name__)


class RemoteRecognitionPublisher:
    """Decodes WsRecognitionResult JSON frames and dispatches to a local RecognitionResultPublisher.

    Receives raw JSON text frames from WsClientTransport and translates them
    into local RecognitionResult calls on the publisher.  All decoding errors
    are logged and swallowed — dispatch() never raises.

    Args:
        publisher: Publisher that receives publish_partial_update / publish_finalization calls.
    """

    def __init__(self, publisher: "RecognitionResultPublisher") -> None:
        self._publisher = publisher

    def dispatch(self, json_text: str) -> None:
        """Decode a JSON text frame and call the appropriate subscriber method.

        Algorithm:
            1. Parse JSON; log and return on parse error.
            2. Check type == "recognition_result"; log and return for other types.
            3. Build RecognitionResult from JSON fields; log and return on missing fields.
            4. Route by status: "partial" → publish_partial_update; "final" → publish_finalization.
            5. Unknown status: log and return.

        Args:
            json_text: Raw UTF-8 JSON string from a WebSocket text frame.
        """
        try:
            obj = json.loads(json_text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("RemoteRecognitionPublisher: invalid JSON frame, ignoring")
            return

        msg_type = obj.get("type")
        if msg_type != "recognition_result":
            logger.debug("RemoteRecognitionPublisher: ignoring message type %r", msg_type)
            return

        try:
            result = RecognitionResult(
                text=obj["text"],
                start_time=obj["start_time"],
                end_time=obj["end_time"],
                chunk_ids=obj.get("chunk_ids", []),
                utterance_id=obj.get("utterance_id", 0),
                confidence=obj.get("confidence", 0.0),
                token_confidences=obj.get("token_confidences", []),
                audio_rms=obj.get("audio_rms", 0.0),
                confidence_variance=obj.get("confidence_variance", 0.0),
            )
        except KeyError as exc:
            logger.warning("RemoteRecognitionPublisher: missing field %s in recognition_result", exc)
            return

        status = obj.get("status")
        if status == "partial":
            self._publisher.publish_partial_update(result)
        elif status == "final":
            self._publisher.publish_finalization(result)
        else:
            logger.warning("RemoteRecognitionPublisher: unknown status %r, ignoring", status)
