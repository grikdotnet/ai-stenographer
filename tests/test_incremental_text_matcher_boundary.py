import queue
from unittest.mock import Mock

from src.postprocessing.IncrementalTextMatcher import IncrementalTextMatcher
from src.types import RecognitionResult, SpeechEndSignal


def test_speech_end_signal_finalizes_pending_and_resets_state() -> None:
    publisher = Mock(spec=["publish_partial_update", "publish_finalization"])
    matcher = IncrementalTextMatcher(
        text_queue=queue.Queue(),
        publisher=publisher,
        app_state=Mock(),
    )

    matcher.process_incremental(
        RecognitionResult(
            text="hello world",
            start_time=0.0,
            end_time=1.0,
            chunk_ids=[1],
        )
    )
    publisher.reset_mock()

    matcher.process_item(SpeechEndSignal(utterance_id=1, end_time=1.0))

    publisher.publish_finalization.assert_called_once()
    finalized = publisher.publish_finalization.call_args[0][0]
    assert finalized.text == "hello world"
    assert matcher.previous_result is None
    assert matcher.prev_finalized_words == 0
