# tests/test_incremental_text_matcher.py
import pytest
import queue
from unittest.mock import Mock
from src.postprocessing.IncrementalTextMatcher import IncrementalTextMatcher
from src.types import RecognitionResult, SpeechEndSignal


class TestIncrementalTextMatcher:
    """Tests for IncrementalTextMatcher with growing window recognition.

    IncrementalTextMatcher processes sequential recognition results from growing windows.
    Uses prefix comparison to detect stable (finalized) vs unstable (preliminary) regions.
    """

    @pytest.fixture
    def text_queue(self):
        return queue.Queue()

    @pytest.fixture
    def mock_publisher(self):
        return Mock(spec=['publish_partial_update', 'publish_finalization'])

    @pytest.fixture
    def mock_app_state(self):
        return Mock()

    def test_first_result_all_preliminary(self, text_queue, mock_publisher, mock_app_state):
        """First incremental result should be published as all preliminary."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result = RecognitionResult(text='hello world', start_time=0.0, end_time=1.0, chunk_ids=[0, 1])

        matcher.process_incremental(result)

        # Should publish as partial_update
        mock_publisher.publish_partial_update.assert_called_once()
        published = mock_publisher.publish_partial_update.call_args[0][0]
        assert published.text == 'hello world'
        mock_publisher.publish_finalization.assert_not_called()

    def test_progressive_finalization(self, text_queue, mock_publisher, mock_app_state):
        """Multiple results should progressively finalize stable prefixes."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        results = [
            RecognitionResult(text='one', start_time=0.0, end_time=0.5, chunk_ids=[0]),
            RecognitionResult(text='one two', start_time=0.0, end_time=1.0, chunk_ids=[0, 1]),
            RecognitionResult(text='one two three', start_time=0.0, end_time=1.5, chunk_ids=[0, 1, 2]),
            RecognitionResult(text='one two three four', start_time=0.0, end_time=2.0, chunk_ids=[0, 1, 2, 3]),
        ]

        # First result: all preliminary
        matcher.process_incremental(results[0])
        assert mock_publisher.publish_partial_update.call_count == 1
        assert mock_publisher.publish_partial_update.call_args[0][0].text == 'one'
        mock_publisher.reset_mock()

        # Second result: finalize "one", preliminary "two"
        matcher.process_incremental(results[1])
        assert mock_publisher.publish_finalization.call_count == 1
        assert mock_publisher.publish_finalization.call_args[0][0].text == 'one'
        assert mock_publisher.publish_partial_update.call_count == 1
        assert mock_publisher.publish_partial_update.call_args[0][0].text == 'two'
        mock_publisher.reset_mock()

        # Third result: finalize "two", preliminary "three"
        matcher.process_incremental(results[2])
        assert mock_publisher.publish_finalization.call_count == 1
        assert mock_publisher.publish_finalization.call_args[0][0].text == 'two'
        assert mock_publisher.publish_partial_update.call_count == 1
        assert mock_publisher.publish_partial_update.call_args[0][0].text == 'three'
        mock_publisher.reset_mock()

        # Fourth result: finalize "three", preliminary "four"
        matcher.process_incremental(results[3])
        assert mock_publisher.publish_finalization.call_count == 1
        assert mock_publisher.publish_finalization.call_args[0][0].text == 'three'
        assert mock_publisher.publish_partial_update.call_count == 1
        assert mock_publisher.publish_partial_update.call_args[0][0].text == 'four'

    def test_no_common_prefix_garbage_transition(self, text_queue, mock_publisher, mock_app_state):
        """When no overlap found, replace preliminary with new garbage."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='garbage text', start_time=0.0, end_time=0.5, chunk_ids=[0])
        result2 = RecognitionResult(text='completely different', start_time=0.0, end_time=1.0, chunk_ids=[0, 1])

        matcher.process_incremental(result1)
        mock_publisher.reset_mock()

        matcher.process_incremental(result2)

        # No finalization, just replace preliminary
        mock_publisher.publish_finalization.assert_not_called()
        assert mock_publisher.publish_partial_update.call_count == 1
        preliminary = mock_publisher.publish_partial_update.call_args[0][0]
        assert preliminary.text == 'completely different'

    def test_window_sliding_with_garbled_start(self, text_queue, mock_publisher, mock_app_state):
        """After window slides, garbled start should not prevent matching using find_word_overlap."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        # Simulate growing window
        result1 = RecognitionResult(text='one two three four', start_time=0.0, end_time=2.0, chunk_ids=[0, 1, 2, 3])
        matcher.process_incremental(result1)
        mock_publisher.reset_mock()

        result2 = RecognitionResult(text='one two three four five', start_time=0.0, end_time=2.5, chunk_ids=[0, 1, 2, 3, 4])
        matcher.process_incremental(result2)
        # After result2: finalized "one two three four", preliminary "five", finalized_word_count=4
        mock_publisher.reset_mock()

        assert matcher.prev_finalized_words == 4

        # Window slides, loses "one two", garbled start
        # find_word_overlap will find "three four five" at positions (2,1,3)
        # Stable region starts at idx_curr=1; words before anchor are unreliable and ignored
        result3 = RecognitionResult(text='florp three four five six', start_time=1.0, end_time=3.0, chunk_ids=[2, 3, 4, 5])
        matcher.process_incremental(result3)

        assert matcher.prev_finalized_words == 4

        # After slide with garbled start, preliminary should only include tail
        assert mock_publisher.publish_partial_update.call_count == 1
        preliminary = mock_publisher.publish_partial_update.call_args[0][0]
        assert preliminary.text == 'six'
        # Finalization should be the stable overlap only
        assert mock_publisher.publish_finalization.call_count == 1
        finalized = mock_publisher.publish_finalization.call_args[0][0]
        assert finalized.text == 'three four five'

    def test_no_overlap_reset_does_not_leak_old_stream_words(self, text_queue, mock_publisher, mock_app_state):
        """After no-overlap reset, finalization should come only from the new stream."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        # Build finalized state in first stream.
        matcher.process_incremental(RecognitionResult(text='alpha beta', start_time=0.0, end_time=1.0, chunk_ids=[0, 1]))
        matcher.process_incremental(RecognitionResult(text='alpha beta gamma', start_time=0.0, end_time=1.5, chunk_ids=[0, 1, 2]))
        assert matcher.prev_finalized_words == 2
        mock_publisher.reset_mock()

        # Jump to unrelated stream (no overlap), then extend it.
        matcher.process_incremental(RecognitionResult(text='trash', start_time=5.0, end_time=5.5, chunk_ids=[5]))
        assert matcher.prev_finalized_words == 0
        mock_publisher.reset_mock()

        matcher.process_incremental(RecognitionResult(text='trash data', start_time=5.0, end_time=6.0, chunk_ids=[5, 6]))
        assert mock_publisher.publish_finalization.call_count == 1
        finalized = mock_publisher.publish_finalization.call_args[0][0]
        assert finalized.text == 'trash'

    def test_finalization_emits_result(self, text_queue, mock_publisher, mock_app_state):
        """Finalized results should be published on stable overlap."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='hello world', start_time=0.0, end_time=1.0, chunk_ids=[0, 1])
        result2 = RecognitionResult(text='hello world today', start_time=0.0, end_time=2.0, chunk_ids=[0, 1, 2])
        matcher.process_incremental(result1)
        mock_publisher.reset_mock()
        matcher.process_incremental(result2)

        assert mock_publisher.publish_finalization.call_count == 1
        finalized = mock_publisher.publish_finalization.call_args[0][0]
        assert finalized.text == 'hello world'

    def test_speech_end_finalizes_pending_text(self, text_queue, mock_publisher, mock_app_state):
        """SpeechEndSignal finalizes pending words."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result = RecognitionResult(text='hello world', start_time=0.0, end_time=1.0, chunk_ids=[0, 1])
        matcher.process_incremental(result)
        mock_publisher.reset_mock()

        # Empty flush text still finalizes pending words from previous_result.
        signal = SpeechEndSignal(utterance_id=1, end_time=1.0)
        matcher.process_speech_end(signal)

        assert mock_publisher.publish_finalization.call_count == 1
        finalized = mock_publisher.publish_finalization.call_args[0][0]
        assert finalized.text == 'hello world'

    def test_flush_finalizes_remaining_text(self, text_queue, mock_publisher, mock_app_state):
        """Flush should finalize any remaining preliminary text."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='hello world', start_time=0.0, end_time=1.0, chunk_ids=[0, 1])
        result2 = RecognitionResult(text='hello world how', start_time=0.0, end_time=2.0, chunk_ids=[0, 1, 2])

        matcher.process_incremental(result1)
        matcher.process_incremental(result2)
        mock_publisher.reset_mock()

        # Flush
        matcher.process_incremental(
            RecognitionResult(text='hello world how are', start_time=0.0, end_time=2.5, chunk_ids=[0, 1, 2, 3])
        )
        matcher.process_speech_end(SpeechEndSignal(utterance_id=1, end_time=2.5))

        # Should finalize remaining "how are"
        # First call processes as incremental (finalizes "how"), second call finalizes remaining ("are")
        assert mock_publisher.publish_finalization.call_count == 2
        assert ' '.join(call[0][0].text for call in mock_publisher.publish_finalization.call_args_list) == 'how are'

    def test_flush_resets_state(self, text_queue, mock_publisher, mock_app_state):
        """After flush, state should be reset for next speech sequence."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='hello world', start_time=0.0, end_time=1.0, chunk_ids=[0, 1])
        matcher.process_incremental(result1)

        matcher.process_incremental(
            RecognitionResult(text='hello world done', start_time=0.0, end_time=2.0, chunk_ids=[0, 1, 2])
        )
        matcher.process_speech_end(SpeechEndSignal(utterance_id=1, end_time=2.0))

        # State should be reset
        assert matcher.previous_result is None
        assert matcher.prev_finalized_words == 0

        # Next sequence should start fresh
        result2 = RecognitionResult(text='new speech', start_time=5.0, end_time=6.0, chunk_ids=[10])
        matcher.process_incremental(result2)

        # Should be treated as first result (all preliminary)
        mock_publisher.publish_partial_update.assert_called()
        preliminary = [call[0][0] for call in mock_publisher.publish_partial_update.call_args_list]
        assert any(p.text == 'new speech' for p in preliminary)

    def test_single_word_flush(self, text_queue, mock_publisher, mock_app_state):
        """Single word followed by flush should finalize correctly."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result = RecognitionResult(text='hello', start_time=0.0, end_time=1.0, chunk_ids=[0])
        matcher.process_incremental(result)
        mock_publisher.reset_mock()

        matcher.process_speech_end(SpeechEndSignal(utterance_id=1, end_time=1.0))

        # Should finalize "hello"
        assert mock_publisher.publish_finalization.call_count >= 1

    def test_find_word_overlap_exact_prefix(self, text_queue, mock_publisher, mock_app_state):
        """Test find_word_overlap with exact prefix match."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)
        matcher.previous_result = RecognitionResult(text='hello world', start_time=0.0, end_time=1.0, chunk_ids=[0, 1])

        result = RecognitionResult(text='hello world how', start_time=0.0, end_time=2.0, chunk_ids=[0, 1, 2])
        curr_normalized = matcher._split_and_normalize(result.text)
        idx1, idx2, length = matcher.find_word_overlap(curr_normalized)

        assert idx1 == 0
        assert idx2 == 0
        assert length == 2

    def test_find_word_overlap_with_garbled_start(self, text_queue, mock_publisher, mock_app_state):
        """Test find_word_overlap when start is garbled (after sliding)."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)
        matcher.previous_result = RecognitionResult(text='one two three four', start_time=0.0, end_time=2.0, chunk_ids=[0, 1, 2, 3])

        result = RecognitionResult(text='florp three four five', start_time=0.5, end_time=2.5, chunk_ids=[1, 2, 3, 4])
        curr_normalized = matcher._split_and_normalize(result.text)
        idx1, idx2, length = matcher.find_word_overlap(curr_normalized)

        assert idx1 == 2
        assert idx2 == 1
        assert length == 2

    def test_find_word_overlap_no_match(self, text_queue, mock_publisher, mock_app_state):
        """Test find_word_overlap with no common words."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)
        matcher.previous_result = RecognitionResult(text='hello world', start_time=0.0, end_time=1.0, chunk_ids=[0, 1])

        result = RecognitionResult(text='completely different', start_time=2.0, end_time=3.0, chunk_ids=[3, 4])
        curr_normalized = matcher._split_and_normalize(result.text)
        idx1, idx2, length = matcher.find_word_overlap(curr_normalized)

        assert length == 0

    def test_punctuation_normalization(self, text_queue, mock_publisher, mock_app_state):
        """Test that punctuation differences don't prevent overlap detection."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='Hello?', start_time=0.0, end_time=1.0, chunk_ids=[0])
        result2 = RecognitionResult(text='Hello, how', start_time=0.0, end_time=2.0, chunk_ids=[0, 1])

        matcher.process_incremental(result1)
        mock_publisher.reset_mock()

        matcher.process_incremental(result2)

        # Should finalize "Hello?" (with original punctuation)
        assert mock_publisher.publish_finalization.call_count == 1
        finalized = mock_publisher.publish_finalization.call_args[0][0]
        assert 'Hello' in finalized.text

    def test_case_normalization(self, text_queue, mock_publisher, mock_app_state):
        """Test that case differences don't prevent overlap detection."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='HELLO WORLD', start_time=0.0, end_time=1.0, chunk_ids=[0])
        result2 = RecognitionResult(text='hello world today', start_time=0.0, end_time=2.0, chunk_ids=[0, 1])

        matcher.process_incremental(result1)
        mock_publisher.reset_mock()

        matcher.process_incremental(result2)

        # Should finalize "hello world" (from result2's text, which replaces result1)
        # In growing window model, result2 is the canonical text
        assert mock_publisher.publish_finalization.call_count == 1
        finalized = mock_publisher.publish_finalization.call_args[0][0]
        assert 'hello world' == finalized.text

    def test_route_incremental_item(self, text_queue, mock_publisher, mock_app_state):
        """process_item should route RecognitionResult to incremental processing."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result = RecognitionResult(text='hello', start_time=0.0, end_time=1.0, chunk_ids=[0])
        matcher.process_item(result)

        # Should call publish_partial_update
        mock_publisher.publish_partial_update.assert_called_once()

    def test_route_speech_end_signal(self, text_queue, mock_publisher, mock_app_state):
        """process_item should route SpeechEndSignal to finalization path."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='hello', start_time=0.0, end_time=1.0, chunk_ids=[0])
        matcher.process_item(result1)
        mock_publisher.reset_mock()

        matcher.process_item(SpeechEndSignal(utterance_id=1, end_time=2.0))

        # Should finalize remaining text
        assert mock_publisher.publish_finalization.call_count >= 1

    def test_observer_pattern_shutdown(self, text_queue, mock_publisher, mock_app_state):
        """Test that shutdown via observer finalizes pending and stops."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result = RecognitionResult(text='pending text', start_time=0.0, end_time=1.0, chunk_ids=[0])
        matcher.process_item(result)
        mock_publisher.reset_mock()

        # Trigger shutdown
        matcher.on_state_change('running', 'shutdown')

        # Should finalize pending
        assert mock_publisher.publish_finalization.call_count >= 1

        # Should stop
        assert matcher.is_running is False


class TestIncrementalTextMatcherEdgeCases:
    """Tests for edge cases in IncrementalTextMatcher."""

    @pytest.fixture
    def text_queue(self):
        return queue.Queue()

    @pytest.fixture
    def mock_publisher(self):
        return Mock(spec=['publish_partial_update', 'publish_finalization'])

    @pytest.fixture
    def mock_app_state(self):
        return Mock()

    def test_empty_result_text(self, text_queue, mock_publisher, mock_app_state):
        """Test handling of empty result text."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result = RecognitionResult(text='', start_time=0.0, end_time=1.0, chunk_ids=[0])
        matcher.process_incremental(result)

        # Should handle gracefully (publish empty preliminary or skip)
        # Implementation should not crash

    def test_result_with_extra_whitespace(self, text_queue, mock_publisher, mock_app_state):
        """Test handling of results with extra whitespace."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='hello  world', start_time=0.0, end_time=1.0, chunk_ids=[0])
        result2 = RecognitionResult(text='hello world  today', start_time=0.0, end_time=2.0, chunk_ids=[0, 1])

        matcher.process_incremental(result1)
        mock_publisher.reset_mock()

        matcher.process_incremental(result2)

        # Should normalize whitespace and find overlap
        assert mock_publisher.publish_finalization.call_count == 1

    def test_very_long_result(self, text_queue, mock_publisher, mock_app_state):
        """Test handling of very long recognition results."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        long_text = ' '.join(['word'] * 100)
        result = RecognitionResult(text=long_text, start_time=0.0, end_time=10.0, chunk_ids=list(range(100)))
        matcher.process_incremental(result)

        # Should handle without performance issues
        mock_publisher.publish_partial_update.assert_called_once()

    def test_unicode_characters(self, text_queue, mock_publisher, mock_app_state):
        """Test handling of Unicode characters."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='café here', start_time=0.0, end_time=1.0, chunk_ids=[0])
        result2 = RecognitionResult(text='cafe here today', start_time=0.0, end_time=2.0, chunk_ids=[0, 1])

        matcher.process_incremental(result1)
        mock_publisher.reset_mock()

        matcher.process_incremental(result2)

        # Should normalize and find overlap
        assert mock_publisher.publish_finalization.call_count == 1

    def test_repeated_same_words(self, text_queue, mock_publisher, mock_app_state):
        """Test handling when user says the same words multiple times."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='hello hello', start_time=0.0, end_time=1.0, chunk_ids=[0])
        result2 = RecognitionResult(text='hello hello hello', start_time=0.0, end_time=2.0, chunk_ids=[0, 1])

        matcher.process_incremental(result1)
        mock_publisher.reset_mock()

        matcher.process_incremental(result2)

        # Should find common prefix and finalize correctly
        assert mock_publisher.publish_finalization.call_count == 1

    def test_finalized_word_count_tracking(self, text_queue, mock_publisher, mock_app_state):
        """Test that finalized_word_count is tracked correctly."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        results = [
            RecognitionResult(text='one', start_time=0.0, end_time=0.5, chunk_ids=[0]),
            RecognitionResult(text='one two', start_time=0.0, end_time=1.0, chunk_ids=[0, 1]),
            RecognitionResult(text='one two three', start_time=0.0, end_time=1.5, chunk_ids=[0, 1, 2]),
        ]

        matcher.process_incremental(results[0])
        assert matcher.prev_finalized_words == 0

        matcher.process_incremental(results[1])
        assert matcher.prev_finalized_words == 1  # "one" finalized

        matcher.process_incremental(results[2])
        assert matcher.prev_finalized_words == 2  # "one two" finalized

    def test_flush_with_no_previous_result(self, text_queue, mock_publisher, mock_app_state):
        """Test flush when there's no previous result."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        matcher.process_speech_end(SpeechEndSignal(utterance_id=1, end_time=1.0))

        # No pending state means no finalization should be emitted.
        mock_publisher.publish_finalization.assert_not_called()

class TestPrevNormalizedCaching:
    """Tests for caching of previous normalized words."""

    @pytest.fixture
    def text_queue(self):
        return queue.Queue()

    @pytest.fixture
    def mock_publisher(self):
        return Mock(spec=['publish_partial_update', 'publish_finalization'])

    @pytest.fixture
    def mock_app_state(self):
        return Mock()

    def test_cache_starts_none(self, text_queue, mock_publisher, mock_app_state):
        """Cache should be None before any results are processed."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)
        assert matcher._prev_normalized_words is None

    def test_cache_invalidated_on_no_overlap(self, text_queue, mock_publisher, mock_app_state):
        """Cache should be invalidated when no overlap is found."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='hello world', start_time=0.0, end_time=1.0, chunk_ids=[0, 1])
        result2 = RecognitionResult(text='completely different', start_time=2.0, end_time=3.0, chunk_ids=[3, 4])

        matcher.process_incremental(result1)
        matcher.process_incremental(result2)

        assert matcher._prev_normalized_words == matcher._split_and_normalize(result2.text)

    def test_cache_invalidated_after_flush(self, text_queue, mock_publisher, mock_app_state):
        """Cache should be None after flush resets state."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result = RecognitionResult(text='hello world', start_time=0.0, end_time=1.0, chunk_ids=[0, 1])
        matcher.process_incremental(result)

        flush = SpeechEndSignal(utterance_id=1, end_time=1.0)
        matcher.process_speech_end(flush)

        assert matcher._prev_normalized_words is None
        assert matcher.previous_result is None


class TestSameStartChunkIdPrefixMatching:
    """Tests for same-start chunk_id prefix-only matching policy.

    When chunk_ids[0] is the same between consecutive results (growing window),
    the matcher must use strict prefix matching instead of arbitrary-position
    find_word_overlap().
    """

    @pytest.fixture
    def text_queue(self):
        return queue.Queue()

    @pytest.fixture
    def mock_publisher(self):
        return Mock(spec=['publish_partial_update', 'publish_finalization'])

    @pytest.fixture
    def mock_app_state(self):
        return Mock()

    def test_same_start_no_prefix_match_becomes_preliminary(self, text_queue, mock_publisher, mock_app_state):
        """Same-start result with no prefix match must not finalize; becomes new preliminary."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='How do I', start_time=0.0, end_time=1.0, chunk_ids=list(range(32)))
        result2 = RecognitionResult(text='And says how do I get to W', start_time=0.0, end_time=2.0, chunk_ids=list(range(52)))

        matcher.process_incremental(result1)
        mock_publisher.reset_mock()

        matcher.process_incremental(result2)

        mock_publisher.publish_finalization.assert_not_called()
        mock_publisher.publish_partial_update.assert_called_once()
        assert mock_publisher.publish_partial_update.call_args[0][0].text == result2.text

    def test_same_start_no_prefix_match_state_reset(self, text_queue, mock_publisher, mock_app_state):
        """After same-start no-prefix-match, state must be fully reset."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='How do I', start_time=0.0, end_time=1.0, chunk_ids=list(range(32)))
        result2 = RecognitionResult(text='And says how do I get to W', start_time=0.0, end_time=2.0, chunk_ids=list(range(52)))

        matcher.process_incremental(result1)
        matcher.process_incremental(result2)

        assert matcher.prev_finalized_words == 0
        assert matcher.previous_result == result2
        assert matcher._prev_normalized_words == matcher._split_and_normalize(result2.text)

    def test_same_start_prefix_match_finalizes_prefix(self, text_queue, mock_publisher, mock_app_state):
        """Same-start result with prefix match finalizes the stable prefix, tail is preliminary."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='How do I', start_time=0.0, end_time=1.0, chunk_ids=list(range(32)))
        result2 = RecognitionResult(text='And says how do I get to W', start_time=0.0, end_time=2.0, chunk_ids=list(range(52)))
        result3 = RecognitionResult(text='and says, How do I get to Dublin?', start_time=0.0, end_time=3.0, chunk_ids=list(range(62)))

        matcher.process_incremental(result1)
        matcher.process_incremental(result2)
        mock_publisher.reset_mock()

        matcher.process_incremental(result3)

        assert mock_publisher.publish_finalization.call_count == 1
        finalized = mock_publisher.publish_finalization.call_args[0][0]
        assert finalized.text == 'and says, How do I get to'

        assert mock_publisher.publish_partial_update.call_count == 1
        preliminary = mock_publisher.publish_partial_update.call_args[0][0]
        assert preliminary.text == 'Dublin?'

    def test_same_start_then_continuation_can_finalize(self, text_queue, mock_publisher, mock_app_state):
        """Full three-step: result1 preliminary, result2 no-finalize, result3 finalizes prefix."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='How do I', start_time=0.0, end_time=1.0, chunk_ids=list(range(32)))
        result2 = RecognitionResult(text='And says how do I get to W', start_time=0.0, end_time=2.0, chunk_ids=list(range(52)))
        result3 = RecognitionResult(text='and says, How do I get to Dublin?', start_time=0.0, end_time=3.0, chunk_ids=list(range(62)))

        matcher.process_incremental(result1)
        assert mock_publisher.publish_finalization.call_count == 0

        matcher.process_incremental(result2)
        assert mock_publisher.publish_finalization.call_count == 0

        matcher.process_incremental(result3)
        assert mock_publisher.publish_finalization.call_count == 1
        finalized = mock_publisher.publish_finalization.call_args[0][0]
        assert finalized.text == 'and says, How do I get to'

    def test_same_start_prefix_monotonic_finalization(self, text_queue, mock_publisher, mock_app_state):
        """prev_finalized_words only increases; no re-finalization of already-emitted words."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='How do I', start_time=0.0, end_time=1.0, chunk_ids=list(range(32)))
        result2 = RecognitionResult(text='And says how do I get to W', start_time=0.0, end_time=2.0, chunk_ids=list(range(52)))
        result3 = RecognitionResult(text='and says, How do I get to Dublin?', start_time=0.0, end_time=3.0, chunk_ids=list(range(62)))
        result4 = RecognitionResult(text='and says, How do I get to Dublin? Right?', start_time=0.0, end_time=4.0, chunk_ids=list(range(72)))

        matcher.process_incremental(result1)
        matcher.process_incremental(result2)
        assert matcher.prev_finalized_words == 0

        matcher.process_incremental(result3)
        assert matcher.prev_finalized_words == 7

        matcher.process_incremental(result4)
        assert matcher.prev_finalized_words == 8

    def test_start_change_no_finalization_on_zero_overlap(self, text_queue, mock_publisher, mock_app_state):
        """Different chunk_ids[0], no word overlap → no finalization, state reset."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='hello world', start_time=0.0, end_time=1.0, chunk_ids=[0, 1])
        result2 = RecognitionResult(text='completely different', start_time=2.0, end_time=3.0, chunk_ids=[5, 6])

        matcher.process_incremental(result1)
        mock_publisher.reset_mock()

        matcher.process_incremental(result2)

        mock_publisher.publish_finalization.assert_not_called()
        assert matcher.prev_finalized_words == 0
        assert matcher.previous_result == result2

    def test_start_change_with_overlap_still_finalizes(self, text_queue, mock_publisher, mock_app_state):
        """Different chunk_ids[0] with lexical overlap → find_word_overlap path still finalizes."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='one two three four', start_time=0.0, end_time=2.0, chunk_ids=[0, 1, 2, 3])
        result2 = RecognitionResult(text='one two three four five', start_time=0.0, end_time=2.5, chunk_ids=[0, 1, 2, 3, 4])
        matcher.process_incremental(result1)
        matcher.process_incremental(result2)
        mock_publisher.reset_mock()

        result3 = RecognitionResult(text='florp three four five six', start_time=1.0, end_time=3.0, chunk_ids=[2, 3, 4, 5])
        matcher.process_incremental(result3)

        assert mock_publisher.publish_finalization.call_count == 1
        finalized = mock_publisher.publish_finalization.call_args[0][0]
        assert finalized.text == 'three four five'
        assert mock_publisher.publish_partial_update.call_args[0][0].text == 'six'

    def test_empty_chunk_ids_fallback(self, text_queue, mock_publisher, mock_app_state):
        """Empty chunk_ids on either side must not crash and must not finalize."""
        matcher = IncrementalTextMatcher(text_queue, mock_publisher, app_state=mock_app_state)

        result1 = RecognitionResult(text='hello world', start_time=0.0, end_time=1.0, chunk_ids=[])
        result2 = RecognitionResult(text='hello world today', start_time=0.0, end_time=2.0, chunk_ids=[0, 1, 2])

        matcher.process_incremental(result1)
        mock_publisher.reset_mock()

        matcher.process_incremental(result2)

        assert matcher.prev_finalized_words == 0
        assert matcher.previous_result == result2

        result3 = RecognitionResult(text='something else', start_time=3.0, end_time=4.0, chunk_ids=[])
        matcher.process_incremental(result3)
        assert matcher.prev_finalized_words == 0
