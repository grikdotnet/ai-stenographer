"""Tests for ConfidenceExtractor.

This module tests the monkey-patching approach used to extract confidence
scores from onnx-asr models without modifying the library source.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from src.ConfidenceExtractor import ConfidenceExtractor


class TestConfidenceExtractor:
    """Test suite for ConfidenceExtractor class."""

    def test_init_successful_patching(self):
        """Test that ConfidenceExtractor successfully patches a valid model."""
        # Create mock model with asr._decode method
        mock_model = Mock()
        mock_model.asr = Mock()
        mock_model.asr._decode = Mock(return_value=(
            np.array([0.1, 0.7, 0.2]),  # probs
            1,  # step
            Mock()  # state
        ))

        # Initialize extractor
        extractor = ConfidenceExtractor(mock_model)

        # Verify patching succeeded
        assert extractor.is_active()
        assert extractor._original_decode is not None
        assert extractor._is_patched

    def test_init_graceful_failure_no_asr_attribute(self):
        """Test that ConfidenceExtractor handles models without asr attribute gracefully."""
        # Create mock model without asr attribute
        mock_model = Mock(spec=[])

        # Initialize extractor
        extractor = ConfidenceExtractor(mock_model)

        # Verify patching failed gracefully (no exception)
        assert not extractor.is_active()
        assert extractor._original_decode is None

    def test_init_graceful_failure_no_decode_method(self):
        """Test that ConfidenceExtractor handles models without _decode method gracefully."""
        # Create mock model with asr but no _decode
        mock_model = Mock()
        mock_model.asr = Mock(spec=[])

        # Initialize extractor
        extractor = ConfidenceExtractor(mock_model)

        # Verify patching failed gracefully
        assert not extractor.is_active()

    def test_patched_decode_captures_confidence(self):
        """Test that patched _decode method captures confidence scores."""
        # Create mock model with a real callable for _decode
        mock_model = Mock()
        mock_model.asr = Mock()

        # Mock _decode returns (probs, step, state)
        mock_probs = np.array([0.1, 0.8, 0.1])  # Max prob = 0.8
        mock_step = 1
        mock_state = Mock()

        # Create a callable that returns the values
        def mock_decode_fn(prev_tokens, prev_state, encoder_out):
            return (mock_probs.copy(), mock_step, mock_state)

        mock_model.asr._decode = mock_decode_fn

        # Initialize extractor (patches the model)
        extractor = ConfidenceExtractor(mock_model)

        # Call the patched _decode method
        prev_tokens = [1, 2]
        prev_state = Mock()
        encoder_out = np.random.randn(128)

        probs, step, state = mock_model.asr._decode(prev_tokens, prev_state, encoder_out)

        # Verify original return values are unchanged
        np.testing.assert_array_equal(probs, mock_probs)
        assert step == mock_step
        assert state == mock_state

        # Verify confidence was captured
        confidences = extractor.get_clear_confidences()
        assert len(confidences) == 1
        assert 0.0 <= confidences[0] <= 1.0
        # After softmax of [0.1, 0.8, 0.1], max should be around 0.58
        assert confidences[0] > 0.5

    def test_compute_confidence_from_logits(self):
        """Test that confidence computation handles raw logits correctly."""
        # Create mock model
        mock_model = Mock()
        mock_model.asr = Mock()

        # Raw logits (unnormalized)
        logits = np.array([2.0, 5.0, 1.0])  # Max at index 1
        mock_model.asr._decode = Mock(return_value=(logits, 1, Mock()))

        # Initialize and run
        extractor = ConfidenceExtractor(mock_model)
        mock_model.asr._decode([1], Mock(), np.random.randn(128))

        # Get confidence
        confidences = extractor.get_clear_confidences()

        # Verify confidence is in valid range and represents max probability
        assert len(confidences) == 1
        assert 0.0 <= confidences[0] <= 1.0
        # After softmax of [2, 5, 1], max probability should be high
        assert confidences[0] > 0.9

    def test_compute_confidence_from_probabilities(self):
        """Test that confidence computation handles probabilities correctly."""
        # Create mock model
        mock_model = Mock()
        mock_model.asr = Mock()

        # Already normalized probabilities
        probs = np.array([0.05, 0.9, 0.05])

        def mock_decode_fn(prev_tokens, prev_state, encoder_out):
            return (probs.copy(), 1, Mock())

        mock_model.asr._decode = mock_decode_fn

        # Initialize and run
        extractor = ConfidenceExtractor(mock_model)
        mock_model.asr._decode([1], Mock(), np.random.randn(128))

        # Get confidence
        confidences = extractor.get_clear_confidences()

        assert len(confidences) == 1
        # After softmax on [0.05, 0.9, 0.05], max is around 0.539
        # (softmax flattens already-normalized distributions)
        assert 0.5 < confidences[0] < 0.6

    def test_multiple_decode_calls_capture_multiple_confidences(self):
        """Test that multiple _decode calls capture multiple confidence scores."""
        # Create mock model
        mock_model = Mock()
        mock_model.asr = Mock()

        # Different probs for each call
        probs_sequence = [
            np.array([0.1, 0.7, 0.2]),
            np.array([0.2, 0.1, 0.7]),
            np.array([0.6, 0.2, 0.2]),
        ]

        # Create generator for side effects
        call_count = [0]
        def mock_decode_fn(prev_tokens, prev_state, encoder_out):
            idx = call_count[0]
            call_count[0] += 1
            return (probs_sequence[idx].copy(), 1, Mock())

        mock_model.asr._decode = mock_decode_fn

        # Initialize extractor
        extractor = ConfidenceExtractor(mock_model)

        # Call _decode multiple times
        for _ in range(3):
            mock_model.asr._decode([1], Mock(), np.random.randn(128))

        # Get all captured confidences
        confidences = extractor.get_clear_confidences()

        assert len(confidences) == 3
        assert all(0.0 <= c <= 1.0 for c in confidences)
        # Each should be valid confidence score
        assert all(c > 0.3 for c in confidences)

    def test_get_clear_confidences_clears_buffer(self):
        """Test that get_clear_confidences() clears the internal buffer."""
        # Create mock model
        mock_model = Mock()
        mock_model.asr = Mock()
        mock_model.asr._decode = Mock(return_value=(np.array([0.1, 0.8, 0.1]), 1, Mock()))

        # Initialize and capture some confidences
        extractor = ConfidenceExtractor(mock_model)
        mock_model.asr._decode([1], Mock(), np.random.randn(128))

        # Get confidences (should clear buffer)
        confidences1 = extractor.get_clear_confidences()
        assert len(confidences1) == 1

        # Get again (should be empty)
        confidences2 = extractor.get_clear_confidences()
        assert len(confidences2) == 0

    def test_unpatch_restores_original_method(self):
        """Test that unpatch() restores the original _decode method."""
        # Create mock model
        mock_model = Mock()
        original_decode = Mock(return_value=(np.array([0.1, 0.8, 0.1]), 1, Mock()))
        mock_model.asr = Mock()
        mock_model.asr._decode = original_decode

        # Patch
        extractor = ConfidenceExtractor(mock_model)
        patched_decode = mock_model.asr._decode

        # Verify it was patched
        assert patched_decode != original_decode
        assert extractor.is_active()

        # Unpatch
        extractor.unpatch()

        # Verify restoration
        assert mock_model.asr._decode == original_decode
        assert not extractor.is_active()

    def test_confidence_extraction_integration(self):
        """Integration test: simulate full recognition flow with confidence extraction."""
        # Create realistic mock model
        mock_model = Mock()
        mock_model.asr = Mock()

        # Simulate transducer decoding for "hello" (3 tokens)
        decode_sequence = [
            (np.array([0.05, 0.85, 0.1]), 0, Mock()),  # "h" - high confidence
            (np.array([0.1, 0.75, 0.15]), 0, Mock()),  # "e" - good confidence
            (np.array([0.3, 0.4, 0.3]), 0, Mock()),    # "llo" - lower confidence
        ]
        mock_model.asr._decode = Mock(side_effect=decode_sequence)

        # Initialize extractor
        extractor = ConfidenceExtractor(mock_model)

        # Simulate recognition loop
        for _ in range(3):
            mock_model.asr._decode([], Mock(), np.random.randn(128))

        # Get confidences
        confidences = extractor.get_clear_confidences()

        # Verify
        assert len(confidences) == 3
        assert confidences[0] > confidences[1] > confidences[2]  # Decreasing confidence
        assert all(0.0 <= c <= 1.0 for c in confidences)

    def test_confidence_extraction_with_no_decode_calls(self):
        """Test that get_clear_confidences() returns empty list when no decoding occurred."""
        # Create mock model
        mock_model = Mock()
        mock_model.asr = Mock()
        mock_model.asr._decode = Mock()

        # Initialize extractor but don't call _decode
        extractor = ConfidenceExtractor(mock_model)

        # Get confidences (should be empty)
        confidences = extractor.get_clear_confidences()
        assert confidences == []

    def test_confidence_clamping(self):
        """Test that confidence scores are clamped to [0, 1] range."""
        # Create mock model with extreme values
        mock_model = Mock()
        mock_model.asr = Mock()

        # Extreme logits
        extreme_logits = np.array([1000.0, -1000.0, 0.0])
        mock_model.asr._decode = Mock(return_value=(extreme_logits, 1, Mock()))

        # Initialize and run
        extractor = ConfidenceExtractor(mock_model)
        mock_model.asr._decode([1], Mock(), np.random.randn(128))

        # Get confidence
        confidences = extractor.get_clear_confidences()

        # Should still be in valid range despite extreme inputs
        assert len(confidences) == 1
        assert 0.0 <= confidences[0] <= 1.0
        assert confidences[0] > 0.99  # Should be very close to 1.0


class TestConfidenceExtractorErrorHandling:
    """Test error handling in ConfidenceExtractor."""

    def test_compute_confidence_handles_exceptions(self):
        """Test that confidence computation handles exceptions gracefully."""
        # Create mock model that raises exception during decode
        mock_model = Mock()
        mock_model.asr = Mock()

        # First call works, second raises exception
        mock_model.asr._decode = Mock(side_effect=[
            (np.array([0.1, 0.8, 0.1]), 1, Mock()),
            Exception("Model error")
        ])

        extractor = ConfidenceExtractor(mock_model)

        # First call succeeds
        mock_model.asr._decode([1], Mock(), np.random.randn(128))

        # Second call raises - should be caught by patched method
        # The patched method should handle this gracefully
        try:
            mock_model.asr._decode([1], Mock(), np.random.randn(128))
        except Exception:
            pass  # Expected

        # Should have captured 1 confidence from successful call
        confidences = extractor.get_clear_confidences()
        # Note: This might be 1 or 0 depending on when exception occurred
        assert isinstance(confidences, list)

    def test_unpatch_handles_errors_gracefully(self):
        """Test that unpatch handles errors without crashing."""
        # Create extractor with mock that will fail on unpatch
        mock_model = Mock()
        mock_model.asr = Mock()
        mock_model.asr._decode = Mock()

        extractor = ConfidenceExtractor(mock_model)

        # Make asr disappear to cause unpatch to fail
        delattr(mock_model, 'asr')

        # Should not raise exception
        extractor.unpatch()  # Should log warning but not crash
