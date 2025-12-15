"""Confidence score extraction from onnx-asr models.

This module extracts confidence scores from onnx-asr
models by monkey-patching the internal _decode() method.
"""

from typing import Optional, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    from onnx_asr.adapters import TimestampedResultsAsrAdapter


class ConfidenceExtractor:
    """Extracts confidence scores from onnx-asr models via monkey-patching.

    This class intercepts the internal _decode() method of onnx-asr's transducer
    models to capture the probability distributions. Probabilities are converted to confidence scores.

    Usage:
        extractor = ConfidenceExtractor(model)
        result = model.recognize(audio)
        confidences = extractor.get_clear_confidences()
    """

    def __init__(self, model: "TimestampedResultsAsrAdapter"):
        """Initialize the confidence extractor and monkey-patch the model.

        Args:
            model: The onnx-asr model adapter (TimestampedResultsAsrAdapter)
        """
        self.model: "TimestampedResultsAsrAdapter" = model
        self.token_confidences: list[float] = []
        self._is_patched = False
        self._original_decode: Optional[callable] = None
        self.logger = logging.getLogger(__name__)

        self._patch_model()

    def _patch_model(self) -> None:
        """Monkey-patch the model's _decode method to capture probabilities.

        This method replaces the internal _decode() method with our wrapper that
        captures probability distributions before they're consumed by argmax.
        """
        try:
            if not hasattr(self.model, 'asr'):
                self.logger.warning("Model doesn't have 'asr' attribute, confidence extraction disabled")
                return

            # Check if the model has a _decode method (transducer models only)
            if not hasattr(self.model.asr, '_decode'):
                self.logger.warning("Model doesn't have '_decode' method, confidence extraction disabled")
                return

            self._original_decode = self.model.asr._decode
            self.model.asr._decode = self._patched_decode
            self._is_patched = True

            self.logger.debug("Patched model._decode for confidence extraction")

        except Exception as e:
            self.logger.warning(f"Failed to patch model for confidence extraction: {e}")

    def _patched_decode(self, prev_tokens: list[int], prev_state, encoder_out: np.ndarray):
        """Wrapper around the original _decode method that captures probabilities.

        Args:
            prev_tokens: Previously decoded token IDs
            prev_state: Decoder state (model-specific)
            encoder_out: Encoder output for current time step

        Returns:
            tuple: (probs, step, state) - same as original _decode()
        """
        # Call the original method
        probs, step, state = self._original_decode(prev_tokens, prev_state, encoder_out)

        # Extract confidence from probability distribution
        # probs might be logits (raw) or probabilities (after softmax)
        # We apply softmax to be safe, then take the max probability
        confidence = self._compute_confidence(probs)
        self.token_confidences.append(confidence)

        # Return unchanged - onnx-asr continues its normal flow
        return probs, step, state

    def _compute_confidence(self, probs: np.ndarray) -> float:
        """Compute confidence score from probability distribution.

        The input might be:
        - Raw logits (unnormalized)
        - Softmax probabilities (normalized)

        Args:
            probs: Probability distribution over vocabulary (shape: vocab_size)

        Returns:
            float: Confidence score in range [0.0, 1.0]
        """
        try:
            # Apply softmax, subtract max prevents overflow
            exp_probs = np.exp(probs - np.max(probs))
            softmax_probs = exp_probs / np.sum(exp_probs)

            confidence = float(np.max(softmax_probs))

            # Clamp to [0, 1] range (safety check)
            return max(0.0, min(1.0, confidence))

        except Exception as e:
            self.logger.warning(f"Failed to compute confidence: {e}")
            return 0.0

    def get_clear_confidences(self) -> list[float]:
        """Get the captured confidence scores and reset the buffer.

        This method should be called after each recognition to retrieve the
        per-token confidence scores. The internal buffer is cleared after
        retrieval to prepare for the next recognition.

        Returns:
            list[float]: Confidence scores for each decoded token (0.0-1.0 range)
                        Empty list if no tokens were decoded or patching failed
        """
        confidences = self.token_confidences.copy()
        self.token_confidences.clear()
        return confidences

    def is_active(self) -> bool:
        """Check if confidence extraction is active.

        Returns:
            bool: True if the model was successfully patched, False otherwise
        """
        return self._is_patched

    def unpatch(self) -> None:
        """Restore the original _decode method.

        This method should be called when confidence extraction is no longer needed
        or before the model is used elsewhere. It's good practice to unpatch to
        avoid unexpected behavior.
        """
        if self._is_patched and self._original_decode is not None:
            try:
                self.model.asr._decode = self._original_decode
                self._is_patched = False
                self.logger.debug("Successfully unpatched model._decode")
            except Exception as e:
                self.logger.warning(f"Failed to unpatch model: {e}")

    def __del__(self):
        """Cleanup: unpatch on deletion."""
        self.unpatch()
