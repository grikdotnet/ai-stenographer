# tests/test_post_recognition_filter.py
import pytest
from src.postprocessing.PostRecognitionFilter import PostRecognitionFilter


class TestPostRecognitionFilter:
    """Tests for PostRecognitionFilter filler-word and confidence-gated filtering."""

    @pytest.fixture
    def filter(self):
        return PostRecognitionFilter()

    def test_always_drop_um(self, filter):
        """'um' is always dropped."""
        text, confidences = filter.filter("um hello", [" um", " hello"], [0.9, 0.8])
        assert "um" not in text
        assert "hello" in text

    def test_always_drop_all_variants(self, filter):
        """All always-drop words produce empty output."""
        tokens = [" um", " uh", " ah", " oh"]
        text, confidences = filter.filter("um uh ah oh", tokens, [0.9, 0.9, 0.9, 0.9])
        assert text == ""
        assert confidences == []

    def test_mm_hmm_multi_token(self, filter):
        """Multi-token ' Mm', '-', 'hmm' normalizes to 'mm-hmm' and is dropped."""
        tokens = [" Mm", "-", "hmm"]
        text, confidences = filter.filter("Mm-hmm", tokens, [0.9, 0.9, 0.9])
        assert text == ""
        assert confidences == []

    def test_confidence_drop_yeah_below_threshold(self, filter):
        """'yeah' with confidence 0.4 (below threshold) is dropped."""
        tokens = [" yeah", " world"]
        text, confidences = filter.filter("yeah world", tokens, [0.4, 0.8])
        assert "yeah" not in text
        assert "world" in text

    def test_confidence_drop_yeah_at_threshold(self, filter):
        """'yeah' with confidence exactly 0.5 is dropped (strict >)."""
        tokens = [" yeah", " world"]
        text, confidences = filter.filter("yeah world", tokens, [0.5, 0.8])
        assert "yeah" not in text
        assert "world" in text

    def test_confidence_drop_yeah_above_threshold(self, filter):
        """'yeah' with confidence 0.6 (above threshold) is kept."""
        tokens = [" yeah", " world"]
        text, confidences = filter.filter("yeah world", tokens, [0.6, 0.8])
        assert "yeah" in text
        assert "world" in text

    def test_confidence_drop_okay_kept(self, filter):
        """'okay' with confidence 0.8 (above threshold) is kept."""
        tokens = [" okay", " world"]
        text, confidences = filter.filter("okay world", tokens, [0.8, 0.9])
        assert "okay" in text
        assert "world" in text

    def test_multi_token_word_uses_min_confidence(self, filter):
        """Multi-token 'yeah' uses min confidence; [0.9, 0.3] → min=0.3 → dropped."""
        tokens = [" ye", "ah", " world"]
        text, confidences = filter.filter("yeah world", tokens, [0.9, 0.3, 0.8])
        assert "yeah" not in text
        assert "ye" not in text
        assert "world" in text

    def test_confidence_missing_conditional_kept(self, filter):
        """When confidences list is empty, conditional word is kept."""
        tokens = [" yeah", " world"]
        text, confidences = filter.filter("yeah world", tokens, [])
        assert "yeah" in text
        assert "world" in text

    def test_confidence_misaligned_conditional_kept(self, filter):
        """When confidences shorter than tokens, conditional word is kept."""
        tokens = [" yeah", " world"]
        text, confidences = filter.filter("yeah world", tokens, [0.3])
        assert "yeah" in text
        assert "world" in text

    def test_non_filler_always_kept(self, filter):
        """Normal words are always kept regardless of confidence."""
        tokens = [" hello", " world"]
        text, confidences = filter.filter("hello world", tokens, [0.1, 0.1])
        assert text == "hello world"

    def test_confidence_alignment_preserved(self, filter):
        """Kept tokens return only their aligned confidences."""
        tokens = [" um", " hello", " world"]
        text, confidences = filter.filter("um hello world", tokens, [0.9, 0.7, 0.8])
        assert "um" not in text
        assert len(confidences) == 2
        assert 0.7 in confidences
        assert 0.8 in confidences

    def test_empty_tokens_fallback(self, filter):
        """When tokens is None, return (text, []) unchanged."""
        text, confidences = filter.filter("hello world", None, [0.9, 0.8])
        assert text == "hello world"
        assert confidences == []

    def test_all_dropped_returns_empty(self, filter):
        """When all words are dropped result is exactly '' not ' '."""
        tokens = [" um", " uh"]
        text, confidences = filter.filter("um uh", tokens, [0.9, 0.9])
        assert text == ""
        assert confidences == []

    def test_unknown_filler_variant_not_dropped(self, filter):
        """'uhh' is not in either drop set and is kept."""
        tokens = [" uhh", " world"]
        text, confidences = filter.filter("uhh world", tokens, [0.9, 0.8])
        assert "uhh" in text
        assert "world" in text

    def test_only_conditional_words_all_dropped(self, filter):
        """'yeah okay' both below confidence threshold → ('', [])."""
        tokens = [" yeah", " okay"]
        text, confidences = filter.filter("yeah okay", tokens, [0.3, 0.4])
        assert text == ""
        assert confidences == []

    def test_always_drop_with_trailing_punctuation(self, filter):
        """'hmm.' (tokens [' H', 'mm', '.']) normalizes to 'hmm' after punctuation strip → dropped."""
        tokens = [" H", "mm", "."]
        text, confidences = filter.filter("Hmm.", tokens, [0.87, 0.80, 0.55])
        assert text == ""
        assert confidences == []
