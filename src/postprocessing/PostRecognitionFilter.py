import string
from typing import Optional


class PostRecognitionFilter:
    """Filters recognized words using filler-word rules and per-word confidence.

    Responsibilities:
    - Group subword tokens into reconstructed words.
    - Drop words in the always-drop set unconditionally.
    - Drop confidence-gated words when min token confidence <= 0.5.
    - Preserve token-confidence alignment for kept tokens.
    """

    _ALWAYS_DROP: frozenset[str] = frozenset({"um", "uh", "ah", "oh", "hmm", "mm", "mm-hmm"})
    _CONFIDENCE_DROP: frozenset[str] = frozenset({"yeah", "okay"})

    def filter(
        self,
        text: str,
        tokens: Optional[list[str]],
        confidences: list[float],
    ) -> tuple[str, list[float]]:
        """Filter words from recognized text using filler and confidence rules.

        Algorithm:
        1. If tokens missing/empty: return (text, []) unchanged.
        2. Group token indices into words (new word when i==0 or token starts with space).
        3. For each word:
           a. Normalize: ''.join(tokens[i] for i in word_indices).strip().lower().strip(punctuation)
           b. If in _ALWAYS_DROP: drop.
           c. If in _CONFIDENCE_DROP:
              - If len(confidences) == len(tokens): keep only if min(word_confidences) > 0.5
              - Else (missing/misaligned): keep.
           d. Otherwise: keep.
        4. If no words kept: return ('', []).
        5. Reconstruct: ''.join(kept_tokens).strip(); return with aligned confidences.

        Args:
            text: Full recognized text
            tokens: Token list from ASR (subword units with space prefix)
            confidences: Per-token confidence scores

        Returns:
            tuple of (filtered_text, filtered_confidences)
        """
        if not tokens:
            return text, []

        words: list[list[int]] = []
        for i, token in enumerate(tokens):
            if i == 0 or token.startswith(' '):
                words.append([i])
            else:
                words[-1].append(i)

        has_confidences = len(confidences) == len(tokens)
        kept_indices: list[int] = []

        for word_indices in words:
            normalized = ''.join(tokens[i] for i in word_indices).strip().lower().strip(string.punctuation)

            if normalized in self._ALWAYS_DROP:
                continue

            if normalized in self._CONFIDENCE_DROP:
                if has_confidences:
                    word_confidences = [confidences[i] for i in word_indices]
                    if min(word_confidences) > 0.5:
                        kept_indices.extend(word_indices)
                else:
                    kept_indices.extend(word_indices)
                continue

            kept_indices.extend(word_indices)

        if not kept_indices:
            return '', []

        kept_tokens = [tokens[i] for i in kept_indices]
        kept_confidences = [confidences[i] for i in kept_indices] if has_confidences else []
        return ''.join(kept_tokens).strip(), kept_confidences
