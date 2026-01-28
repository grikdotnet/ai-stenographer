# src/postprocessing/TextNormalizer.py
import string
import unicodedata
from typing import Dict


class TextNormalizer:
    """Normalizes text for duplicate detection by removing punctuation,
    normalizing case, handling Unicode variants, and cleaning whitespace.

    This class provides text normalization functionality to enable detection
    of duplicate speech recognition results that may differ only in formatting
    (punctuation, case, Unicode representation) but represent the same spoken content.
    """

    def __init__(self):
        """Initialize the text normalizer with language-specific character mappings."""
        # Language-specific character replacements
        self.character_replacements: Dict[str, str] = {
            'ß': 'ss',
            'ё': 'е',
            'Ё': 'Е',
        }

        # Characters to preserve during accent removal (distinct letters, not accents)
        # These will be temporarily replaced before NFKD decomposition
        self.preserve_chars = {
            'й': '\x00PRESERVE_SHORT_I\x00',
            'Й': '\x00PRESERVE_SHORT_I_UPPER\x00',
        }
        self.restore_chars = {v: k for k, v in self.preserve_chars.items()}

    def normalize_text(self, text: str) -> str:
        """Normalize text for duplicate detection.

        Performs the following normalization steps:
        1. Language-specific character replacements
        2. Remove accents using NFKD decomposition
        3. Replace hyphens with spaces
        4. Remove punctuation
        5. Case normalization to lowercase
        6. Whitespace cleanup

        Args:
            text: Input text to normalize

        Returns:
            Normalized text string suitable for comparison
        """
        if not text:
            return ""

        # Step 1: Language-specific character replacements (before normalization)
        normalized = text
        for original, replacement in self.character_replacements.items():
            normalized = normalized.replace(original, replacement)

        # Step 1.5: Preserve distinct letters that would be corrupted by NFKD
        for original, placeholder in self.preserve_chars.items():
            normalized = normalized.replace(original, placeholder)

        # Step 2: Remove accents using NFKD and filtering
        # NFKD compatibility decomposition separates accents from base characters
        decomposed = unicodedata.normalize('NFKD', normalized)
        # Remove only combining diacritical marks (category Mn)
        without_accents = ''.join(
            char for char in decomposed
            if unicodedata.category(char) != 'Mn'
        )

        # Step 2.5: Restore preserved characters
        for placeholder, original in self.restore_chars.items():
            without_accents = without_accents.replace(placeholder, original)

        # Step 3: Replace hyphens with spaces to preserve word boundaries
        # Hyphens in speech often represent word boundaries (e.g., "one-two-three")
        without_accents = without_accents.replace('-', ' ')

        # Step 4: Remove punctuation
        without_punctuation = ''.join(
            char for char in without_accents
            if char not in string.punctuation
        )

        # Step 5: Case normalization to lowercase
        lowercased = without_punctuation.lower()

        # Step 6: Whitespace cleanup
        # Split and rejoin to normalize all whitespace to single spaces
        # and remove leading/trailing whitespace
        cleaned = ' '.join(lowercased.split())

        return cleaned
