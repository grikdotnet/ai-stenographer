# src/TextNormalizer.py
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
            # German ß handling
            'ß': 'ss',
            # Russian ё handling (with NFD this becomes normalized)
            'ё': 'е',
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text for duplicate detection.

        Performs the following normalization steps:
        1. Unicode normalization (NFD decomposition)
        2. Language-specific character replacements
        3. Punctuation removal
        4. Case normalization to lowercase
        5. Whitespace cleanup

        Args:
            text: Input text to normalize

        Returns:
            Normalized text string suitable for comparison
        """
        if not text:
            return ""

        # Step 1: Unicode normalization (NFD decomposition)
        # This separates base characters from combining marks (accents)
        normalized = unicodedata.normalize('NFD', text)

        # Step 2: Language-specific character replacements
        for original, replacement in self.character_replacements.items():
            normalized = normalized.replace(original, replacement)

        # Step 3: Remove punctuation and combining marks
        # Filter out punctuation characters and Unicode combining marks
        without_punctuation = ''.join(
            char for char in normalized
            if char not in string.punctuation and unicodedata.category(char) != 'Mn'
        )

        # Step 4: Case normalization to lowercase
        lowercased = without_punctuation.lower()

        # Step 5: Whitespace cleanup
        # Split and rejoin to normalize all whitespace to single spaces
        # and remove leading/trailing whitespace
        cleaned = ' '.join(lowercased.split())

        return cleaned