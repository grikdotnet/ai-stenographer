# tests/test_text_normalizer.py
import unittest
from src.postprocessing.TextNormalizer import TextNormalizer


class TestTextNormalization(unittest.TestCase):
    """Test the TextNormalizer class for text normalization."""

    def setUp(self):
        """Set up test fixtures."""
        self.text_normalizer = TextNormalizer()

    def test_normalization(self):
        """Test case normalization."""
        result = self.text_normalizer.normalize_text("Hello.")
        self.assertEqual(result, "hello")

        result = self.text_normalizer.normalize_text("Hello. One-two-three")
        self.assertEqual(result, "hello one two three")

        result = self.text_normalizer.normalize_text("Hello")
        self.assertEqual(result, "hello")

        result = self.text_normalizer.normalize_text("чёрный")
        self.assertEqual(result, "черный")

        # Test with accented characters
        result = self.text_normalizer.normalize_text("café")
        self.assertEqual(result, "cafe")

        result = self.text_normalizer.normalize_text("Weiß")
        self.assertEqual(result, "weiss")

        result = self.text_normalizer.normalize_text("Straße")
        self.assertEqual(result, "strasse")

        result = self.text_normalizer.normalize_text("hello  world")
        self.assertEqual(result, "hello world")

        result = self.text_normalizer.normalize_text("  hello   world  ")
        self.assertEqual(result, "hello world")

        result = self.text_normalizer.normalize_text("hello\t\nworld")
        self.assertEqual(result, "hello world")

        """Test complex text with multiple normalization needs."""
        result = self.text_normalizer.normalize_text("HELLO, Wörld!  Café?")
        self.assertEqual(result, "hello world cafe")

        result = self.text_normalizer.normalize_text("Straße  123!?")
        self.assertEqual(result, "strasse 123")

    def test_edge_cases(self):
        """Test edge cases for normalize_text()."""
        result = self.text_normalizer.normalize_text("")
        self.assertEqual(result, "")

        result = self.text_normalizer.normalize_text("!?.,")
        self.assertEqual(result, "")

        result = self.text_normalizer.normalize_text("   ")
        self.assertEqual(result, "")

        result = self.text_normalizer.normalize_text("  !?  ")
        self.assertEqual(result, "")


if __name__ == '__main__':
    unittest.main()