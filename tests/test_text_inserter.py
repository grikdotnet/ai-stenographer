import unittest
import threading
from unittest.mock import Mock
from src.types import RecognitionResult


class TestTextInserter(unittest.TestCase):

    def setUp(self):
        from src.gui.TextInserter import TextInserter
        self.mock_keyboard = Mock()
        self.inserter = TextInserter(self.mock_keyboard, verbose=False)

    def _make_result(self, text: str) -> RecognitionResult:
        """Helper to create RecognitionResult for testing."""
        return RecognitionResult(
            text=text,
            start_time=0.0,
            end_time=1.0,
            chunk_ids=[1]
        )

    # --- Enable/Disable State Tests ---

    def test_starts_disabled_by_default(self):
        self.assertFalse(self.inserter.is_enabled())

    def test_enable_sets_enabled_state(self):
        self.inserter.enable()
        self.assertTrue(self.inserter.is_enabled())

    def test_disable_clears_enabled_state(self):
        """Verify disable() turns off insertion."""
        self.inserter.enable()
        self.inserter.disable()
        self.assertFalse(self.inserter.is_enabled())

    # --- on_finalization Tests ---

    def test_on_finalization_inserts_text_when_enabled(self):
        self.inserter.enable()
        result = self._make_result("hello world")
        self.inserter.on_finalization(result)
        self.mock_keyboard.type_text.assert_called_once_with("hello world ")

    def test_on_finalization_does_nothing_when_disabled(self):
        result = self._make_result("hello world")
        self.inserter.on_finalization(result)
        self.mock_keyboard.type_text.assert_not_called()

    def test_on_finalization_adds_trailing_space(self):
        """Verify trailing space is added after inserted text."""
        self.inserter.enable()
        result = self._make_result("word")
        self.inserter.on_finalization(result)
        self.mock_keyboard.type_text.assert_called_once_with("word ")

    def test_on_finalization_handles_flush_status(self):
        self.inserter.enable()
        result = self._make_result("end of speech")

        self.inserter.on_finalization(result)

        self.mock_keyboard.type_text.assert_called_once_with("end of speech ")

    def test_on_finalization_skips_empty_text(self):
        self.inserter.enable()
        result = self._make_result("")
        self.inserter.on_finalization(result)
        self.mock_keyboard.type_text.assert_not_called()

    def test_on_finalization_skips_whitespace_only_text(self):
        self.inserter.enable()
        result = self._make_result("   \t\n  ")
        self.inserter.on_finalization(result)
        self.mock_keyboard.type_text.assert_not_called()

    def test_on_partial_update_does_nothing_when_enabled(self):
        self.inserter.enable()
        result = self._make_result("unstable")
        self.inserter.on_partial_update(result)
        self.mock_keyboard.type_text.assert_not_called()

    def test_on_partial_update_does_nothing_when_disabled(self):
        result = self._make_result("unstable")
        self.inserter.on_partial_update(result)
        self.mock_keyboard.type_text.assert_not_called()

    # --- Thread Safety Tests ---

    def test_toggle_during_finalization_is_safe(self):
        """Verify toggling enabled while finalization is running is safe."""
        errors = []

        def toggle_loop():
            try:
                for _ in range(100):
                    self.inserter.enable()
                    self.inserter.disable()
            except Exception as e:
                errors.append(e)

        def finalization_loop():
            try:
                for i in range(100):
                    result = self._make_result(f"text{i}")
                    self.inserter.on_finalization(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=toggle_loop),
            threading.Thread(target=finalization_loop)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


class TestTextInserterProtocolCompliance(unittest.TestCase):
    """Test that TextInserter correctly implements TextRecognitionSubscriber."""

    def test_has_on_partial_update_method(self):
        from src.gui.TextInserter import TextInserter
        mock_keyboard = Mock()
        inserter = TextInserter(mock_keyboard)

        # Should be callable with RecognitionResult
        result = RecognitionResult(
            text="test", start_time=0.0, end_time=1.0,
            chunk_ids=[1]
        )
        inserter.on_partial_update(result)  # Should not raise


if __name__ == '__main__':
    unittest.main()
