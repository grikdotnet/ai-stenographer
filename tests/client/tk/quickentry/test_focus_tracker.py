"""Tests for FocusTracker - Windows focus save/restore functionality."""
import pytest
from unittest.mock import MagicMock, patch


class TestFocusTracker:
    """Test FocusTracker save/restore focus behavior."""

    def test_save_focus_stores_foreground_window_handle(self):
        """save_focus() should call GetForegroundWindow and store the handle."""
        with patch('src.client.tk.quickentry.FocusTracker.ctypes') as mock_ctypes:
            mock_ctypes.windll.user32.GetForegroundWindow.return_value = 12345

            from src.client.tk.quickentry.FocusTracker import FocusTracker
            tracker = FocusTracker()
            tracker.save_focus()

            mock_ctypes.windll.user32.GetForegroundWindow.assert_called_once()
            assert tracker._saved_hwnd == 12345

    def test_restore_focus_calls_set_foreground_window(self):
        """restore_focus() should call SetForegroundWindow with saved handle."""
        with patch('src.client.tk.quickentry.FocusTracker.ctypes') as mock_ctypes:
            mock_ctypes.windll.user32.GetForegroundWindow.return_value = 12345
            mock_ctypes.windll.user32.SetForegroundWindow.return_value = 1

            from src.client.tk.quickentry.FocusTracker import FocusTracker
            tracker = FocusTracker()
            tracker.save_focus()
            result = tracker.restore_focus()

            mock_ctypes.windll.user32.SetForegroundWindow.assert_called_once_with(12345)
            assert result is True

    def test_restore_focus_returns_false_when_no_saved_focus(self):
        """restore_focus() should return False if no focus was saved."""
        with patch('src.client.tk.quickentry.FocusTracker.ctypes'):
            from src.client.tk.quickentry.FocusTracker import FocusTracker
            tracker = FocusTracker()

            result = tracker.restore_focus()

            assert result is False

    def test_restore_focus_returns_false_when_set_foreground_fails(self):
        """restore_focus() should return False if SetForegroundWindow fails."""
        with patch('src.client.tk.quickentry.FocusTracker.ctypes') as mock_ctypes:
            mock_ctypes.windll.user32.GetForegroundWindow.return_value = 12345
            mock_ctypes.windll.user32.SetForegroundWindow.return_value = 0

            from src.client.tk.quickentry.FocusTracker import FocusTracker
            tracker = FocusTracker()
            tracker.save_focus()
            result = tracker.restore_focus()

            assert result is False

    def test_clear_resets_saved_handle(self):
        """clear() should reset the saved window handle."""
        with patch('src.client.tk.quickentry.FocusTracker.ctypes') as mock_ctypes:
            mock_ctypes.windll.user32.GetForegroundWindow.return_value = 12345

            from src.client.tk.quickentry.FocusTracker import FocusTracker
            tracker = FocusTracker()
            tracker.save_focus()
            tracker.clear()

            assert tracker._saved_hwnd is None
            assert tracker.restore_focus() is False

    def test_verbose_logging(self):
        """Verbose mode should log focus operations."""
        with patch('src.client.tk.quickentry.FocusTracker.ctypes') as mock_ctypes:
            with patch('src.client.tk.quickentry.FocusTracker.logging') as mock_logging:
                mock_ctypes.windll.user32.GetForegroundWindow.return_value = 12345
                mock_ctypes.windll.user32.SetForegroundWindow.return_value = 1

                from src.client.tk.quickentry.FocusTracker import FocusTracker
                tracker = FocusTracker(verbose=True)
                tracker.save_focus()
                tracker.restore_focus()

                assert mock_logging.info.call_count >= 2
