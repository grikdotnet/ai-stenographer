"""Tests for GlobalHotkeyListener - global hotkey detection."""
import pytest
from unittest.mock import MagicMock, patch, call
from pynput import keyboard


class TestGlobalHotkeyListener:
    """Test GlobalHotkeyListener hotkey parsing and callback invocation."""

    def test_parse_single_key_f2(self):
        """Should parse 'F2' as a single function key."""
        from src.client.tk.quickentry.GlobalHotkeyListener import GlobalHotkeyListener

        listener = GlobalHotkeyListener('F2', lambda: None)

        assert listener._target_key == keyboard.Key.f2
        assert listener._required_modifiers == set()

    def test_parse_ctrl_space(self):
        """Should parse 'ctrl+space' as Ctrl modifier + space key."""
        from src.client.tk.quickentry.GlobalHotkeyListener import GlobalHotkeyListener

        listener = GlobalHotkeyListener('ctrl+space', lambda: None)

        assert listener._target_key == keyboard.Key.space
        assert keyboard.Key.ctrl_l in listener._required_modifiers or \
               keyboard.Key.ctrl in listener._required_modifiers

    def test_parse_alt_enter(self):
        """Should parse 'alt+enter' as Alt modifier + enter key."""
        from src.client.tk.quickentry.GlobalHotkeyListener import GlobalHotkeyListener

        listener = GlobalHotkeyListener('alt+enter', lambda: None)

        assert listener._target_key == keyboard.Key.enter
        assert keyboard.Key.alt_l in listener._required_modifiers or \
               keyboard.Key.alt in listener._required_modifiers

    def test_callback_invoked_on_hotkey(self):
        """Callback should be invoked when hotkey combination is detected."""
        from src.client.tk.quickentry.GlobalHotkeyListener import GlobalHotkeyListener

        callback = MagicMock()
        listener = GlobalHotkeyListener('F2', callback)

        # Simulate F2 key press
        listener._on_press(keyboard.Key.f2)

        callback.assert_called_once()

    def test_callback_not_invoked_on_wrong_key(self):
        """Callback should not be invoked for non-matching keys."""
        from src.client.tk.quickentry.GlobalHotkeyListener import GlobalHotkeyListener

        callback = MagicMock()
        listener = GlobalHotkeyListener('F2', callback)

        # Simulate F3 key press
        listener._on_press(keyboard.Key.f3)

        callback.assert_not_called()

    def test_modifier_tracking_for_ctrl_space(self):
        """Should track Ctrl modifier and invoke callback on Ctrl+Space."""
        from src.client.tk.quickentry.GlobalHotkeyListener import GlobalHotkeyListener

        callback = MagicMock()
        listener = GlobalHotkeyListener('ctrl+space', callback)

        # Press Ctrl (should not trigger)
        listener._on_press(keyboard.Key.ctrl_l)
        callback.assert_not_called()

        # Press Space while Ctrl is held (should trigger)
        listener._on_press(keyboard.Key.space)
        callback.assert_called_once()

    def test_modifier_released_clears_state(self):
        """Releasing modifier should clear tracked state."""
        from src.client.tk.quickentry.GlobalHotkeyListener import GlobalHotkeyListener

        callback = MagicMock()
        listener = GlobalHotkeyListener('ctrl+space', callback)

        # Press and release Ctrl
        listener._on_press(keyboard.Key.ctrl_l)
        listener._on_release(keyboard.Key.ctrl_l)

        # Press Space without Ctrl (should not trigger)
        listener._on_press(keyboard.Key.space)
        callback.assert_not_called()

    def test_start_creates_listener(self):
        """start() should create and start pynput Listener."""
        from src.client.tk.quickentry.GlobalHotkeyListener import GlobalHotkeyListener

        with patch('src.client.tk.quickentry.GlobalHotkeyListener.keyboard.Listener') as MockListener:
            mock_instance = MagicMock()
            MockListener.return_value = mock_instance

            listener = GlobalHotkeyListener('F2', lambda: None)
            listener.start()

            MockListener.assert_called_once()
            mock_instance.start.assert_called_once()

    def test_stop_stops_listener(self):
        """stop() should stop the pynput Listener."""
        from src.client.tk.quickentry.GlobalHotkeyListener import GlobalHotkeyListener

        with patch('src.client.tk.quickentry.GlobalHotkeyListener.keyboard.Listener') as MockListener:
            mock_instance = MagicMock()
            MockListener.return_value = mock_instance

            listener = GlobalHotkeyListener('F2', lambda: None)
            listener.start()
            listener.stop()

            mock_instance.stop.assert_called_once()

    def test_invalid_hotkey_raises_error(self):
        """Invalid hotkey string should raise ValueError."""
        from src.client.tk.quickentry.GlobalHotkeyListener import GlobalHotkeyListener

        with pytest.raises(ValueError):
            GlobalHotkeyListener('invalid+hotkey+combo', lambda: None)

    def test_handles_character_keys(self):
        """Should handle character keys like 'a' in hotkey combinations."""
        from src.client.tk.quickentry.GlobalHotkeyListener import GlobalHotkeyListener

        callback = MagicMock()
        listener = GlobalHotkeyListener('ctrl+a', callback)

        # Press Ctrl
        listener._on_press(keyboard.Key.ctrl_l)

        # Press 'a' as KeyCode
        listener._on_press(keyboard.KeyCode.from_char('a'))
        callback.assert_called_once()
