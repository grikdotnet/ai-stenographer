import logging
import threading
from typing import Callable, Optional, Set, Union
from pynput import keyboard


class GlobalHotkeyListener:

    _MODIFIER_MAP = {
        'ctrl': keyboard.Key.ctrl_l,
        'alt': keyboard.Key.alt_l,
        'shift': keyboard.Key.shift_l,
    }

    _SPECIAL_KEY_MAP = {
        'space': keyboard.Key.space,
        'enter': keyboard.Key.enter,
        'tab': keyboard.Key.tab,
        'escape': keyboard.Key.esc,
        'esc': keyboard.Key.esc,
        'f1': keyboard.Key.f1,
        'f2': keyboard.Key.f2,
        'f3': keyboard.Key.f3,
        'f4': keyboard.Key.f4,
        'f5': keyboard.Key.f5,
        'f6': keyboard.Key.f6,
        'f7': keyboard.Key.f7,
        'f8': keyboard.Key.f8,
        'f9': keyboard.Key.f9,
        'f10': keyboard.Key.f10,
        'f11': keyboard.Key.f11,
        'f12': keyboard.Key.f12,
    }

    def __init__(
        self,
        hotkey: str,
        callback: Callable[[], None],
        verbose: bool = False
    ) -> None:
        self._callback = callback
        self._verbose = verbose
        self._listener: Optional[keyboard.Listener] = None
        self._pressed_modifiers: Set[keyboard.Key] = set()
        self._lock = threading.Lock()

        self._target_key: Union[keyboard.Key, keyboard.KeyCode]
        self._required_modifiers: Set[keyboard.Key]
        self._target_key, self._required_modifiers = self._parse_hotkey(hotkey)

        if verbose:
            logging.info(f"GlobalHotkeyListener: parsed '{hotkey}' -> "
                        f"key={self._target_key}, modifiers={self._required_modifiers}")

    def _parse_hotkey(self, hotkey: str) -> tuple:
        parts = hotkey.lower().split('+')
        modifiers: Set[keyboard.Key] = set()
        target_key = None

        for part in parts:
            part = part.strip()

            if part in self._MODIFIER_MAP:
                modifiers.add(self._MODIFIER_MAP[part])
            elif part in self._SPECIAL_KEY_MAP:
                if target_key is not None:
                    raise ValueError(f"Multiple non-modifier keys in hotkey: {hotkey}")
                target_key = self._SPECIAL_KEY_MAP[part]
            elif len(part) == 1:
                if target_key is not None:
                    raise ValueError(f"Multiple non-modifier keys in hotkey: {hotkey}")
                target_key = keyboard.KeyCode.from_char(part)
            else:
                raise ValueError(f"Unknown key in hotkey: {part}")

        if target_key is None:
            raise ValueError(f"No target key found in hotkey: {hotkey}")

        return target_key, modifiers

    def start(self) -> None:
        if self._listener is not None:
            return

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

        if self._verbose:
            logging.info("GlobalHotkeyListener: started")

    def stop(self) -> None:
        if self._listener is None:
            return

        self._listener.stop()
        self._listener = None

        with self._lock:
            self._pressed_modifiers.clear()

        if self._verbose:
            logging.info("GlobalHotkeyListener: stopped")

    def _on_press(self, key) -> None:
        with self._lock:
            if self._is_modifier(key):
                normalized = self._normalize_modifier(key)
                self._pressed_modifiers.add(normalized)
                return

            if self._matches_target(key):
                if self._required_modifiers.issubset(self._pressed_modifiers):
                    if self._verbose:
                        logging.info("GlobalHotkeyListener: hotkey triggered")
                    # Release lock before callback to prevent deadlocks
                    self._lock.release()
                    try:
                        self._callback()
                    finally:
                        self._lock.acquire()

    def _on_release(self, key) -> None:
        with self._lock:
            if self._is_modifier(key):
                normalized = self._normalize_modifier(key)
                self._pressed_modifiers.discard(normalized)

    def _is_modifier(self, key) -> bool:
        modifier_keys = {
            keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
            keyboard.Key.alt_l, keyboard.Key.alt_r,
            keyboard.Key.shift_l, keyboard.Key.shift_r,
            keyboard.Key.cmd_l, keyboard.Key.cmd_r,
        }
        return key in modifier_keys

    def _normalize_modifier(self, key) -> keyboard.Key:
        """Normalize left/right modifier variants to left variant."""
        mapping = {
            keyboard.Key.ctrl_r: keyboard.Key.ctrl_l,
            keyboard.Key.alt_r: keyboard.Key.alt_l,
            keyboard.Key.shift_r: keyboard.Key.shift_l,
            keyboard.Key.cmd_r: keyboard.Key.cmd_l,
        }
        return mapping.get(key, key)

    def _matches_target(self, key) -> bool:
        if key == self._target_key:
            return True

        if isinstance(self._target_key, keyboard.KeyCode) and isinstance(key, keyboard.KeyCode):
            return key.char == self._target_key.char

        return False
