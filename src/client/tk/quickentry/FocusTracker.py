import ctypes
import logging
from typing import Optional


class FocusTracker:

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        self._saved_hwnd: Optional[int] = None

    def save_focus(self) -> None:
        user32 = ctypes.windll.user32
        self._saved_hwnd = user32.GetForegroundWindow()
        if self._verbose:
            logging.info(f"FocusTracker: saved hwnd={self._saved_hwnd}")

    def restore_focus(self) -> bool:
        if self._saved_hwnd is None:
            return False

        user32 = ctypes.windll.user32
        result = user32.SetForegroundWindow(self._saved_hwnd)
        if self._verbose:
            logging.info(f"FocusTracker: restore hwnd={self._saved_hwnd} result={result}")
        return bool(result)

    def clear(self) -> None:
        self._saved_hwnd = None
