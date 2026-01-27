import logging
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber
    from src.quickentry.FocusTracker import FocusTracker
    from src.quickentry.QuickEntryPopup import QuickEntryPopup
    from src.KeyboardSimulator import KeyboardSimulator
    import tkinter as tk


class QuickEntryController:

    _INSERT_DELAY_MS = 50

    def __init__(
        self,
        subscriber: 'QuickEntrySubscriber',
        focus_tracker: 'FocusTracker',
        keyboard_simulator: 'KeyboardSimulator',
        root: 'tk.Tk',
        verbose: bool = False
    ) -> None:
        self._subscriber = subscriber
        self._focus_tracker = focus_tracker
        self._keyboard = keyboard_simulator
        self._root = root
        self._verbose = verbose

        self._popup: Optional['QuickEntryPopup'] = None
        self._lock = threading.Lock()
        self._is_showing = False

    @property
    def is_showing(self) -> bool:
        with self._lock:
            return self._is_showing

    def on_hotkey(self) -> None:
        # Schedule on main thread to avoid tkinter threading issues
        self._root.after(0, self._toggle_popup)

    def _toggle_popup(self) -> None:
        with self._lock:
            if self._is_showing:
                self._hide_and_cancel_locked()
            else:
                self._show_popup_locked()

    def _show_popup(self) -> None:
        with self._lock:
            self._show_popup_locked()

    def _show_popup_locked(self) -> None:
        self._focus_tracker.save_focus()
        self._subscriber.activate()

        if self._popup is None:
            from src.quickentry.QuickEntryPopup import QuickEntryPopup
            self._popup = QuickEntryPopup(
                self._root,
                on_submit=self._on_submit,
                on_cancel=self._on_cancel,
                verbose=self._verbose
            )

        self._popup.set_text("")
        self._popup.show()
        self._is_showing = True

        if self._verbose:
            logging.info("QuickEntryController: popup shown")

    def _hide_and_cancel_locked(self) -> None:
        self._subscriber.deactivate()

        if self._popup:
            self._popup.hide()

        self._is_showing = False
        self._focus_tracker.restore_focus()

        if self._verbose:
            logging.info("QuickEntryController: popup hidden (cancelled)")

    def _on_submit(self) -> None:
        text = self._subscriber.get_accumulated_text()
        self._subscriber.deactivate()

        with self._lock:
            if self._popup:
                self._popup.hide()
            self._is_showing = False

        self._focus_tracker.restore_focus()

        if text.strip():
            self._root.after(self._INSERT_DELAY_MS, lambda: self._keyboard.type_text(text))

        if self._verbose:
            logging.info(f"QuickEntryController: submitted '{text}'")

    def _on_cancel(self) -> None:
        with self._lock:
            self._hide_and_cancel_locked()

    def on_text_change(self, finalized: str, partial: str) -> None:
        if partial:
            display_text = f"{finalized} {partial}".strip()
        else:
            display_text = finalized

        # Schedule update on main thread
        self._root.after(0, lambda: self._update_display(display_text))

    def _update_display(self, text: str) -> None:
        with self._lock:
            if self._popup and self._is_showing:
                self._popup.set_text(text)
