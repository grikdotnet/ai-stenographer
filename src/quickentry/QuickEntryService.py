import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.RecognitionResultPublisher import RecognitionResultPublisher
    from src.quickentry.QuickEntryController import QuickEntryController
    import tkinter as tk


class QuickEntryService:

    def __init__(
        self,
        publisher: 'RecognitionResultPublisher',
        root: 'tk.Tk',
        hotkey: str = 'ctrl+space',
        enabled: bool = True,
        verbose: bool = False
    ) -> None:
        from src.quickentry.GlobalHotkeyListener import GlobalHotkeyListener
        from src.quickentry.FocusTracker import FocusTracker
        from src.quickentry.QuickEntrySubscriber import QuickEntrySubscriber
        from src.quickentry.QuickEntryController import QuickEntryController
        from src.gui.KeyboardSimulator import KeyboardSimulator

        self._verbose = verbose
        self._enabled = enabled

        self._focus_tracker = FocusTracker(verbose=verbose)
        self._keyboard = KeyboardSimulator()

        self._controller: Optional[QuickEntryController] = None

        def on_text_change(finalized: str, partial: str) -> None:
            if self._controller:
                self._controller.on_text_change(finalized, partial)

        self._subscriber = QuickEntrySubscriber(
            on_text_change=on_text_change,
            verbose=verbose
        )

        self._controller = QuickEntryController(
            subscriber=self._subscriber,
            focus_tracker=self._focus_tracker,
            keyboard_simulator=self._keyboard,
            root=root,
            verbose=verbose
        )

        self._hotkey_listener = GlobalHotkeyListener(
            hotkey=hotkey,
            callback=self._controller.on_hotkey,
            verbose=verbose
        )

        publisher.subscribe(self._subscriber)

        if verbose:
            logging.info(f"QuickEntryService: initialized with hotkey='{hotkey}', "
                        f"enabled={enabled}")

    @property
    def controller(self) -> 'QuickEntryController':
        return self._controller

    def start(self) -> None:
        if not self._enabled:
            if self._verbose:
                logging.info("QuickEntryService: disabled, not starting")
            return

        self._hotkey_listener.start()

        if self._verbose:
            logging.info("QuickEntryService: started")

    def stop(self) -> None:
        self._hotkey_listener.stop()

        if self._verbose:
            logging.info("QuickEntryService: stopped")
