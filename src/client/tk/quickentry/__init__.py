"""Quick Entry subsystem - global hotkey triggered popup for speech input.

Components:
- QuickEntryService: Facade that wires all components together
- QuickEntryController: Orchestrates the popup workflow
- QuickEntrySubscriber: Accumulates recognized text
- QuickEntryPopup: Floating popup window (GUI)
- GlobalHotkeyListener: Listens for global hotkeys via pynput
- FocusTracker: Saves/restores Windows focus via Win32 API
"""

from src.client.tk.quickentry.QuickEntryService import QuickEntryService

__all__ = ['QuickEntryService']
