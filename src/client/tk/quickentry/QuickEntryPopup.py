import logging
import platform
import tkinter as tk
from typing import Callable, Optional

from src.client.tk.quickentry.windows_effects import apply_all_effects


class QuickEntryPopup:
    """
    Floating popup window for quick text entry via speech.

    Displays transcribed text in a modern UI with acrylic blur effect.
    Supports Enter to submit and Escape to cancel.
    """

    WIDTH = 650
    HEIGHT = 100
    MAX_HEIGHT = 400
    CORNER_RADIUS = 16

    # Colors for acrylic effect (opaque backgrounds for clickability)
    BG_COLOR = '#1E1E1E'  # Dark background
    TEXT_COLOR = '#FFFFFF'
    HINT_COLOR = '#AAAAAA'  # Slightly brighter for visibility
    BORDER_COLOR = '#3D3D3D'  # Subtle border
    ACCENT_COLOR = '#4A9EFF'  # Blue accent for button hover
    BUTTON_BG = '#3D3D3D'  # Button background

    # Fonts
    FONT_TITLE = ('Segoe UI Semibold', 11)
    FONT_TEXT = ('Segoe UI', 13)
    FONT_HINT = ('Segoe UI', 10)
    FONT_ICON = ('Segoe UI Emoji', 14)
    FONT_BUTTON = ('Segoe UI', 12)

    # Padding
    PADDING_X = 16
    PADDING_Y = 12
    ROW1_HEIGHT = 36
    ROW2_HEIGHT = 30

    def __init__(
        self,
        root: tk.Tk,
        on_submit: Callable[[], None],
        on_cancel: Callable[[], None],
        verbose: bool = False
    ) -> None:
        self._root = root
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self._verbose = verbose
        self._current_text = ""
        self._current_height = self.HEIGHT
        self._effects_applied = False

        self._window: Optional[tk.Toplevel] = tk.Toplevel(root)
        self._window.withdraw()
        self._window.overrideredirect(True)
        self._window.attributes('-topmost', True)
        self._window.configure(bg=self.BG_COLOR)

        self._window.geometry(f"{self.WIDTH}x{self.HEIGHT}")

        self._create_layout()
        self._bind_events()

        if verbose:
            logging.info("QuickEntryPopup: created")

    def _create_layout(self) -> None:
        """
        Create the layout with three rows:
        - Row 1: icon, title, hints, submit button (fixed height)
        - Row 2: separator line
        - Row 3: text area that grows dynamically
        """
        # Main container frame
        self._frame = tk.Frame(
            self._window,
            bg=self.BG_COLOR,
            highlightthickness=1,
            highlightbackground=self.BORDER_COLOR
        )
        self._frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Row 1: Icon, Title, Hints, Buttons (fixed height header)
        self._row1 = tk.Frame(self._frame, bg=self.BG_COLOR, height=self.ROW1_HEIGHT)
        self._row1.pack(fill=tk.X, padx=self.PADDING_X, pady=(self.PADDING_Y, 0))
        self._row1.pack_propagate(False)

        # Mic icon (static)
        self._icon_label = tk.Label(
            self._row1,
            text="\U0001F399",   # Studio microphone emoji
            font=self.FONT_ICON,
            fg=self.TEXT_COLOR,
            bg=self.BG_COLOR
        )
        self._icon_label.pack(side=tk.LEFT, padx=(0, 8))

        # Title
        self._title_label = tk.Label(
            self._row1,
            text="AI Stenographer",
            font=self.FONT_TITLE,
            fg=self.HINT_COLOR,
            bg=self.BG_COLOR
        )
        self._title_label.pack(side=tk.LEFT, padx=(0, 12))

        # Submit button (on the right)
        self._submit_btn = tk.Label(
            self._row1,
            text=" \u23CE ",  # Return symbol
            font=self.FONT_BUTTON,
            fg=self.TEXT_COLOR,
            bg=self.BUTTON_BG,
            cursor='hand2',
            padx=8,
            pady=4
        )
        self._submit_btn.pack(side=tk.RIGHT, padx=(8, 0))
        self._submit_btn.bind('<Button-1>', self._handle_submit)
        self._submit_btn.bind('<Enter>', self._on_submit_hover)
        self._submit_btn.bind('<Leave>', self._on_submit_leave)

        # Hints label (right-aligned, before button)
        self._hints_label = tk.Label(
            self._row1,
            text="Enter \u23CE send    Esc \u2715",
            font=self.FONT_HINT,
            fg=self.HINT_COLOR,
            bg=self.BG_COLOR
        )
        self._hints_label.pack(side=tk.RIGHT, padx=(0, 12))

        # Separator line
        self._separator = tk.Frame(self._frame, bg=self.BORDER_COLOR, height=1)
        self._separator.pack(fill=tk.X, padx=self.PADDING_X, pady=(4, 0))

        # Row 2: Text area (expands vertically)
        self._row2 = tk.Frame(self._frame, bg=self.BG_COLOR)
        self._row2.pack(fill=tk.BOTH, expand=True, padx=self.PADDING_X, pady=(4, self.PADDING_Y))

        # Text label (multiline, grows with content)
        self._text_label = tk.Label(
            self._row2,
            text="",
            font=self.FONT_TEXT,
            fg=self.TEXT_COLOR,
            bg=self.BG_COLOR,
            anchor='nw',
            justify='left',
            wraplength=self.WIDTH - 2 * self.PADDING_X - 10
        )
        self._text_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _bind_events(self) -> None:
        """Bind keyboard and window events."""
        self._window.bind('<Return>', self._handle_submit)
        self._window.bind('<Escape>', self._handle_cancel)
        self._window.protocol('WM_DELETE_WINDOW', self._handle_cancel)

    def _on_submit_hover(self, event=None) -> None:
        """Handle mouse hover on submit button."""
        self._submit_btn.configure(bg=self.ACCENT_COLOR)

    def _on_submit_leave(self, event=None) -> None:
        """Handle mouse leave on submit button."""
        self._submit_btn.configure(bg=self.BUTTON_BG)

    def _calculate_required_height(self) -> int:
        """
        Calculate total window height required for current text content.

        Uses tkinter's geometry manager to determine the label's required height
        after text wrapping, then adds fixed layout overhead.

        Returns:
            Required window height clamped to MAX_HEIGHT.
        """
        self._window.update_idletasks()
        text_height = self._text_label.winfo_reqheight()

        # Fixed overhead: PADDING_Y(top) + ROW1_HEIGHT + separator padding(4)
        # + separator(1) + row2 padding(4) + PADDING_Y(bottom)
        fixed_overhead = self.PADDING_Y + self.ROW1_HEIGHT + 4 + 1 + 4 + self.PADDING_Y

        required_height = fixed_overhead + text_height
        return min(required_height, self.MAX_HEIGHT)

    def _resize_if_needed(self) -> None:
        """
        Resize window if text content requires more height.

        Height only increases, never decreases during a session.
        Reapplies Windows effects after resize to maintain rounded corners.
        """
        if self._window is None:
            return

        required_height = self._calculate_required_height()

        # Only grow, never shrink during session
        if required_height <= self._current_height:
            return

        self._current_height = required_height

        # Update geometry keeping current position
        x = self._window.winfo_x()
        y = self._window.winfo_y()
        self._window.geometry(f"{self.WIDTH}x{self._current_height}+{x}+{y}")

        # Re-apply Windows effects with new dimensions
        self._effects_applied = False
        self._apply_windows_effects()

        if self._verbose:
            logging.info(f"QuickEntryPopup: Auto-resized to height {self._current_height}")

    def _apply_windows_effects(self) -> None:
        """Apply Windows DWM effects (rounded corners, acrylic, shadow)."""
        if self._effects_applied:
            return

        if platform.system() != 'Windows':
            logging.debug("QuickEntryPopup: Not on Windows, skipping effects")
            return

        try:
            results = apply_all_effects(
                window=self._window,
                width=self.WIDTH,
                height=self._current_height,
                corner_radius=self.CORNER_RADIUS,
                tint_color=(30, 30, 30),  # Dark tint
                opacity=220  # ~86% opacity
            )

            self._effects_applied = True

            if self._verbose:
                logging.info(f"QuickEntryPopup: Windows effects - {results}")

        except Exception as e:
            logging.warning(f"QuickEntryPopup: Failed to apply Windows effects: {e}")

    def _center_window(self) -> None:
        """Center the window on screen, slightly above center."""
        self._window.update_idletasks()

        screen_width = self._window.winfo_screenwidth()
        screen_height = self._window.winfo_screenheight()

        x = (screen_width - self.WIDTH) // 2
        y = (screen_height - self._current_height) // 2 - 100  # Slightly above center

        self._window.geometry(f"{self.WIDTH}x{self._current_height}+{x}+{y}")

    def _handle_submit(self, event=None) -> None:
        """Handle submit action (Enter key or button click)."""
        if self._verbose:
            logging.info("QuickEntryPopup: Submit triggered")
        self._on_submit()

    def _handle_cancel(self, event=None) -> None:
        """Handle cancel action (Escape key)."""
        if self._verbose:
            logging.info("QuickEntryPopup: Cancel triggered")
        self._on_cancel()

    def show(self) -> None:
        """Show the popup window."""
        if self._window is None:
            return

        self._center_window()
        self._window.deiconify()
        self._window.lift()
        self._window.focus_force()

        # Apply Windows effects after window is visible
        self._window.after(10, self._apply_windows_effects)

        if self._verbose:
            logging.info("QuickEntryPopup: shown")

    def hide(self) -> None:
        """Hide the popup window and reset size to original."""
        if self._window is None:
            return

        # Reset to original size for next show
        self._current_height = self.HEIGHT
        self._window.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self._effects_applied = False

        self._window.withdraw()

        if self._verbose:
            logging.info("QuickEntryPopup: hidden")

    def set_text(self, text: str) -> None:
        """Update the displayed transcribed text with auto-resize."""
        self._current_text = text
        display_text = text + "_"  # Cursor indicator
        self._text_label.config(text=display_text)
        self._resize_if_needed()

    def get_text(self) -> str:
        """Get the current displayed text."""
        return self._current_text

    def is_visible(self) -> bool:
        """Check if the popup is currently visible."""
        if self._window is None:
            return False
        try:
            return self._window.winfo_viewable() == 1
        except tk.TclError:
            return False

    def destroy(self) -> None:
        """Destroy the popup window and clean up resources."""
        if self._window is not None:
            try:
                self._window.destroy()
            except tk.TclError:
                pass
            self._window = None

        if self._verbose:
            logging.info("QuickEntryPopup: destroyed")
