import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Dict, Optional, Protocol, TYPE_CHECKING
import webbrowser
from src.ApplicationState import ApplicationState
from src.client.tk.gui.HeaderPanel import HeaderPanel
from src.client.tk.gui.ControlPanel import ControlPanel
from src.client.tk.gui.InsertionModePanel import InsertionModePanel
from src.controllers.PauseController import PauseController
from src.client.tk.gui.TextDisplayWidget import TextDisplayWidget
from src.client.tk.gui.TextFormatter import TextFormatter

if TYPE_CHECKING:
    from src.controllers.InsertionController import InsertionController

GITHUB_ISSUES_URL = "https://github.com/grikdotnet/ai-stenographer/issues"


class SupportsPauseControl(Protocol):
    """Structural protocol for pause/resume control.

    Any object implementing pause(), resume(), and toggle() satisfies this protocol.
    """

    def pause(self) -> None:
        """Pause the application."""
        ...

    def resume(self) -> None:
        """Resume the application."""
        ...

    def toggle(self) -> None:
        """Toggle between paused and running states."""
        ...


class ApplicationWindow:
    """Top-level application window.
    Owns tkinter root and creates all GUI components directly.
    """

    def __init__(
        self,
        app_state: ApplicationState,
        config: Dict,
        verbose: bool = False,
        insertion_controller: Optional['InsertionController'] = None,
        pause_controller: Optional[SupportsPauseControl] = None
    ):
        """Initialize TK root and all GUI components.

        Args:
            app_state: ApplicationState instance for state management
            config: Application configuration dictionary
            verbose: Enable verbose logging
            insertion_controller: Optional controller for text insertion toggle button
            pause_controller: Optional pause controller; constructs PauseController(app_state) when None
        """
        self.root = tk.Tk()
        self.root.title("Speech-to-Text Display")
        self.root.geometry("800x600")

        app_state.setTkRoot(self.root)

        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create header panel first (it creates internal frame)
        header_panel = HeaderPanel(main_frame)

        # Create report button inside header panel's frame
        report_button = ttk.Button(
            header_panel.frame,
            text='⚠️ Report',
            command=lambda: webbrowser.open(GITHUB_ISSUES_URL)
        )
        report_button.pack(side=tk.RIGHT)
        self._create_tooltip(report_button, "Report inappropriate content and issues")

        # Create control panel
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        effective_pause_controller = pause_controller if pause_controller is not None else PauseController(app_state)
        control_panel = ControlPanel(
            button_frame,
            app_state,
            effective_pause_controller
        )
        control_panel.pack(side=tk.LEFT, anchor='w')

        # Create insertion mode panel if controller provided
        if insertion_controller is not None:
            insertion_panel = InsertionModePanel(button_frame, insertion_controller)
            insertion_panel.pack(side=tk.LEFT, anchor='w', padx=(10, 0))

        status_label = tk.Label(
            button_frame,
            text="Press Cntrl+Space to insert text at cursor in other applications",
            font=("Arial", 10)
        )
        status_label.pack(side=tk.RIGHT, padx=(20, 0))

        text_widget = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            width=80,
            height=30,
            font=("Arial", 12),
            bg="white",
            fg="black"
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.config(state=tk.NORMAL)

        # Configure text styles for preliminary and final display
        text_widget.tag_configure(
            "preliminary",
            foreground="gray",
            font=("TkDefaultFont", 10, "normal")
        )
        text_widget.tag_configure(
            "final",
            foreground="black",
            font=("TkDefaultFont", 10, "normal")
        )

        # display (view) → formatter (controller)
        self.display = TextDisplayWidget(text_widget, self.root, verbose=verbose)
        self.formatter = TextFormatter(display=self.display, verbose=verbose)

    def get_formatter(self) -> TextFormatter:
        """Returns TextFormatter for pipeline integration.

        Returns:
            TextFormatter instance for TextMatcher to call
        """
        return self.formatter

    def get_display(self) -> TextDisplayWidget:
        """Returns TextDisplayWidget for testing.

        Returns:
            TextDisplayWidget instance
        """
        return self.display

    def get_root(self) -> tk.Tk:
        """Access root window.

        Returns:
            tk.Tk root window instance
        """
        return self.root

    def _create_tooltip(self, widget: tk.Widget, text: str) -> None:
        """
        Create a tooltip that appears on hover.

        Args:
            widget: Widget to attach tooltip to
            text: Tooltip text to display
        """
        tooltip = None

        def show_tooltip(event):
            nonlocal tooltip
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + widget.winfo_height() + 5
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                tooltip,
                text=text,
                background="#FFFFE0",
                relief=tk.SOLID,
                borderwidth=1,
                font=("Arial", 9)
            )
            label.pack()

        def hide_tooltip(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)
