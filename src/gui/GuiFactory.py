"""
GuiFactory provides shared GUI infrastructure for STT application.
"""
import tkinter as tk
from tkinter import scrolledtext, ttk


class GuiFactory:
    """Factory for creating GUI components with consistent configuration."""

    @staticmethod
    def create_window(title: str, geometry: str) -> tk.Tk:
        """
        Creates main Tk window with specified title and geometry.

        Args:
            title: Window title
            geometry: Window geometry (e.g., "800x600")

        Returns:
            Configured Tk window
        """
        window = tk.Tk()
        window.title(title)
        window.geometry(geometry)
        return window

    @staticmethod
    def create_dialog(parent: tk.Tk, title: str, geometry: str) -> tk.Toplevel:
        """
        Creates modal dialog window.

        Args:
            parent: Parent Tk window
            title: Dialog title
            geometry: Dialog geometry (e.g., "400x300")

        Returns:
            Configured modal Toplevel dialog
        """
        dialog = tk.Toplevel(parent)
        dialog.title(title)
        dialog.geometry(geometry)
        dialog.transient(parent)
        dialog.grab_set()
        return dialog

    @staticmethod
    def create_scrolled_text(parent, **kwargs) -> scrolledtext.ScrolledText:
        """
        Creates scrolled text widget.

        Args:
            parent: Parent widget
            **kwargs: Additional arguments passed to ScrolledText

        Returns:
            Configured ScrolledText widget
        """
        return scrolledtext.ScrolledText(parent, **kwargs)

    @staticmethod
    def create_progress_bar(parent, **kwargs) -> ttk.Progressbar:
        """
        Creates progress bar widget with 0-100 range.

        Args:
            parent: Parent widget
            **kwargs: Additional arguments passed to Progressbar

        Returns:
            Configured Progressbar widget
        """
        # Set default maximum to 100 if not specified
        if 'maximum' not in kwargs:
            kwargs['maximum'] = 100

        return ttk.Progressbar(parent, **kwargs)
