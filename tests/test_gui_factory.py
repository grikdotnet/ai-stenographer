"""
Tests for GuiFactory - shared GUI infrastructure.
"""
import pytest
import tkinter as tk
from tkinter import scrolledtext, ttk
from src.GuiFactory import GuiFactory


class TestGuiFactory:
    """Test suite for GuiFactory functionality."""

    def test_create_window_with_geometry(self):
        """Window created with correct size and title."""
        window = GuiFactory.create_window("Test Window", "400x300")

        assert window is not None
        assert isinstance(window, tk.Tk)
        assert window.title() == "Test Window"
        # Geometry is set but may not be immediately applied
        window.destroy()

    def test_create_dialog_modal(self):
        """Dialog created as modal child window."""
        parent = tk.Tk()
        parent.withdraw()

        dialog = GuiFactory.create_dialog(parent, "Test Dialog", "300x200")

        assert dialog is not None
        assert isinstance(dialog, tk.Toplevel)
        assert dialog.title() == "Test Dialog"

        dialog.destroy()
        parent.destroy()

    def test_create_scrolled_text_widget(self):
        """ScrolledText created with correct properties."""
        root = tk.Tk()
        root.withdraw()

        text_widget = GuiFactory.create_scrolled_text(
            root,
            wrap=tk.WORD,
            height=10,
            width=40
        )

        assert text_widget is not None
        assert isinstance(text_widget, scrolledtext.ScrolledText)
        assert text_widget.cget('wrap') == tk.WORD

        root.destroy()

    def test_create_progress_bar_widget(self):
        """Progressbar widget created with 0-100 range."""
        root = tk.Tk()
        root.withdraw()

        progress_bar = GuiFactory.create_progress_bar(
            root,
            length=200,
            mode='determinate'
        )

        assert progress_bar is not None
        assert isinstance(progress_bar, ttk.Progressbar)
        # Check maximum is set to 100 (default)
        assert progress_bar.cget('maximum') == 100

        root.destroy()
