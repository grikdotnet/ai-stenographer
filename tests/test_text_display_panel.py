"""
Tests for TextDisplayPanel component.
"""
import tkinter as tk
import pytest
from src.gui.TextDisplayPanel import TextDisplayPanel
from src.GuiWindow import GuiWindow


@pytest.fixture
def root():
    """Create tkinter root for testing."""
    root = tk.Tk()
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def config():
    """Minimal config for testing."""
    return {}


def test_text_display_panel_creates_text_widget(root, config):
    """Test that TextDisplayPanel creates a scrolled text widget."""
    frame = tk.Frame(root)
    panel = TextDisplayPanel(frame, root, config)

    assert panel.text_widget is not None
    assert panel.text_widget.winfo_class() == 'Text'


def test_text_display_panel_creates_gui_window(root, config):
    """Test that TextDisplayPanel creates a GuiWindow instance."""
    frame = tk.Frame(root)
    panel = TextDisplayPanel(frame, root, config)

    assert panel.gui_window is not None
    assert isinstance(panel.gui_window, GuiWindow)


def test_text_display_panel_gui_window_wraps_text_widget(root, config):
    """Test that GuiWindow wraps the created text widget."""
    frame = tk.Frame(root)
    panel = TextDisplayPanel(frame, root, config)

    # GuiWindow should wrap the text_widget
    assert panel.gui_window.text_widget == panel.text_widget


def test_text_display_panel_text_widget_is_editable(root, config):
    """Test that text widget is in NORMAL state (editable programmatically)."""
    frame = tk.Frame(root)
    panel = TextDisplayPanel(frame, root, config)

    assert str(panel.text_widget.cget('state')) == 'normal'


def test_text_display_panel_text_widget_has_correct_styling(root, config):
    """Test that text widget has correct font and colors."""
    frame = tk.Frame(root)
    panel = TextDisplayPanel(frame, root, config)

    assert panel.text_widget.cget('bg') == 'white'
    assert panel.text_widget.cget('fg') == 'black'
    font = panel.text_widget.cget('font')
    assert "Arial" in str(font)
    assert "12" in str(font)
