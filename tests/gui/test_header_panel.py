"""
Tests for HeaderPanel component.
"""
import tkinter as tk
import pytest
from src.gui.HeaderPanel import HeaderPanel

pytestmark = pytest.mark.gui


@pytest.fixture
def root():
    """Create tkinter root for testing."""
    root = tk.Tk()
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


def test_header_panel_creates_title_label(root):
    """Test that HeaderPanel creates a title label with correct text."""
    frame = tk.Frame(root)
    panel = HeaderPanel(frame)

    assert panel.title_label is not None
    assert panel.title_label.cget('text') == "Real-time Speech Recognition"


def test_header_panel_title_label_has_correct_font(root):
    """Test that title label has correct font styling."""
    frame = tk.Frame(root)
    panel = HeaderPanel(frame)

    font = panel.title_label.cget('font')
    # Font can be returned as tuple or string depending on tkinter version
    assert "Arial" in str(font)
    assert "16" in str(font)
    assert "bold" in str(font)


def test_header_panel_title_label_is_packed(root):
    """Test that title label is packed into parent frame."""
    frame = tk.Frame(root)
    panel = HeaderPanel(frame)

    # Verify label is a child of HeaderPanel's internal frame
    assert panel.title_label.master == panel.frame
