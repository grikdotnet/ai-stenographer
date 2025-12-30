"""
Tests for MainView component.
"""
import tkinter as tk
import pytest
from src.gui.MainView import MainView
from src.ApplicationState import ApplicationState
from src.gui.HeaderPanel import HeaderPanel
from src.gui.TextDisplayPanel import TextDisplayPanel
from src.gui.ControlPanel import ControlPanel


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
def app_state():
    """Create ApplicationState for testing."""
    config = {'audio': {'sample_rate': 16000}}
    return ApplicationState(config=config)


@pytest.fixture
def config():
    """Minimal config for testing."""
    return {'audio': {'sample_rate': 16000}}


def test_main_view_creates_main_frame(root, app_state, config):
    """Test that MainView creates the main frame."""
    view = MainView(root, app_state, config)

    assert view.main_frame is not None
    assert isinstance(view.main_frame, tk.Frame)


def test_main_view_creates_header_panel(root, app_state, config):
    """Test that MainView creates a HeaderPanel."""
    view = MainView(root, app_state, config)

    assert view.header_panel is not None
    assert isinstance(view.header_panel, HeaderPanel)


def test_main_view_creates_control_panel(root, app_state, config):
    """Test that MainView creates a ControlPanel."""
    app_state.setTkRoot(root)  # Required for control panel
    view = MainView(root, app_state, config)

    assert view.control_panel is not None
    assert isinstance(view.control_panel, ControlPanel)


def test_main_view_creates_text_display_panel(root, app_state, config):
    """Test that MainView creates a TextDisplayPanel."""
    view = MainView(root, app_state, config)

    assert view.text_display_panel is not None
    assert isinstance(view.text_display_panel, TextDisplayPanel)


def test_main_view_creates_status_label(root, app_state, config):
    """Test that MainView creates the status label."""
    view = MainView(root, app_state, config)

    assert view.status_label is not None
    assert isinstance(view.status_label, tk.Label)
    assert "Gray text: Preliminary" in view.status_label.cget('text')


def test_main_view_layout_hierarchy(root, app_state, config):
    """Test that components are properly nested in the layout."""
    view = MainView(root, app_state, config)

    # Main frame should be child of root
    assert view.main_frame.master == root

    # Header panel's label should be child of main_frame
    assert view.header_panel.title_label.master == view.main_frame

    # Button frame should be child of main_frame
    assert view.button_frame.master == view.main_frame

    # Control panel should be child of button_frame
    assert view.control_panel.master == view.button_frame

    # Text display panel's text widget exists (ScrolledText creates its own internal frame)
    assert view.text_display_panel.text_widget is not None
