"""
Tests for ApplicationWindow component.
"""
import tkinter as tk
import pytest
from src.gui.ApplicationWindow import ApplicationWindow
from src.ApplicationState import ApplicationState
from src.gui.MainView import MainView
from src.GuiWindow import GuiWindow


@pytest.fixture
def app_state():
    """Create ApplicationState for testing."""
    config = {'audio': {'sample_rate': 16000}}
    return ApplicationState(config=config)


@pytest.fixture
def config():
    """Minimal config for testing."""
    return {'audio': {'sample_rate': 16000}}


def test_application_window_creates_root(app_state, config):
    """Test that ApplicationWindow creates tk.Tk root."""
    app_window = ApplicationWindow(app_state, config)

    assert app_window.root is not None
    assert isinstance(app_window.root, tk.Tk)

    # Cleanup
    app_window.root.destroy()


def test_application_window_sets_root_in_app_state(app_state, config):
    """Test that ApplicationWindow sets root in ApplicationState."""
    app_window = ApplicationWindow(app_state, config)

    # ApplicationState should have root set
    assert app_state._root is not None
    assert app_state._root == app_window.root

    # Cleanup
    app_window.root.destroy()


def test_application_window_creates_main_view(app_state, config):
    """Test that ApplicationWindow creates MainView."""
    app_window = ApplicationWindow(app_state, config)

    assert app_window.main_view is not None
    assert isinstance(app_window.main_view, MainView)

    # Cleanup
    app_window.root.destroy()


def test_application_window_get_gui_window(app_state, config):
    """Test that get_gui_window() returns GuiWindow instance."""
    app_window = ApplicationWindow(app_state, config)

    gui_window = app_window.get_gui_window()

    assert gui_window is not None
    assert isinstance(gui_window, GuiWindow)
    # Should be the same instance from the main view's text display panel
    assert gui_window == app_window.main_view.text_display_panel.gui_window

    # Cleanup
    app_window.root.destroy()


def test_application_window_get_root(app_state, config):
    """Test that get_root() returns tk.Tk instance."""
    app_window = ApplicationWindow(app_state, config)

    root = app_window.get_root()

    assert root is not None
    assert isinstance(root, tk.Tk)
    assert root == app_window.root

    # Cleanup
    app_window.root.destroy()


def test_application_window_title(app_state, config):
    """Test that window has correct title."""
    app_window = ApplicationWindow(app_state, config)

    assert app_window.root.title() == "Speech-to-Text Display"

    # Cleanup
    app_window.root.destroy()
