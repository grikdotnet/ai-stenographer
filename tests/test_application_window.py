"""
Tests for ApplicationWindow component.

NOTE: These tests may occasionally fail on Windows due to tkinter/TCL interpreter
state corruption when creating multiple Tk() instances in the same process.

To run these tests reliably, use pytest-xdist to isolate each test in its own process:
    python -m pytest tests/test_application_window.py -n auto

Or run tests individually:
    python -m pytest tests/test_application_window.py::test_name -v
"""
import pytest
import os
import sys
from src.ApplicationState import ApplicationState
from src.gui.HeaderPanel import HeaderPanel
from src.gui.ControlPanel import ControlPanel

# Fix TCL/TK library paths if not set (Windows Python 3.13 issue)
# NOTE: This is already done in conftest.py, but keeping it here for clarity
if sys.platform == 'win32' and not os.environ.get('TCL_LIBRARY'):
    # Find base Python installation (not venv)
    if hasattr(sys, 'base_prefix'):
        python_root = sys.base_prefix  # Use base installation for venvs
    else:
        python_root = os.path.dirname(sys.executable)
    os.environ['TCL_LIBRARY'] = os.path.join(python_root, 'tcl', 'tcl8.6')
    os.environ['TK_LIBRARY'] = os.path.join(python_root, 'tcl', 'tk8.6')

# Try to import tkinter and test if Tk() can be created
# We use subprocess to avoid polluting the tcl interpreter state
import subprocess

def _check_tkinter_available():
    """Test if tkinter can actually create Tk instances."""
    # First check if imports work
    try:
        import tkinter as tk
        from src.gui.ApplicationWindow import ApplicationWindow
        from src.gui.TextFormatter import TextFormatter
    except ImportError:
        return False
    except Exception:
        return False

    # Then test if Tk() creation works reliably (create multiple to detect instability)
    test_code = """
import tkinter as tk
import gc
try:
    # Test creating multiple Tk instances like the tests will do
    for i in range(3):
        root = tk.Tk()
        root.destroy()
        gc.collect()
    exit(0)
except Exception as e:
    import sys
    print(f"Tkinter test failed: {e}", file=sys.stderr)
    exit(1)
"""
    # Pass environment variables to subprocess (including TCL_LIBRARY/TK_LIBRARY fixes)
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        timeout=10,
        env=os.environ.copy()
    )
    return result.returncode == 0

TKINTER_AVAILABLE = _check_tkinter_available()

# Import after check (will be available if check passed)
if TKINTER_AVAILABLE:
    import tkinter as tk
    from src.gui.ApplicationWindow import ApplicationWindow
    from src.gui.TextFormatter import TextFormatter
    from src.gui.TextDisplayWidget import TextDisplayWidget


@pytest.fixture
def app_state():
    """Create ApplicationState for testing."""
    config = {'audio': {'sample_rate': 16000}}
    return ApplicationState(config=config)


@pytest.fixture
def config():
    """Minimal config for testing."""
    return {'audio': {'sample_rate': 16000}}


# Mark all tests in this module as GUI tests (may be flaky on Windows)
pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not TKINTER_AVAILABLE, reason="tkinter not available or properly configured")
]


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


def test_application_window_creates_formatter_and_display(app_state, config):
    """Test that ApplicationWindow creates TextFormatter and TextDisplayWidget instances."""
    app_window = ApplicationWindow(app_state, config)

    assert isinstance(app_window.formatter, TextFormatter)
    assert isinstance(app_window.display, TextDisplayWidget)
    assert isinstance(app_window.display.text_widget, tk.scrolledtext.ScrolledText)

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
