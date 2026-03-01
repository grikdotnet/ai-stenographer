"""
Tests for ApplicationWindow pause_controller injection (Phase 5).

Verifies:
- Constructor with injected pause_controller passes it to ControlPanel
- Constructor without argument creates PauseController internally
"""
import pytest
import os
import sys
from unittest.mock import MagicMock, patch
from src.ApplicationState import ApplicationState

# Fix TCL/TK library paths if not set (Windows Python 3.13 issue)
if sys.platform == 'win32' and not os.environ.get('TCL_LIBRARY'):
    if hasattr(sys, 'base_prefix'):
        python_root = sys.base_prefix
    else:
        python_root = os.path.dirname(sys.executable)
    os.environ['TCL_LIBRARY'] = os.path.join(python_root, 'tcl', 'tcl8.6')
    os.environ['TK_LIBRARY'] = os.path.join(python_root, 'tcl', 'tk8.6')

import subprocess


def _check_tkinter_available() -> bool:
    """Test if tkinter can actually create Tk instances."""
    try:
        import tkinter as tk
        from src.client.tk.gui.ApplicationWindow import ApplicationWindow
    except ImportError:
        return False
    except Exception:
        return False

    test_code = """
import tkinter as tk
import gc
try:
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
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        timeout=10,
        env=os.environ.copy()
    )
    return result.returncode == 0


TKINTER_AVAILABLE = _check_tkinter_available()

if TKINTER_AVAILABLE:
    import tkinter as tk
    from src.client.tk.gui.ApplicationWindow import ApplicationWindow, SupportsPauseControl
    from src.client.tk.controllers.PauseController import PauseController

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not TKINTER_AVAILABLE, reason="tkinter not available or properly configured")
]


@pytest.fixture
def app_state():
    """ApplicationState for testing."""
    return ApplicationState(config={'audio': {'sample_rate': 16000}})


@pytest.fixture
def config():
    """Minimal config for testing."""
    return {'audio': {'sample_rate': 16000}}


def test_injected_pause_controller_is_used(app_state, config):
    """Constructor with injected pause_controller passes it to ControlPanel."""
    injected = MagicMock(spec=SupportsPauseControl)

    with patch('src.client.tk.gui.ApplicationWindow.ControlPanel') as mock_panel_cls:
        mock_panel_cls.return_value = MagicMock()
        mock_panel_cls.return_value.pack = MagicMock()

        app_window = ApplicationWindow(app_state, config, pause_controller=injected)

        _, kwargs = mock_panel_cls.call_args
        positional_args = mock_panel_cls.call_args.args
        passed_controller = positional_args[2] if len(positional_args) > 2 else kwargs.get('controller')
        assert passed_controller is injected

    app_window.root.destroy()


def test_default_pause_controller_created_when_none(app_state, config):
    """Constructor without pause_controller creates PauseController internally."""
    with patch('src.client.tk.gui.ApplicationWindow.ControlPanel') as mock_panel_cls:
        mock_panel_cls.return_value = MagicMock()
        mock_panel_cls.return_value.pack = MagicMock()

        with patch('src.client.tk.gui.ApplicationWindow.PauseController') as mock_pc_cls:
            mock_pc_instance = MagicMock()
            mock_pc_cls.return_value = mock_pc_instance

            app_window = ApplicationWindow(app_state, config)

            mock_pc_cls.assert_called_once_with(app_state)

            _, kwargs = mock_panel_cls.call_args
            positional_args = mock_panel_cls.call_args.args
            passed_controller = positional_args[2] if len(positional_args) > 2 else kwargs.get('controller')
            assert passed_controller is mock_pc_instance

    app_window.root.destroy()
