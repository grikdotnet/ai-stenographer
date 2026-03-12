"""GUI-specific tests for ApplicationState observer behavior with tkinter root."""
import threading
import unittest
from unittest.mock import Mock

import pytest

try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


pytestmark = pytest.mark.gui


@unittest.skipUnless(TKINTER_AVAILABLE, "tkinter not available")
class TestApplicationStateGui(unittest.TestCase):
    """Test ApplicationState GUI observer behavior with tkinter scheduling."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from src.ApplicationState import ApplicationState
            self.ApplicationState = ApplicationState
        except ImportError:
            self.skipTest("ApplicationState not implemented yet")

    def test_gui_observer_called_directly_on_main_thread(self):
        """Test that GUI observers are called directly when on main thread."""
        try:
            root = tk.Tk()
            root.withdraw()
        except Exception as e:
            self.skipTest(f"tkinter initialization failed: {e}")

        try:
            app_state = self.ApplicationState(config={}, root=root)
            observer = Mock()
            app_state.register_gui_observer(observer)
            app_state.set_state('running')
            observer.assert_called_once_with('starting', 'running')
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def test_gui_observer_scheduled_via_root_after_on_background_thread(self):
        """Test that GUI observers use root.after() when called from background thread."""
        try:
            root = tk.Tk()
            root.withdraw()
        except Exception as e:
            self.skipTest(f"tkinter initialization failed: {e}")

        try:
            app_state = self.ApplicationState(config={}, root=root)
            observer = Mock()
            app_state.register_gui_observer(observer)

            def background_update():
                app_state.set_state('running')

            thread = threading.Thread(target=background_update)
            thread.start()
            thread.join()

            root.update()
            observer.assert_called_once_with('starting', 'running')
        finally:
            try:
                root.destroy()
            except Exception:
                pass


if __name__ == '__main__':
    unittest.main()
