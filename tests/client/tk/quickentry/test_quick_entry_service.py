"""Tests for QuickEntryService - facade for Quick Entry subsystem."""
import pytest
from unittest.mock import MagicMock, patch
import tkinter as tk

pytestmark = pytest.mark.gui


@pytest.fixture
def root():
    """Create a tkinter root window for testing."""
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def mock_publisher():
    """Create mock RecognitionResultPublisher."""
    publisher = MagicMock()
    return publisher


class TestQuickEntryService:
    """Test QuickEntryService facade."""

    def test_creates_all_components(self, root, mock_publisher):
        """Service should create all Quick Entry components."""
        from src.client.tk.quickentry.QuickEntryService import QuickEntryService

        service = QuickEntryService(
            publisher=mock_publisher,
            root=root,
            hotkey='F2'
        )

        assert service._focus_tracker is not None
        assert service._keyboard is not None
        assert service._subscriber is not None
        assert service._controller is not None
        assert service._hotkey_listener is not None

    def test_subscribes_to_publisher(self, root, mock_publisher):
        """Service should subscribe QuickEntrySubscriber to publisher."""
        from src.client.tk.quickentry.QuickEntryService import QuickEntryService

        service = QuickEntryService(
            publisher=mock_publisher,
            root=root,
            hotkey='F2'
        )

        mock_publisher.subscribe.assert_called_once_with(service._subscriber)

    def test_start_starts_hotkey_listener(self, root, mock_publisher):
        """start() should start the hotkey listener."""
        from src.client.tk.quickentry.QuickEntryService import QuickEntryService

        service = QuickEntryService(
            publisher=mock_publisher,
            root=root,
            hotkey='F2'
        )

        with patch.object(service._hotkey_listener, 'start') as mock_start:
            service.start()
            mock_start.assert_called_once()

    def test_stop_stops_hotkey_listener(self, root, mock_publisher):
        """stop() should stop the hotkey listener."""
        from src.client.tk.quickentry.QuickEntryService import QuickEntryService

        service = QuickEntryService(
            publisher=mock_publisher,
            root=root,
            hotkey='F2'
        )

        with patch.object(service._hotkey_listener, 'stop') as mock_stop:
            service.stop()
            mock_stop.assert_called_once()

    def test_controller_property(self, root, mock_publisher):
        """controller property should return the controller."""
        from src.client.tk.quickentry.QuickEntryService import QuickEntryService

        service = QuickEntryService(
            publisher=mock_publisher,
            root=root,
            hotkey='F2'
        )

        assert service.controller is service._controller

    def test_configurable_hotkeys(self, root, mock_publisher):
        """Should accept different hotkey configurations."""
        from src.client.tk.quickentry.QuickEntryService import QuickEntryService

        # Test F2
        service1 = QuickEntryService(mock_publisher, root, hotkey='F2')
        assert service1._hotkey_listener._target_key is not None

        # Test ctrl+space
        service2 = QuickEntryService(mock_publisher, root, hotkey='ctrl+space')
        assert service2._hotkey_listener._target_key is not None

        # Test alt+enter
        service3 = QuickEntryService(mock_publisher, root, hotkey='alt+enter')
        assert service3._hotkey_listener._target_key is not None

    def test_text_change_callback_wired_correctly(self, root, mock_publisher):
        """Subscriber's text change callback should reach controller."""
        from src.client.tk.quickentry.QuickEntryService import QuickEntryService

        service = QuickEntryService(
            publisher=mock_publisher,
            root=root,
            hotkey='F2'
        )

        # Verify callback chain by checking subscriber's callback invokes controller
        with patch.object(service._controller, 'on_text_change') as mock_change:
            # Directly invoke subscriber's callback
            service._subscriber._on_text_change("hello", "wor")
            mock_change.assert_called_once_with("hello", "wor")

    def test_hotkey_callback_wired_correctly(self, root, mock_publisher):
        """Hotkey listener's callback should be controller's on_hotkey."""
        from src.client.tk.quickentry.QuickEntryService import QuickEntryService

        service = QuickEntryService(
            publisher=mock_publisher,
            root=root,
            hotkey='F2'
        )

        # Verify hotkey listener's callback is wired to controller
        assert service._hotkey_listener._callback == service._controller.on_hotkey

    def test_enabled_flag_controls_service(self, root, mock_publisher):
        """enabled=False should not start hotkey listener."""
        from src.client.tk.quickentry.QuickEntryService import QuickEntryService

        service = QuickEntryService(
            publisher=mock_publisher,
            root=root,
            hotkey='F2',
            enabled=False
        )

        with patch.object(service._hotkey_listener, 'start') as mock_start:
            service.start()
            mock_start.assert_not_called()
