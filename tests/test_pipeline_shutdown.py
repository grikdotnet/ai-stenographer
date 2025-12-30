"""Tests for STTPipeline graceful shutdown."""
import pytest
from unittest.mock import MagicMock, patch, Mock
from src.pipeline import STTPipeline


class TestPipelineShutdown:
    """Test graceful shutdown of STTPipeline."""

    @patch('src.pipeline.ApplicationWindow')
    @patch('src.pipeline.onnx_asr.load_model')
    def test_stop_is_idempotent(self, mock_load_model, mock_app_window):
        """Test that stop() can be called multiple times safely."""
        # Mock ApplicationWindow
        mock_root = Mock()
        mock_gui_window = Mock()
        mock_app_window_instance = Mock()
        mock_app_window_instance.get_root.return_value = mock_root
        mock_app_window_instance.get_gui_window.return_value = mock_gui_window
        mock_app_window.return_value = mock_app_window_instance

        # Mock model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Create pipeline
        pipeline = STTPipeline(verbose=False)

        # Start pipeline (starts threads)
        pipeline.start()

        # Stop pipeline once
        pipeline.stop()

        # Verify stopped flag is set
        assert pipeline._is_stopped is True

        # Stop pipeline again - should not raise exception
        pipeline.stop()

        # Should still be stopped
        assert pipeline._is_stopped is True

    @patch('src.pipeline.ApplicationWindow')
    @patch('src.pipeline.onnx_asr.load_model')
    def test_stop_called_before_start(self, mock_load_model, mock_app_window):
        """Test that stop() can be called even before start()."""
        # Mock ApplicationWindow
        mock_root = Mock()
        mock_gui_window = Mock()
        mock_app_window_instance = Mock()
        mock_app_window_instance.get_root.return_value = mock_root
        mock_app_window_instance.get_gui_window.return_value = mock_gui_window
        mock_app_window.return_value = mock_app_window_instance

        # Mock model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Create pipeline
        pipeline = STTPipeline(verbose=False)

        # Stop without starting - should not raise exception
        pipeline.stop()

        # Should be stopped
        assert pipeline._is_stopped is True

    @patch('src.pipeline.ApplicationWindow')
    @patch('src.pipeline.onnx_asr.load_model')
    def test_window_close_handler_registered(self, mock_load_model, mock_app_window):
        """Test that window close handler is registered."""
        # Mock ApplicationWindow
        mock_root = Mock()
        mock_gui_window = Mock()
        mock_app_window_instance = Mock()
        mock_app_window_instance.get_root.return_value = mock_root
        mock_app_window_instance.get_gui_window.return_value = mock_gui_window
        mock_app_window.return_value = mock_app_window_instance

        # Mock model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Create pipeline
        pipeline = STTPipeline(verbose=False)

        # Mock mainloop to stop pipeline when called (simulates window closing)
        def mock_mainloop():
            pipeline.stop()

        mock_root.mainloop = mock_mainloop
        pipeline.run()

        # Verify protocol was registered
        mock_root.protocol.assert_called_once()
        call_args = mock_root.protocol.call_args
        assert call_args[0][0] == "WM_DELETE_WINDOW"
        assert callable(call_args[0][1])
