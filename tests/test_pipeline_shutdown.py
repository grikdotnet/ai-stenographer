"""Tests for STTPipeline graceful shutdown."""
import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import STTPipeline


class TestPipelineShutdown:
    """Test graceful shutdown of STTPipeline."""

    @patch('src.pipeline.create_stt_window')
    @patch('src.pipeline.onnx_asr.load_model')
    def test_stop_is_idempotent(self, mock_load_model, mock_create_window):
        """Test that stop() can be called multiple times safely."""
        # Mock window creation
        mock_root = MagicMock()
        mock_text_widget = MagicMock()
        mock_create_window.return_value = (mock_root, mock_text_widget)

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

    @patch('src.pipeline.create_stt_window')
    @patch('src.pipeline.onnx_asr.load_model')
    def test_stop_called_before_start(self, mock_load_model, mock_create_window):
        """Test that stop() can be called even before start()."""
        # Mock window creation
        mock_root = MagicMock()
        mock_text_widget = MagicMock()
        mock_create_window.return_value = (mock_root, mock_text_widget)

        # Mock model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Create pipeline
        pipeline = STTPipeline(verbose=False)

        # Stop without starting - should not raise exception
        pipeline.stop()

        # Should be stopped
        assert pipeline._is_stopped is True

    @patch('src.pipeline.create_stt_window')
    @patch('src.pipeline.onnx_asr.load_model')
    def test_window_close_handler_registered(self, mock_load_model, mock_create_window):
        """Test that window close handler is registered."""
        # Mock window creation
        mock_root = MagicMock()
        mock_text_widget = MagicMock()
        mock_create_window.return_value = (mock_root, mock_text_widget)

        # Mock model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Create pipeline
        pipeline = STTPipeline(verbose=False)

        # Mock run_gui_loop to avoid blocking
        with patch('src.pipeline.run_gui_loop'):
            pipeline.run()

        # Verify protocol was registered
        mock_root.protocol.assert_called_once()
        call_args = mock_root.protocol.call_args
        assert call_args[0][0] == "WM_DELETE_WINDOW"
        assert callable(call_args[0][1])
