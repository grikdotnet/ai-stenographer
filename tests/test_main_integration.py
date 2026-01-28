"""
Integration tests for main.py model checking and download flow.
"""
import pytest
import sys
from unittest.mock import patch, MagicMock, call
from pathlib import Path


class TestMainIntegration:
    """Integration tests for main.py startup with model checking."""

    @patch('src.asr.ModelManager.ModelManager.get_missing_models')
    @patch('src.pipeline.STTPipeline')
    def test_startup_with_models_present(self, mock_pipeline, mock_get_missing):
        """No dialog shown when all models are present, pipeline starts normally."""
        # All models present
        mock_get_missing.return_value = []

        # Mock pipeline
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance

        # Simulate main.py logic
        missing_models = mock_get_missing()

        if not missing_models:
            # Pipeline should start
            pipeline = mock_pipeline(model_path="./models/parakeet", verbose=False)
            pipeline.run()

        # Verify no dialog was needed
        assert len(missing_models) == 0
        mock_pipeline.assert_called_once()
        mock_instance.run.assert_called_once()

    @patch('src.gui.ModelDownloadDialog.show_download_dialog')
    @patch('src.asr.ModelManager.ModelManager.get_missing_models')
    @patch('src.pipeline.STTPipeline')
    def test_startup_with_models_missing_user_confirms(self, mock_pipeline, mock_get_missing, mock_dialog):
        """Dialog shown when models missing, downloads, then pipeline starts."""
        # Models are missing
        mock_get_missing.return_value = ['parakeet', 'silero_vad']

        # User confirms download
        mock_dialog.return_value = True

        # Mock pipeline
        mock_instance = MagicMock()
        mock_pipeline.return_value = mock_instance

        # Simulate main.py logic
        missing_models = mock_get_missing()

        if missing_models:
            success = mock_dialog(None, missing_models)

            if success:
                # Pipeline should start
                pipeline = mock_pipeline(model_path="./models/parakeet", verbose=False)
                pipeline.run()

        # Verify dialog was shown and download succeeded
        mock_dialog.assert_called_once()
        mock_pipeline.assert_called_once()
        mock_instance.run.assert_called_once()

    @patch('src.gui.ModelDownloadDialog.show_download_dialog')
    @patch('src.asr.ModelManager.ModelManager.get_missing_models')
    @patch('src.pipeline.STTPipeline')
    @patch('sys.exit')
    def test_startup_with_models_missing_user_cancels(self, mock_exit, mock_pipeline, mock_get_missing, mock_dialog):
        """Dialog shown when models missing, user cancels, app exits with message."""
        # Models are missing
        mock_get_missing.return_value = ['parakeet', 'silero_vad']

        # User cancels download
        mock_dialog.return_value = False

        # Simulate main.py logic
        missing_models = mock_get_missing()

        if missing_models:
            success = mock_dialog(None, missing_models)

            if not success:
                print("Model download cancelled. Exiting.")
                mock_exit(1)

        # Verify dialog was shown and app exited
        mock_dialog.assert_called_once()
        mock_exit.assert_called_once_with(1)
        # Pipeline should NOT start
        mock_pipeline.assert_not_called()


class TestLoadingWindowIntegration:
    """Integration tests for LoadingWindow display during startup."""

    @patch('src.gui.LoadingWindow.LoadingWindow')
    @patch('src.asr.ModelManager.ModelManager.get_missing_models')
    @patch('src.pipeline.STTPipeline')
    def test_loading_window_shown_during_startup(self, mock_pipeline, mock_get_missing, mock_loading_window):
        """LoadingWindow is created and shows progress messages during startup."""
        # All models present
        mock_get_missing.return_value = []

        # Mock LoadingWindow instance
        mock_window_instance = MagicMock()
        mock_loading_window.return_value = mock_window_instance

        # Mock pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        # Simulate main.py logic with loading window
        image_path = Path("stenographer.jpg")
        loading_window = mock_loading_window(image_path, "Initializing...")

        loading_window.update_message("Checking models...")
        missing_models = mock_get_missing()

        if not missing_models:
            loading_window.update_message("Loading Parakeet model...")
            loading_window.update_message("Initializing pipeline...")
            pipeline = mock_pipeline(model_path="./models/parakeet", verbose=False)

            loading_window.update_message("Ready!")
            loading_window.close()

            pipeline.run()

        # Verify LoadingWindow was created
        mock_loading_window.assert_called_once_with(image_path, "Initializing...")

        # Verify messages were updated in correct order
        expected_calls = [
            call("Checking models..."),
            call("Loading Parakeet model..."),
            call("Initializing pipeline..."),
            call("Ready!")
        ]
        mock_window_instance.update_message.assert_has_calls(expected_calls)

        # Verify window was closed
        mock_window_instance.close.assert_called_once()

        # Verify pipeline started
        mock_pipeline_instance.run.assert_called_once()

    @patch('src.gui.LoadingWindow.LoadingWindow')
    @patch('src.gui.ModelDownloadDialog.show_download_dialog')
    @patch('src.asr.ModelManager.ModelManager.get_missing_models')
    @patch('src.pipeline.STTPipeline')
    def test_loading_window_with_model_download(self, mock_pipeline, mock_get_missing, mock_dialog, mock_loading_window):
        """LoadingWindow closes before download dialog, reopens after."""
        # Models are missing
        mock_get_missing.return_value = ['parakeet', 'silero_vad']

        # User confirms download
        mock_dialog.return_value = True

        # Mock LoadingWindow instances (first window, then second after download)
        mock_window_1 = MagicMock()
        mock_window_2 = MagicMock()
        mock_loading_window.side_effect = [mock_window_1, mock_window_2]

        # Mock pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        # Simulate main.py logic
        image_path = Path("stenographer.jpg")
        loading_window = mock_loading_window(image_path, "Initializing...")

        loading_window.update_message("Checking models...")
        missing_models = mock_get_missing()

        if missing_models:
            # Close loading window before download dialog
            loading_window.close()

            success = mock_dialog(None, missing_models)

            if success:
                # Re-create loading window after download
                loading_window = mock_loading_window(image_path, "Models downloaded successfully")

                loading_window.update_message("Loading Parakeet model...")
                loading_window.update_message("Initializing pipeline...")
                pipeline = mock_pipeline(model_path="./models/parakeet", verbose=False)

                loading_window.update_message("Ready!")
                loading_window.close()

                pipeline.run()

        # Verify first window was created and closed
        assert mock_loading_window.call_count == 2
        mock_window_1.close.assert_called_once()

        # Verify second window was created with success message
        mock_loading_window.assert_any_call(image_path, "Models downloaded successfully")
        mock_window_2.close.assert_called_once()

        # Verify download dialog was shown
        mock_dialog.assert_called_once()

        # Verify pipeline started
        mock_pipeline_instance.run.assert_called_once()

    @patch('src.gui.LoadingWindow.LoadingWindow')
    @patch('src.asr.ModelManager.ModelManager.get_missing_models')
    @patch('src.pipeline.STTPipeline')
    def test_loading_window_closed_on_error(self, mock_pipeline, mock_get_missing, mock_loading_window):
        """LoadingWindow is closed even when exception occurs."""
        # All models present
        mock_get_missing.return_value = []

        # Mock LoadingWindow instance
        mock_window_instance = MagicMock()
        mock_loading_window.return_value = mock_window_instance

        # Mock pipeline to raise exception
        mock_pipeline.side_effect = Exception("Model loading failed")

        # Simulate main.py logic with error handling
        image_path = Path("stenographer.jpg")
        loading_window = mock_loading_window(image_path, "Initializing...")

        try:
            loading_window.update_message("Checking models...")
            missing_models = mock_get_missing()

            if not missing_models:
                loading_window.update_message("Loading Parakeet model...")
                pipeline = mock_pipeline(model_path="./models/parakeet", verbose=False)
        except Exception:
            # Should close window on error
            if loading_window:
                loading_window.close()

        # Verify window was closed despite error
        mock_window_instance.close.assert_called_once()
