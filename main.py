# main.py
import sys
from pathlib import Path
from src.ModelManager import ModelManager
from src.ModelDownloadDialog import show_download_dialog
from src.LoadingWindow import LoadingWindow

if __name__ == "__main__":
    loading_window = None
    try:
        verbose = "-v" in sys.argv

        window_duration = 2.0  # Default: 2 seconds
        step_duration = 1.0    # Default: 1 second (50% overlap)

        for arg in sys.argv:
            if arg.startswith("--window="):
                window_duration = float(arg.split("=")[1])
            elif arg.startswith("--step="):
                step_duration = float(arg.split("=")[1])

        # Show loading window with stenographer image
        image_path = Path("stenographer.jpg")
        loading_window = LoadingWindow(image_path, "Initializing...")

        # Check for missing models BEFORE importing pipeline
        loading_window.update_message("Checking AI ...")
        missing_models = ModelManager.get_missing_models()

        if missing_models:
            # Close loading window before showing download dialog
            loading_window.close()
            loading_window = None

            # Show download dialog (creates its own window)
            success = show_download_dialog(None, missing_models)

            if not success:
                print("Model download cancelled. Exiting.")
                sys.exit(1)

            # Re-create loading window after download
            loading_window = LoadingWindow(image_path, "Models downloaded successfully")

        # Import pipeline only after models are confirmed present
        loading_window.update_message("Loading Parakeet ...")
        from src.pipeline import STTPipeline

        # Create pipeline (this loads the model)
        loading_window.update_message("The Stenographer is getting ready ...")
        pipeline = STTPipeline(
            model_path="./models/parakeet",
            verbose=verbose,
            window_duration=window_duration,
            step_duration=step_duration
        )

        # Close loading window before starting pipeline
        loading_window.update_message("Ready!")
        loading_window.close()
        loading_window = None

        # Run pipeline (opens STT window)
        pipeline.run()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if loading_window:
            loading_window.close()
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        if loading_window:
            loading_window.close()
        sys.exit(1)