# main.py
import sys
import logging
from pathlib import Path
from src.PathResolver import PathResolver
from src.asr.ModelManager import ModelManager
from src.gui.LoadingWindow import LoadingWindow
from src.LoggingSetup import setup_logging


# ============================================================================
# RESOLVE PATHS AT MODULE LOAD TIME
# ============================================================================
# Determine script path (works for both .py and .pyc)
if hasattr(sys.modules['__main__'], '__file__'):
    SCRIPT_PATH = Path(sys.modules['__main__'].__file__).resolve()
else:
    SCRIPT_PATH = Path(__file__).resolve()

# Initialize path resolver
path_resolver = PathResolver(SCRIPT_PATH)
PATHS = path_resolver.paths

# Export convenient globals
APP_DIR = PATHS.app_dir
ROOT_DIR = PATHS.root_dir
MODELS_DIR = PATHS.models_dir
CONFIG_DIR = PATHS.config_dir
LOGS_DIR = PATHS.logs_dir

# Ensure writable directories exist
path_resolver.ensure_local_dir_structure()

if __name__ == "__main__":
    loading_window = None
    try:
        verbose = "-v" in sys.argv

        # Detect if running from frozen executable
        is_frozen = getattr(sys, 'frozen', False)

        # Setup logging BEFORE anything else
        setup_logging(LOGS_DIR, verbose=verbose, is_frozen=is_frozen)

        input_file = None      # Default: use microphone

        for arg in sys.argv:
            if arg.startswith("--input-file="):
                input_file = arg.split("=", 1)[1]

        # Show loading window with stenographer image
        image_path = path_resolver.get_asset_path("stenographer.gif")
        loading_window = LoadingWindow(image_path, "Initializing...")

        # Check for missing models BEFORE importing pipeline
        loading_window.update_message("Checking AI ...")
        missing_models = ModelManager.get_missing_models(MODELS_DIR)

        if missing_models:
            # Transform loading window to download dialog (keeps window alive, no lag)
            success = loading_window.transform_to_download_dialog(missing_models, MODELS_DIR)

            if not success:
                logging.info("Model download cancelled. Exiting.")
                sys.exit(1)

            # Transform back to loading screen after successful download
            loading_window.transform_back_to_loading("Models downloaded successfully")

        # Import pipeline only after models are confirmed present
        loading_window.update_message("Loading Parakeet ...")
        from src.pipeline import STTPipeline

        # Create pipeline (this loads the model)
        loading_window.update_message("The Stenographer is getting ready ...")
        pipeline = STTPipeline(
            model_path=str(MODELS_DIR / "parakeet"),
            models_dir=MODELS_DIR,
            config_path=str(path_resolver.get_config_path("stt_config.json")),
            verbose=verbose,
            input_file=input_file
        )

        # Close loading window before starting pipeline
        loading_window.update_message("Ready!")
        loading_window.close()
        loading_window = None

        # Run pipeline (opens STT window)
        pipeline.run()

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        if loading_window:
            loading_window.close()
        sys.exit(0)
    except Exception as e:
        logging.error(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        if loading_window:
            loading_window.close()
        sys.exit(1)
