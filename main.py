# main.py
import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from src.PathResolver import PathResolver
from src.ModelManager import ModelManager
from src.gui.LoadingWindow import LoadingWindow


def setup_logging(logs_dir: Path, verbose: bool = False, is_frozen: bool = False):
    """
    Configure logging for both terminal and frozen GUI modes.

    Creates log directory, sets up file rotation, and optionally adds console
    output when running from terminal.

    Args:
        logs_dir: Directory to store log files
        verbose: If True, set DEBUG level; otherwise INFO
        is_frozen: If True, skip console handler (frozen app has no console)
    """
    # Reconfigure sys.stdout to use UTF-8 encoding
    if not is_frozen and hasattr(sys.stdout, 'buffer'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

    logs_dir.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.WARNING

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    root_logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

    # Add rotating file handler (10MB max, keep 5 files)
    log_file = logs_dir / "stenographer.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Add console handler only if not frozen (terminal mode)
    if not is_frozen:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized: level={logging.getLevelName(level)}, frozen={is_frozen}")


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

        window_duration = 2.0  # Default: 2 seconds
        input_file = None      # Default: use microphone

        for arg in sys.argv:
            if arg.startswith("--window="):
                window_duration = float(arg.split("=")[1])
            elif arg.startswith("--input-file="):
                input_file = arg.split("=", 1)[1]

        # Show loading window with stenographer image
        image_path = path_resolver.get_asset_path("stenographer.jpg")
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
            window_duration=window_duration,
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
