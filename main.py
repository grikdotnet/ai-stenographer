# main.py
import sys
import logging
from pathlib import Path
from typing import Dict
from logging.handlers import RotatingFileHandler
from src.ModelManager import ModelManager
from src.ModelDownloadDialog import show_download_dialog
from src.LoadingWindow import LoadingWindow


# ============================================================================
# PATH RESOLUTION FOR _INTERNAL DISTRIBUTION STRUCTURE
# ============================================================================
def is_distribution_mode(script_path: Path) -> bool:
    """
    Detects if running from distribution (_internal structure) or development.

    Distribution structure:
        STT-Stenographer/_internal/app/main.pyc

    Development structure:
        stt-project/main.py

    Args:
        script_path: Path to main script (.py or .pyc)

    Returns:
        True if running from _internal distribution structure
    """
    script_path = script_path.resolve()

    # Check if path contains "_internal/app"
    parts = script_path.parts
    return "_internal" in parts and "app" in parts


def resolve_paths(script_path: Path) -> Dict[str, Path]:
    """
    Resolves application paths for both distribution and development modes.

    Distribution structure:
        STT-Stenographer/              # ROOT_DIR
        ├── stenographer.jpg           # User-visible assets
        ├── icon.ico
        └── _internal/                 # INTERNAL_DIR
            ├── app/                   # APP_DIR
            │   ├── main.pyc
            │   ├── src/
            │   ├── config/            # CONFIG_DIR
            │   └── assets/            # ASSETS_DIR (not used currently)
            └── models/                # MODELS_DIR

    Development structure:
        stt-project/                   # APP_DIR = ROOT_DIR
        ├── main.py
        ├── src/
        ├── config/                    # CONFIG_DIR
        ├── models/                    # MODELS_DIR
        └── stenographer.jpg

    Args:
        script_path: Path to main script (.py or .pyc)

    Returns:
        Dictionary with resolved paths
    """
    script_path = script_path.resolve()

    if is_distribution_mode(script_path):
        # Distribution mode: _internal/app/main.pyc
        app_dir = script_path.parent              # _internal/app/
        internal_dir = app_dir.parent             # _internal/
        root_dir = internal_dir.parent            # STT-Stenographer/

        return {
            "APP_DIR": app_dir,
            "INTERNAL_DIR": internal_dir,
            "ROOT_DIR": root_dir,
            "MODELS_DIR": internal_dir / "models",
            "CONFIG_DIR": app_dir / "config",
            "ASSETS_DIR": app_dir / "assets",
            "LOGS_DIR": root_dir / "logs",
        }
    else:
        # Development mode: ./main.py
        project_dir = script_path.parent

        return {
            "APP_DIR": project_dir,
            "INTERNAL_DIR": project_dir,  # No _internal in dev
            "ROOT_DIR": project_dir,
            "MODELS_DIR": project_dir / "models",
            "CONFIG_DIR": project_dir / "config",
            "ASSETS_DIR": project_dir,  # Assets in project root during dev
            "LOGS_DIR": project_dir / "logs",
        }


def get_asset_path(asset_name: str, paths: Dict[str, Path]) -> Path:
    """
    Gets path to user-visible asset file.

    In distribution, user-visible assets (stenographer.jpg, icon.ico) are in
    the root directory, not in _internal.

    Args:
        asset_name: Asset filename (e.g., "stenographer.jpg")
        paths: Resolved paths from resolve_paths()

    Returns:
        Path to asset file
    """
    if is_distribution_mode(paths["APP_DIR"] / "main.pyc"):
        # Distribution: assets in root directory (user-visible)
        return paths["ROOT_DIR"] / asset_name
    else:
        # Development: assets in project root
        return paths["ROOT_DIR"] / asset_name


def get_config_path(config_name: str, paths: Dict[str, Path]) -> Path:
    """
    Gets path to configuration file.

    Config files are always in _internal/app/config/ (distribution)
    or ./config/ (development).

    Args:
        config_name: Config filename (e.g., "stt_config.json")
        paths: Resolved paths from resolve_paths()

    Returns:
        Path to config file
    """
    return paths["CONFIG_DIR"] / config_name


def ensure_models_dir(paths: Dict[str, Path]) -> Path:
    """
    Ensures models directory exists and returns its path.

    Args:
        paths: Resolved paths from resolve_paths()

    Returns:
        Path to models directory
    """
    models_dir = paths["MODELS_DIR"]
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


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
    # Reconfigure sys.stdout to use UTF-8 encoding BEFORE creating handlers
    # This prevents UnicodeEncodeError on Windows when logging non-ASCII characters
    if not is_frozen and hasattr(sys.stdout, 'buffer'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

    # Create logs directory
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Determine log level
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

    # Add rotating file handler (10MB max, keep 5 files)
    # Use UTF-8 encoding for log file to support Unicode
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

# Resolve all paths
PATHS = resolve_paths(SCRIPT_PATH)

# Export convenient globals
APP_DIR = PATHS["APP_DIR"]
ROOT_DIR = PATHS["ROOT_DIR"]
MODELS_DIR = PATHS["MODELS_DIR"]
CONFIG_DIR = PATHS["CONFIG_DIR"]
LOGS_DIR = PATHS["LOGS_DIR"]

# Ensure models directory exists
ensure_models_dir(PATHS)

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
        image_path = get_asset_path("stenographer.jpg", PATHS)
        loading_window = LoadingWindow(image_path, "Initializing...")

        # Check for missing models BEFORE importing pipeline
        loading_window.update_message("Checking AI ...")
        missing_models = ModelManager.get_missing_models(MODELS_DIR)

        if missing_models:
            # Close loading window before showing download dialog
            loading_window.close()
            loading_window = None

            # Show download dialog (creates its own window)
            success = show_download_dialog(None, missing_models, MODELS_DIR)

            if not success:
                logging.info("Model download cancelled. Exiting.")
                sys.exit(1)

            # Re-create loading window after download
            loading_window = LoadingWindow(image_path, "Models downloaded successfully")

        # Import pipeline only after models are confirmed present
        loading_window.update_message("Loading Parakeet ...")
        from src.pipeline import STTPipeline

        # Create pipeline (this loads the model)
        loading_window.update_message("The Stenographer is getting ready ...")
        pipeline = STTPipeline(
            model_path=str(MODELS_DIR / "parakeet"),
            models_dir=MODELS_DIR,
            config_path=str(get_config_path("stt_config.json", PATHS)),
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