# main.py
import sys
import os
import logging
from pathlib import Path
from typing import Dict
from logging.handlers import RotatingFileHandler
from src.ModelManager import ModelManager
from src.ModelDownloadDialog import show_download_dialog
from src.gui.LoadingWindow import LoadingWindow


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


def _get_msix_local_cache_path() -> Path:
    """
    Gets the LocalCache path for MSIX packaged apps.

    MSIX packages have virtualized file system - %LOCALAPPDATA% doesn't work
    directly. Uses Windows.Storage.ApplicationData API to get the real path.

    Returns:
        Path to LocalCache folder (writable storage for MSIX apps)
    """
    try:
        # Try Windows.Storage API (requires winrt package)
        from winrt.windows.storage import ApplicationData
        return Path(ApplicationData.current.local_cache_folder.path)
    except ImportError:
        # Fallback: construct path from package identity
        # Package folder is at: %LOCALAPPDATA%\Packages\<PackageFamilyName>\LocalCache
        local_app_data = Path(os.environ['LOCALAPPDATA'])
        packages_dir = local_app_data / "Packages"

        # Find our package folder (starts with "AI.Stenographer_")
        if packages_dir.exists():
            for folder in packages_dir.iterdir():
                if folder.is_dir() and folder.name.startswith("AI.Stenographer_"):
                    return folder / "LocalCache" / "Local" / "AI-Stenographer"

        # Ultimate fallback - shouldn't happen but let's not crash
        return local_app_data / "AI-Stenographer"


def resolve_paths(script_path: Path) -> Dict[str, Path]:
    """
    Resolves application paths for development, portable, and Store environments.

    Environments:
    - Development: Uses project directory (./models, ./config)
    - Portable: Uses _internal relative paths (distribution ZIP)
    - Store: Uses AppData for writable data (MSIX sandboxed)

    Args:
        script_path: Path to main script (.py or .pyc)

    Returns:
        Dictionary with resolved paths including ENVIRONMENT key
    """
    script_path = script_path.resolve()

    # Detect Microsoft Store environment (MSIX package)
    # Method 1: Environment variable set by launcher
    is_store = os.environ.get('MSIX_PACKAGE_IDENTITY') is not None
    # Method 2: Fallback - detect WindowsApps installation path
    if not is_store:
        is_store = 'WindowsApps' in str(script_path)
    is_frozen = getattr(sys, 'frozen', False)

    if is_store:
        # Microsoft Store sandboxed environment
        app_data = _get_msix_local_cache_path()
        app_data.mkdir(parents=True, exist_ok=True)

        app_dir = script_path.parent          # _internal/app/
        internal_dir = app_dir.parent         # _internal/
        root_dir = internal_dir.parent        # Package root

        return {
            "APP_DIR": app_dir,
            "INTERNAL_DIR": internal_dir,
            "ROOT_DIR": root_dir,
            "MODELS_DIR": app_data / "models",
            "CONFIG_DIR": app_data / "config",
            "ASSETS_DIR": app_dir,  # Assets bundled with app
            "LOGS_DIR": app_data / "logs",
            "ENVIRONMENT": "store"
        }
    elif is_distribution_mode(script_path):
        # Portable distribution mode: _internal/app/main.pyc
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
            "ENVIRONMENT": "portable"
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
            "ENVIRONMENT": "development"
        }


def get_asset_path(asset_name: str, paths: Dict[str, Path]) -> Path:
    """
    Gets path to user-visible asset file.

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
    Gets path to configuration file, copying from bundled defaults if needed.

    Args:
        config_name: Config filename (e.g., "stt_config.json")
        paths: Resolved paths from resolve_paths()

    Returns:
        Path to config file
    """
    config_path = paths["CONFIG_DIR"] / config_name

    # For Store mode, copy bundled config to AppData on first run
    if paths["ENVIRONMENT"] == "store" and not config_path.exists():
        # Bundled config is at APP_DIR/config/ (i.e., _internal/app/config/)
        bundled_config = paths["APP_DIR"] / "config" / config_name
        logging.info(f"APP_DIR: {paths['APP_DIR']}")
        if bundled_config.exists():
            import shutil
            config_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled_config, config_path)
        else:
            logging.warning(f"Bundled config not found at {bundled_config}")

    return config_path


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