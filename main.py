# main.py
import sys
from pathlib import Path
from typing import Dict
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

# Ensure models directory exists
ensure_models_dir(PATHS)

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
            model_path=str(MODELS_DIR / "parakeet"),
            models_dir=MODELS_DIR,
            config_path=str(get_config_path("stt_config.json", PATHS)),
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