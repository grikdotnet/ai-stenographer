# main.py
import sys
import logging
from pathlib import Path
from src.PathResolver import PathResolver
from src.StartupArgs import StartupArgs
from src.StartupController import (
    StartupController,
    TauriBinaryNotFoundError,
    MissingModelsError,
    DownloadCancelledError,
)
from src.asr.ModelManager import ModelManager
from src.asr.ModelLoader import ModelLoadError
from src.asr.ModelRegistry import ModelRegistry


if hasattr(sys.modules['__main__'], '__file__'):
    SCRIPT_PATH = Path(sys.modules['__main__'].__file__).resolve()
else:
    SCRIPT_PATH = Path(__file__).resolve()

path_resolver = PathResolver(SCRIPT_PATH)


def _main(argv: list[str]) -> None:
    """Core entry-point logic, extracted for testability.

    Args:
        argv: Command-line arguments (typically sys.argv).
    """
    try:
        path_resolver.ensure_local_dir_structure()
        resolved_paths = path_resolver.paths
        model_registry = ModelRegistry(resolved_paths)
        controller = StartupController(
            StartupArgs.from_argv(argv),
            resolved_paths,
            ModelManager(model_registry),
        )
        controller.run()
    except (TauriBinaryNotFoundError, MissingModelsError, ModelLoadError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except DownloadCancelledError:
        sys.exit(0)


if __name__ == "__main__":
    try:
        _main(sys.argv)
    except SystemExit:
        raise
    except Exception as e:
        logging.error(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)
