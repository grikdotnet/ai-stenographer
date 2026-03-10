"""Entry point for downloading missing ASR models via GUI dialog.

Spawned by main.py when models are absent. Shows a Tk progress window and
downloads any missing models. Exits 0 on success, 1 on cancel or failure.

Not intended for direct user invocation in normal use.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.client.ClientPathResolver import ClientPathResolver
from src.client.tk.asr.ModelManager import ModelManager
from src.client.tk.gui.ModelDownloadDialog import show_download_dialog

_PATHS = ClientPathResolver(Path(__file__).resolve()).paths


def _main(models_dir: Path) -> None:
    """Show download dialog for any missing models and exit with appropriate code.

    Args:
        models_dir: Directory where models are stored.
    """
    missing = ModelManager.get_missing_models(models_dir)
    if not missing:
        sys.exit(0)
    success = show_download_dialog(None, missing, models_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    _main(models_dir=_PATHS.models_dir)
