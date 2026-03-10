"""Self-contained path resolution for the Tk client subprocess.

Fully independent of src/ — no imports from server-side modules.
Detects distribution mode and resolves config, assets, and logs paths.
"""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ClientDistributionMode = Literal["msix", "portable", "development"]


@dataclass(frozen=True)
class ClientResolvedPaths:
    """Immutable container for all resolved client application paths."""

    root_dir: Path
    config_dir: Path
    assets_dir: Path
    logs_dir: Path
    models_dir: Path
    environment: ClientDistributionMode


class ClientPathResolver:
    """Resolves client application paths for different distribution environments.

    Environments:
    - msix: MSIX package (sandboxed, uses AppData for writable storage)
    - portable: _internal/app distribution structure (ZIP)
    - development: Running from source (script lives at <repo>/src/client/tk/)
    """

    def __init__(self, script_path: Path) -> None:
        """Initialize with the path of the running script.

        Args:
            script_path: Path to the entry-point script (client.py).
        """
        self._script_path = script_path.resolve()
        self._mode = self._detect_mode()
        self._paths = self._resolve_paths()

    @property
    def paths(self) -> ClientResolvedPaths:
        """Returns resolved paths for current environment."""
        return self._paths

    def get_config_path(self, name: str) -> Path:
        """Returns the full path for a named config file.

        Args:
            name: Config filename (e.g. "client_config.json").

        Returns:
            Absolute path to the config file.
        """
        return self._paths.config_dir / name

    def get_asset_path(self, name: str) -> Path:
        """Returns the full path for a named asset file.

        Args:
            name: Asset filename (e.g. "stenographer.gif").

        Returns:
            Absolute path to the asset.
        """
        return self._paths.assets_dir / name

    def ensure_local_dir_structure(self) -> None:
        """Creates config and logs directories; copies bundled configs in MSIX mode."""
        self._paths.config_dir.mkdir(parents=True, exist_ok=True)
        self._paths.logs_dir.mkdir(parents=True, exist_ok=True)

        if self._mode == "msix":
            self._copy_bundled_configs()

    def _detect_mode(self) -> ClientDistributionMode:
        """Detects distribution mode from environment and script path."""
        if os.environ.get("MSIX_PACKAGE_IDENTITY") is not None:
            return "msix"
        if "WindowsApps" in str(self._script_path):
            return "msix"

        parts = self._script_path.parts
        if "_internal" in parts and "app" in parts:
            return "portable"

        return "development"

    def _get_msix_appdata(self) -> Path:
        """Returns writable AppData path for MSIX packaged apps.

        Returns:
            Writable local cache directory for the application.
        """
        try:
            from winrt.windows.storage import ApplicationData  # type: ignore[import]
            return Path(ApplicationData.current.local_cache_folder.path)
        except ImportError:
            local_app_data = Path(os.environ["LOCALAPPDATA"])
            packages_dir = local_app_data / "Packages"
            if packages_dir.exists():
                for folder in packages_dir.iterdir():
                    if folder.is_dir() and folder.name.startswith("AI.Stenographer_"):
                        return folder / "LocalCache" / "Local" / "AI-Stenographer"
            return local_app_data / "AI-Stenographer"

    def _resolve_paths(self) -> ClientResolvedPaths:
        """Resolves all paths based on detected distribution mode.

        Algorithm:
            1. development: root_dir derived from ClientPathResolver.__file__ (src/client/tk/)
               → always repo root regardless of caller script depth; config/logs relative to root.
            2. portable: script is at <dist>/_internal/app/src/client/tk/client.py →
               app_dir = script^4; root_dir = app_dir^2; config under app_dir, logs under root.
            3. msix: same depth as portable, but writable paths come from AppData.
        """
        if self._mode == "development":
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            return ClientResolvedPaths(
                root_dir=root_dir,
                config_dir=root_dir / "config",
                assets_dir=root_dir,
                logs_dir=root_dir / "logs",
                models_dir=root_dir / "models",
                environment="development",
            )

        # portable and msix: script lives at _internal/app/src/client/tk/client.py
        app_dir = self._script_path.parent.parent.parent.parent  # → _internal/app/

        if self._mode == "portable":
            root_dir = app_dir.parent.parent  # → dist root
            return ClientResolvedPaths(
                root_dir=root_dir,
                config_dir=app_dir / "config",
                assets_dir=app_dir / "assets",
                logs_dir=root_dir / "logs",
                models_dir=root_dir / "models",
                environment="portable",
            )

        # msix
        appdata = self._get_msix_appdata()
        return ClientResolvedPaths(
            root_dir=app_dir,
            config_dir=appdata / "config",
            assets_dir=app_dir,
            logs_dir=appdata / "logs",
            models_dir=appdata / "models",
            environment="msix",
        )

    def _copy_bundled_configs(self) -> None:
        """Copies bundled config files to AppData if not already present (MSIX only)."""
        bundled_config_dir = self._script_path.parent.parent.parent.parent / "config"
        if not bundled_config_dir.exists():
            return
        for bundled in bundled_config_dir.iterdir():
            if bundled.is_file():
                target = self._paths.config_dir / bundled.name
                if not target.exists():
                    shutil.copy2(bundled, target)
                    logging.info("Copied bundled config: %s", bundled.name)
