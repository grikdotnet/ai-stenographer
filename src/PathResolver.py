# PathResolver.py
"""
Path resolution for different distribution environments.

Encapsulates all logic for detecting distribution mode and resolving
application paths for msix, portable, and development environments.
"""
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DistributionMode = Literal["msix", "portable", "development"]


@dataclass(frozen=True)
class ResolvedPaths:
    """Immutable container for all resolved application paths."""
    app_dir: Path
    internal_dir: Path
    root_dir: Path
    models_dir: Path
    config_dir: Path
    assets_dir: Path
    logs_dir: Path
    environment: DistributionMode


class PathResolver:
    """
    Resolves application paths for different distribution environments.

    Environments:
    - msix: MSIX package (sandboxed, uses AppData for writable storage)
    - portable: _internal distribution structure (ZIP)
    - development: Running from source
    """

    def __init__(self, script_path: Path):
        self._script_path = script_path.resolve()
        self._mode = self._detect_distribution_mode()
        self._paths = self._resolve_paths()

    @property
    def paths(self) -> ResolvedPaths:
        """Returns resolved paths for current environment."""
        return self._paths

    @property
    def mode(self) -> DistributionMode:
        """Returns current distribution mode."""
        return self._mode

    def _detect_distribution_mode(self) -> DistributionMode:
        """Detects distribution mode from script path and environment."""
        # Check for MSIX package (sandboxed environment)
        if os.environ.get('MSIX_PACKAGE_IDENTITY') is not None:
            return "msix"
        if 'WindowsApps' in str(self._script_path):
            return "msix"

        # Check for portable distribution (_internal/app structure)
        parts = self._script_path.parts
        if "_internal" in parts and "app" in parts:
            return "portable"

        return "development"

    def _get_msix_local_cache_path(self) -> Path:
        """
        Gets writable LocalCache path for MSIX packaged apps.

        MSIX packages have virtualized file system - %LOCALAPPDATA% doesn't work
        directly. Uses Windows.Storage.ApplicationData API to get the real path.
        """
        try:
            from winrt.windows.storage import ApplicationData
            return Path(ApplicationData.current.local_cache_folder.path)
        except ImportError:
            # Fallback: construct path from package identity
            local_app_data = Path(os.environ['LOCALAPPDATA'])
            packages_dir = local_app_data / "Packages"

            if packages_dir.exists():
                for folder in packages_dir.iterdir():
                    if folder.is_dir() and folder.name.startswith("AI.Stenographer_"):
                        return folder / "LocalCache" / "Local" / "AI-Stenographer"

            # Ultimate fallback
            return local_app_data / "AI-Stenographer"

    def _resolve_paths(self) -> ResolvedPaths:
        """Resolves all paths based on distribution mode."""
        # Base directories (same for msix/portable, different for dev)
        if self._mode in ("msix", "portable"):
            app_dir = self._script_path.parent
            internal_dir = app_dir.parent
            root_dir = internal_dir.parent
        else:
            app_dir = internal_dir = root_dir = self._script_path.parent

        # Resolve environment-specific paths
        if self._mode == "msix":
            app_data = self._get_msix_local_cache_path()
            app_data.mkdir(parents=True, exist_ok=True)
            models_dir = app_data / "models"
            config_dir = app_data / "config"
            assets_dir = app_dir
            logs_dir = app_data / "logs"
        elif self._mode == "portable":
            models_dir = internal_dir / "models"
            config_dir = app_dir / "config"
            assets_dir = app_dir / "assets"
            logs_dir = root_dir / "logs"
        else:  # development
            models_dir = root_dir / "models"
            config_dir = root_dir / "config"
            assets_dir = root_dir
            logs_dir = root_dir / "logs"

        return ResolvedPaths(
            app_dir=app_dir,
            internal_dir=internal_dir,
            root_dir=root_dir,
            models_dir=models_dir,
            config_dir=config_dir,
            assets_dir=assets_dir,
            logs_dir=logs_dir,
            environment=self._mode,
        )

    def get_asset_path(self, asset_name: str) -> Path:
        return self._paths.root_dir / asset_name

    def get_config_path(self, config_name: str) -> Path:
        return self._paths.config_dir / config_name

    def _copy_bundled_silero_if_needed(self) -> None:
        """
        Copies bundled Silero VAD to AppData if not present (MSIX mode only).

        In MSIX builds, Silero VAD is bundled in the read-only application directory
        at _internal/models/silero_vad/. On first run, this copies it to the writable
        AppData location so ModelManager can find it without downloading.
        """
        if self._mode != "msix":
            return

        target_silero = self._paths.models_dir / "silero_vad" / "silero_vad.onnx"
        if target_silero.exists():
            return  # Already copied or downloaded

        # Locate bundled Silero in app directory
        # _internal/app/../models/silero_vad = _internal/models/silero_vad
        bundled_silero = self._paths.app_dir.parent / "models" / "silero_vad"

        if not bundled_silero.exists():
            logging.warning("Bundled Silero VAD not found in app directory")
            return  # ModelManager will handle download fallback

        # Copy bundled Silero to AppData
        target_dir = self._paths.models_dir / "silero_vad"
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            for file in bundled_silero.iterdir():
                if file.is_file():
                    shutil.copy2(file, target_dir / file.name)
            logging.info(f"Copied bundled Silero VAD to {target_dir}")
        except Exception as e:
            logging.error(f"Failed to copy bundled Silero: {e}")

    def ensure_local_dir_structure(self) -> None:
        """
        Ensures directories models and config exist.
        For MSIX mode, copies bundled configs to AppData on first run.
        """
        self._paths.models_dir.mkdir(parents=True, exist_ok=True)
        self._paths.config_dir.mkdir(parents=True, exist_ok=True)
        self._paths.logs_dir.mkdir(parents=True, exist_ok=True)

        # MSIX-specific: Copy bundled configs
        if self._mode == "msix":
            bundled_config_dir = self._paths.app_dir / "config"
            if bundled_config_dir.exists():
                for bundled_config in bundled_config_dir.iterdir():
                    if bundled_config.is_file():
                        target = self._paths.config_dir / bundled_config.name
                        if not target.exists():
                            shutil.copy2(bundled_config, target)
                            logging.info(f"Copied bundled config: {bundled_config.name}")

            # Copy bundled Silero VAD to AppData (MSIX only)
            self._copy_bundled_silero_if_needed()
