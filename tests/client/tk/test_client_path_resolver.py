"""Tests for ClientPathResolver — self-contained path resolution for the Tk client."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.client.ClientPathResolver import ClientPathResolver, ClientResolvedPaths

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class TestClientPathResolver:
    """Tests for mode detection and path resolution in ClientPathResolver."""

    def test_development_resolves_repo_root(self, tmp_path: Path) -> None:
        """In development mode root_dir is always the repo root regardless of script depth."""
        script = tmp_path / "src" / "client" / "client.py"
        script.parent.mkdir(parents=True)
        script.touch()

        resolver = ClientPathResolver(script)

        assert resolver.paths.root_dir == _REPO_ROOT
        assert resolver.paths.config_dir == _REPO_ROOT / "config"
        assert resolver.paths.logs_dir == _REPO_ROOT / "logs"
        assert resolver.paths.models_dir == _REPO_ROOT / "models"
        assert resolver.paths.environment == "development"

    def test_development_script_depth_does_not_affect_root(self, tmp_path: Path) -> None:
        """root_dir is the same whether script is at src/client/ or src/client/tk/ depth."""
        shallow = tmp_path / "src" / "client" / "client.py"
        shallow.parent.mkdir(parents=True)
        shallow.touch()

        deep = tmp_path / "src" / "client" / "tk" / "download_models.py"
        deep.parent.mkdir(parents=True)
        deep.touch()

        assert ClientPathResolver(shallow).paths.root_dir == ClientPathResolver(deep).paths.root_dir

    def test_development_get_config_path(self, tmp_path: Path) -> None:
        """get_config_path returns config_dir / name in development mode."""
        script = tmp_path / "src" / "client" / "client.py"
        script.parent.mkdir(parents=True)
        script.touch()

        resolver = ClientPathResolver(script)

        assert resolver.get_config_path("client_config.json") == _REPO_ROOT / "config" / "client_config.json"

    def test_development_get_asset_path(self, tmp_path: Path) -> None:
        """get_asset_path returns assets_dir / name in development mode."""
        script = tmp_path / "src" / "client" / "client.py"
        script.parent.mkdir(parents=True)
        script.touch()

        resolver = ClientPathResolver(script)

        assert resolver.get_asset_path("stenographer.gif") == _REPO_ROOT / "stenographer.gif"

    def test_portable_resolves_app_dir(self, tmp_path: Path) -> None:
        """Script at tmp/_internal/app/src/client/client.py → portable mode."""
        script = tmp_path / "_internal" / "app" / "src" / "client" / "client.py"
        script.parent.mkdir(parents=True)
        script.touch()

        resolver = ClientPathResolver(script)

        app_dir = tmp_path / "_internal" / "app"
        assert resolver.paths.config_dir == app_dir / "config"
        assert resolver.paths.assets_dir == app_dir / "assets"
        assert resolver.paths.logs_dir == tmp_path / "logs"
        assert resolver.paths.models_dir == tmp_path / "models"
        assert resolver.paths.environment == "portable"

    def test_msix_resolves_appdata(self, tmp_path: Path) -> None:
        """Script under WindowsApps path → msix mode; config_dir under appdata."""
        fake_appdata = tmp_path / "appdata"
        script = tmp_path / "WindowsApps" / "AI.Stenographer_1.0" / "_internal" / "app" / "src" / "client" / "client.py"
        script.parent.mkdir(parents=True)
        script.touch()

        with patch("src.client.ClientPathResolver.ClientPathResolver._get_msix_appdata", return_value=fake_appdata):
            resolver = ClientPathResolver(script)

        assert resolver.paths.config_dir == fake_appdata / "config"
        assert resolver.paths.logs_dir == fake_appdata / "logs"
        assert resolver.paths.models_dir == fake_appdata / "models"
        assert resolver.paths.environment == "msix"

    def test_msix_env_var_detection(self, tmp_path: Path) -> None:
        """MSIX_PACKAGE_IDENTITY env var forces msix mode regardless of path."""
        fake_appdata = tmp_path / "appdata"
        script = tmp_path / "src" / "client" / "client.py"
        script.parent.mkdir(parents=True)
        script.touch()

        with patch.dict(os.environ, {"MSIX_PACKAGE_IDENTITY": "AI.Stenographer_1.0"}), \
             patch("src.client.ClientPathResolver.ClientPathResolver._get_msix_appdata", return_value=fake_appdata):
            resolver = ClientPathResolver(script)

        assert resolver.paths.environment == "msix"

    def test_ensure_creates_dirs_development(self, tmp_path: Path) -> None:
        """ensure_local_dir_structure creates config and logs dirs in development mode."""
        script = tmp_path / "src" / "client" / "client.py"
        script.parent.mkdir(parents=True)
        script.touch()

        resolver = ClientPathResolver(script)
        resolver.ensure_local_dir_structure()

        assert resolver.paths.config_dir.is_dir()
        assert resolver.paths.logs_dir.is_dir()

    def test_resolved_paths_is_dataclass(self, tmp_path: Path) -> None:
        """paths property returns a ClientResolvedPaths instance."""
        script = tmp_path / "src" / "client" / "client.py"
        script.parent.mkdir(parents=True)
        script.touch()

        resolver = ClientPathResolver(script)

        assert isinstance(resolver.paths, ClientResolvedPaths)
