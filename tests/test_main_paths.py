"""
Tests for PathResolver in _internal distribution structure.

Tests verify that paths resolve correctly when running from:
- Development: ./main.py (source code)
- Distribution: _internal/app/main.pyc (bytecode)

Path structure in distribution:
STT-Stenographer/
├── stenographer.jpg
├── icon.ico
└── _internal/
    ├── runtime/
    ├── app/
    │   ├── main.pyc
    │   ├── src/
    │   ├── config/
    │   └── assets/
    └── models/
"""
import pytest
from pathlib import Path
import sys
import os

from src.PathResolver import PathResolver, ResolvedPaths


class TestPathResolution:
    """Test path resolution for _internal structure."""

    def test_paths_from_internal_structure(self, tmp_path):
        """Should resolve paths correctly when running from _internal/app/main.pyc."""
        # Create distribution structure
        root_dir = tmp_path / "STT-Stenographer"
        internal_dir = root_dir / "_internal"
        app_dir = internal_dir / "app"
        models_dir = internal_dir / "models"

        app_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)

        # Create fake main.pyc
        main_pyc = app_dir / "main.pyc"
        main_pyc.write_bytes(b"fake bytecode")

        resolver = PathResolver(main_pyc)
        paths = resolver.paths

        assert paths.app_dir == app_dir
        assert paths.internal_dir == internal_dir
        assert paths.root_dir == root_dir
        assert paths.models_dir == models_dir
        assert paths.config_dir == app_dir / "config"
        assert paths.assets_dir == app_dir / "assets"

    def test_paths_from_development_source(self, tmp_path):
        """Should resolve paths correctly when running from development ./main.py."""
        # Create development structure
        project_dir = tmp_path / "stt-project"
        project_dir.mkdir()

        main_py = project_dir / "main.py"
        main_py.write_text("# main")

        resolver = PathResolver(main_py)
        paths = resolver.paths

        # In development, use current directory structure
        assert paths.app_dir == project_dir
        assert paths.models_dir == project_dir / "models"
        assert paths.config_dir == project_dir / "config"

    def test_assets_loading_from_internal(self, tmp_path):
        """Should load stenographer.jpg from correct location in distribution."""
        # Create distribution structure
        root_dir = tmp_path / "STT-Stenographer"
        app_dir = root_dir / "_internal" / "app"
        assets_dir = app_dir / "assets"
        assets_dir.mkdir(parents=True)

        # Create fake image in root (user-visible)
        root_image = root_dir / "stenographer.jpg"
        root_image.write_bytes(b"ROOT_IMAGE")

        # Create fake image in assets (not used)
        assets_image = assets_dir / "stenographer.jpg"
        assets_image.write_bytes(b"ASSETS_IMAGE")

        # Create fake main.pyc
        main_pyc = app_dir / "main.pyc"
        main_pyc.write_bytes(b"fake")

        resolver = PathResolver(main_pyc)
        image_path = resolver.get_asset_path("stenographer.jpg")

        # Should use root directory image (visible to user)
        assert image_path == root_image
        assert image_path.exists()
        assert image_path.read_bytes() == b"ROOT_IMAGE"

    def test_config_loading_from_internal(self, tmp_path):
        """Should load stt_config.json from _internal/app/config/."""
        # Create distribution structure
        root_dir = tmp_path / "STT-Stenographer"
        app_dir = root_dir / "_internal" / "app"
        config_dir = app_dir / "config"
        config_dir.mkdir(parents=True)

        # Create config file
        config_file = config_dir / "stt_config.json"
        config_file.write_text('{"test": true}')

        # Create fake main.pyc
        main_pyc = app_dir / "main.pyc"
        main_pyc.write_bytes(b"fake")

        resolver = PathResolver(main_pyc)
        config_path = resolver.get_config_path("stt_config.json")

        assert config_path == config_file
        assert config_path.exists()

    def test_local_dir_structure_creation(self, tmp_path):
        """Should create models and config directories if they don't exist."""
        # Create distribution structure WITHOUT models/config dirs
        root_dir = tmp_path / "STT-Stenographer"
        app_dir = root_dir / "_internal" / "app"
        app_dir.mkdir(parents=True)

        # Create fake main.pyc
        main_pyc = app_dir / "main.pyc"
        main_pyc.write_bytes(b"fake")

        resolver = PathResolver(main_pyc)
        resolver.ensure_local_dir_structure()

        # Both directories should exist
        assert resolver.paths.models_dir.exists()
        assert resolver.paths.models_dir.is_dir()
        assert resolver.paths.models_dir == root_dir / "_internal" / "models"

        assert resolver.paths.config_dir.exists()
        assert resolver.paths.config_dir.is_dir()
        assert resolver.paths.config_dir == app_dir / "config"

    def test_paths_work_with_bytecode_no_source(self, tmp_path):
        """Should resolve paths correctly when .py source files don't exist (bytecode-only)."""
        # Create distribution with only .pyc files
        root_dir = tmp_path / "STT-Stenographer"
        app_dir = root_dir / "_internal" / "app"
        src_dir = app_dir / "src"
        src_dir.mkdir(parents=True)

        # Create only .pyc files (no .py)
        main_pyc = app_dir / "main.pyc"
        main_pyc.write_bytes(b"fake bytecode")

        module_pyc = src_dir / "pipeline.pyc"
        module_pyc.write_bytes(b"fake bytecode")

        # Verify no .py files exist
        assert not (app_dir / "main.py").exists()
        assert not (src_dir / "pipeline.py").exists()

        resolver = PathResolver(main_pyc)
        paths = resolver.paths

        # Should work without .py files
        assert paths.app_dir == app_dir

    def test_working_directory_independence(self, tmp_path):
        """Should resolve paths correctly regardless of current working directory."""
        # Create distribution structure
        root_dir = tmp_path / "STT-Stenographer"
        app_dir = root_dir / "_internal" / "app"
        app_dir.mkdir(parents=True)

        main_pyc = app_dir / "main.pyc"
        main_pyc.write_bytes(b"fake")

        # Change working directory to somewhere else
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        original_cwd = os.getcwd()
        try:
            os.chdir(str(other_dir))

            resolver = PathResolver(main_pyc)
            paths = resolver.paths

            # Should resolve to absolute paths, not relative to CWD
            assert paths.root_dir.is_absolute()
            assert paths.root_dir == root_dir

        finally:
            os.chdir(original_cwd)


class TestPathResolutionDevelopmentMode:
    """Test path resolution in development mode (running from source)."""

    def test_development_mode_detection(self, tmp_path):
        """Should detect when running from development source tree."""
        # Create development structure
        dev_dir = tmp_path / "stt-dev"
        dev_dir.mkdir()

        main_py = dev_dir / "main.py"
        main_py.write_text("# main")

        resolver = PathResolver(main_py)

        # Should return "development" for development (no _internal structure)
        assert resolver.mode == "development"

    def test_distribution_mode_detection(self, tmp_path):
        """Should detect when running from _internal distribution structure."""
        # Create distribution structure
        root_dir = tmp_path / "STT-Stenographer"
        app_dir = root_dir / "_internal" / "app"
        app_dir.mkdir(parents=True)

        main_pyc = app_dir / "main.pyc"
        main_pyc.write_bytes(b"fake")

        resolver = PathResolver(main_pyc)

        # Should return "portable" (has _internal structure)
        assert resolver.mode == "portable"

    def test_development_uses_relative_paths(self, tmp_path):
        """In development mode, should use relative paths from project root."""
        # Create development structure
        dev_dir = tmp_path / "stt-dev"
        models_dir = dev_dir / "models"
        config_dir = dev_dir / "config"

        dev_dir.mkdir()
        models_dir.mkdir()
        config_dir.mkdir()

        main_py = dev_dir / "main.py"
        main_py.write_text("# main")

        resolver = PathResolver(main_py)
        paths = resolver.paths

        # Should use project directory structure
        assert paths.models_dir == models_dir
        assert paths.config_dir == config_dir
