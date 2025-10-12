"""
Tests for main.py path resolution in _internal distribution structure.

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
import tempfile
import shutil


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

        # Import path resolution logic (to be implemented)
        from main import resolve_paths

        paths = resolve_paths(main_pyc)

        assert paths["APP_DIR"] == app_dir
        assert paths["INTERNAL_DIR"] == internal_dir
        assert paths["ROOT_DIR"] == root_dir
        assert paths["MODELS_DIR"] == models_dir
        assert paths["CONFIG_DIR"] == app_dir / "config"
        assert paths["ASSETS_DIR"] == app_dir / "assets"

    def test_paths_from_development_source(self, tmp_path):
        """Should resolve paths correctly when running from development ./main.py."""
        # Create development structure
        project_dir = tmp_path / "stt-project"
        project_dir.mkdir()

        main_py = project_dir / "main.py"
        main_py.write_text("# main")

        # Import path resolution logic
        from main import resolve_paths

        paths = resolve_paths(main_py)

        # In development, use current directory structure
        assert paths["APP_DIR"] == project_dir
        assert paths["MODELS_DIR"] == project_dir / "models"
        assert paths["CONFIG_DIR"] == project_dir / "config"

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

        from main import resolve_paths, get_asset_path

        paths = resolve_paths(main_pyc)
        image_path = get_asset_path("stenographer.jpg", paths)

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

        from main import resolve_paths, get_config_path

        paths = resolve_paths(main_pyc)
        config_path = get_config_path("stt_config.json", paths)

        assert config_path == config_file
        assert config_path.exists()

    def test_models_directory_creation(self, tmp_path):
        """Should create models directory in _internal/models/ if it doesn't exist."""
        # Create distribution structure WITHOUT models dir
        root_dir = tmp_path / "STT-Stenographer"
        app_dir = root_dir / "_internal" / "app"
        app_dir.mkdir(parents=True)

        # Create fake main.pyc
        main_pyc = app_dir / "main.pyc"
        main_pyc.write_bytes(b"fake")

        from main import resolve_paths, ensure_models_dir

        paths = resolve_paths(main_pyc)
        models_dir = ensure_models_dir(paths)

        assert models_dir.exists()
        assert models_dir.is_dir()
        assert models_dir == root_dir / "_internal" / "models"

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

        from main import resolve_paths

        # Should work without .py files
        paths = resolve_paths(main_pyc)
        assert paths["APP_DIR"] == app_dir

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

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(str(other_dir))

            from main import resolve_paths

            # Should resolve to absolute paths, not relative to CWD
            paths = resolve_paths(main_pyc)
            assert paths["ROOT_DIR"].is_absolute()
            assert paths["ROOT_DIR"] == root_dir

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

        from main import is_distribution_mode

        # Should return False for development (no _internal structure)
        is_dist = is_distribution_mode(main_py)
        assert is_dist is False

    def test_distribution_mode_detection(self, tmp_path):
        """Should detect when running from _internal distribution structure."""
        # Create distribution structure
        root_dir = tmp_path / "STT-Stenographer"
        app_dir = root_dir / "_internal" / "app"
        app_dir.mkdir(parents=True)

        main_pyc = app_dir / "main.pyc"
        main_pyc.write_bytes(b"fake")

        from main import is_distribution_mode

        # Should return True (has _internal structure)
        is_dist = is_distribution_mode(main_pyc)
        assert is_dist is True

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

        from main import resolve_paths

        paths = resolve_paths(main_py)

        # Should use project directory structure
        assert paths["MODELS_DIR"] == models_dir
        assert paths["CONFIG_DIR"] == config_dir
