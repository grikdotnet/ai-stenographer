"""
Tests for PathResolver - path resolution and first-run initialization.

Focuses on:
- Distribution mode detection (msix, portable, development)
- Path resolution for different modes
- MSIX-specific bundled resource copying (configs, Silero VAD)
"""
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.PathResolver import PathResolver


class TestPathResolver:
    """Test suite for PathResolver functionality."""

    def test_detect_msix_mode_from_env_var(self, tmp_path):
        """Detects MSIX mode when MSIX_PACKAGE_IDENTITY environment variable is set."""
        script_path = tmp_path / "main.py"
        script_path.write_text("")

        appdata = tmp_path / "AppData"
        appdata.mkdir()

        with patch.dict('os.environ', {'MSIX_PACKAGE_IDENTITY': 'AI.Stenographer_1.0.0.0_x64'}):
            with patch.object(PathResolver, '_get_msix_local_cache_path', return_value=appdata):
                resolver = PathResolver(script_path)
                assert resolver.mode == "msix"

    def test_detect_msix_mode_from_path(self, tmp_path):
        """Detects MSIX mode when script path contains WindowsApps."""
        script_path = Path("C:/Program Files/WindowsApps/AI.Stenographer/_internal/app/main.pyc")

        appdata = tmp_path / "AppData"
        appdata.mkdir()

        with patch.object(PathResolver, '_get_msix_local_cache_path', return_value=appdata):
            resolver = PathResolver(script_path)
            assert resolver.mode == "msix"

    def test_detect_portable_mode(self, tmp_path):
        """Detects portable mode from _internal/app structure."""
        script_path = tmp_path / "_internal" / "app" / "main.pyc"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("")

        resolver = PathResolver(script_path)
        assert resolver.mode == "portable"

    def test_detect_development_mode(self, tmp_path):
        """Detects development mode for standard project structure."""
        script_path = tmp_path / "src" / "main.py"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("")

        resolver = PathResolver(script_path)
        assert resolver.mode == "development"

    def test_msix_silero_copy_success(self, tmp_path):
        """Copies bundled Silero VAD to AppData on MSIX first run."""
        # Setup MSIX-like structure
        # _internal/models/silero_vad/ (bundled, read-only)
        # AppData/models/silero_vad/ (target, writable)

        internal_dir = tmp_path / "_internal"
        app_dir = internal_dir / "app"
        bundled_models = internal_dir / "models" / "silero_vad"
        bundled_models.mkdir(parents=True)
        (bundled_models / "silero_vad.onnx").write_text("mock model data")
        (bundled_models / "silero_vad.jit").write_text("mock jit data")

        script_path = app_dir / "main.pyc"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("")

        appdata = tmp_path / "AppData"
        appdata.mkdir()

        # Mock MSIX environment
        with patch.dict('os.environ', {'MSIX_PACKAGE_IDENTITY': 'AI.Stenographer_1.0.0.0_x64'}):
            with patch.object(PathResolver, '_get_msix_local_cache_path', return_value=appdata):
                resolver = PathResolver(script_path)
                resolver.ensure_local_dir_structure()

                # Verify Silero was copied to AppData
                target_silero = appdata / "models" / "silero_vad" / "silero_vad.onnx"
                assert target_silero.exists()
                assert target_silero.read_text() == "mock model data"

                # Verify all files were copied
                target_jit = appdata / "models" / "silero_vad" / "silero_vad.jit"
                assert target_jit.exists()

    def test_msix_silero_already_exists_no_copy(self, tmp_path):
        """Does not copy Silero if already present in AppData (idempotent)."""
        # Setup existing Silero in AppData
        internal_dir = tmp_path / "_internal"
        app_dir = internal_dir / "app"
        bundled_models = internal_dir / "models" / "silero_vad"
        bundled_models.mkdir(parents=True)
        (bundled_models / "silero_vad.onnx").write_text("bundled version")

        script_path = app_dir / "main.pyc"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("")

        appdata = tmp_path / "AppData"
        existing_silero = appdata / "models" / "silero_vad"
        existing_silero.mkdir(parents=True)
        (existing_silero / "silero_vad.onnx").write_text("existing version")

        # Mock MSIX environment
        with patch.dict('os.environ', {'MSIX_PACKAGE_IDENTITY': 'AI.Stenographer_1.0.0.0_x64'}):
            with patch.object(PathResolver, '_get_msix_local_cache_path', return_value=appdata):
                resolver = PathResolver(script_path)
                resolver.ensure_local_dir_structure()

                # Verify existing Silero was NOT overwritten
                target_silero = appdata / "models" / "silero_vad" / "silero_vad.onnx"
                assert target_silero.read_text() == "existing version"

    def test_msix_missing_bundled_silero_graceful(self, tmp_path, caplog):
        """Handles missing bundled Silero gracefully (logs warning, no exception)."""
        # Setup MSIX structure WITHOUT bundled Silero
        internal_dir = tmp_path / "_internal"
        app_dir = internal_dir / "app"
        script_path = app_dir / "main.pyc"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("")

        appdata = tmp_path / "AppData"
        appdata.mkdir()

        # Mock MSIX environment
        with patch.dict('os.environ', {'MSIX_PACKAGE_IDENTITY': 'AI.Stenographer_1.0.0.0_x64'}):
            with patch.object(PathResolver, '_get_msix_local_cache_path', return_value=appdata):
                resolver = PathResolver(script_path)

                # Should not raise exception
                resolver.ensure_local_dir_structure()

                # Verify warning was logged
                assert "Bundled Silero VAD not found" in caplog.text

    def test_portable_mode_no_silero_copy(self, tmp_path):
        """Does not attempt Silero copy in portable mode (not MSIX)."""
        # Setup portable structure with bundled Silero
        internal_dir = tmp_path / "_internal"
        app_dir = internal_dir / "app"
        bundled_models = internal_dir / "models" / "silero_vad"
        bundled_models.mkdir(parents=True)
        (bundled_models / "silero_vad.onnx").write_text("bundled")

        script_path = app_dir / "main.pyc"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("")

        resolver = PathResolver(script_path)
        resolver.ensure_local_dir_structure()

        # In portable mode, models_dir is _internal/models (same as bundled location)
        # No copying should occur
        assert resolver.mode == "portable"
        assert resolver.paths.models_dir == internal_dir / "models"

    def test_msix_config_copy_on_first_run(self, tmp_path):
        """Copies bundled configs to AppData on MSIX first run."""
        # Setup MSIX structure with bundled config
        internal_dir = tmp_path / "_internal"
        app_dir = internal_dir / "app"
        bundled_config = app_dir / "config"
        bundled_config.mkdir(parents=True)
        (bundled_config / "settings.json").write_text('{"test": true}')

        script_path = app_dir / "main.pyc"
        script_path.write_text("")

        appdata = tmp_path / "AppData"
        appdata.mkdir()

        # Mock MSIX environment
        with patch.dict('os.environ', {'MSIX_PACKAGE_IDENTITY': 'AI.Stenographer_1.0.0.0_x64'}):
            with patch.object(PathResolver, '_get_msix_local_cache_path', return_value=appdata):
                resolver = PathResolver(script_path)
                resolver.ensure_local_dir_structure()

                # Verify config was copied to AppData
                target_config = appdata / "config" / "settings.json"
                assert target_config.exists()
                assert target_config.read_text() == '{"test": true}'

    def test_msix_creates_all_directories(self, tmp_path):
        """Ensures all required directories are created in MSIX mode."""
        internal_dir = tmp_path / "_internal"
        app_dir = internal_dir / "app"
        script_path = app_dir / "main.pyc"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("")

        appdata = tmp_path / "AppData"

        with patch.dict('os.environ', {'MSIX_PACKAGE_IDENTITY': 'AI.Stenographer_1.0.0.0_x64'}):
            with patch.object(PathResolver, '_get_msix_local_cache_path', return_value=appdata):
                resolver = PathResolver(script_path)
                resolver.ensure_local_dir_structure()

                # Verify all directories were created
                assert (appdata / "models").exists()
                assert (appdata / "config").exists()
                assert (appdata / "logs").exists()
