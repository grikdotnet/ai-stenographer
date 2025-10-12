"""
Tests for build_distribution.py - Windows distribution builder.

Tests cover:
- Downloading Python embeddable package
- Extracting to _internal/runtime/
- Creating directory structure
- Verifying signed executables
"""
import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import zipfile


# Import functions from build_distribution (to be implemented)
# from build_distribution import (
#     download_embedded_python,
#     extract_embedded_python,
#     create_directory_structure,
#     verify_signatures
# )


class TestDownloadEmbeddedPython:
    """Test downloading Python embeddable package from python.org."""

    def test_download_embedded_python_success(self, tmp_path):
        """Should download Python embeddable package to cache directory."""
        from build_distribution import download_embedded_python

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Mock the download to avoid actual network call
        with patch('build_distribution.requests.get') as mock_get:
            # Create fake zip content
            fake_zip = tmp_path / "fake.zip"
            with zipfile.ZipFile(fake_zip, 'w') as zf:
                zf.writestr("python.exe", b"fake python")

            zip_content = fake_zip.read_bytes()

            mock_response = Mock()
            mock_response.headers = {'content-length': str(len(zip_content))}
            mock_response.iter_content = Mock(return_value=[zip_content])
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result = download_embedded_python("3.13.0", cache_dir)

            assert result.exists()
            assert result.suffix == ".zip"
            assert result.parent == cache_dir

    def test_download_embedded_python_cached(self, tmp_path):
        """Should skip download if package already cached."""
        from build_distribution import download_embedded_python

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create fake cached file
        cached_file = cache_dir / "python-3.13.0-embed-amd64.zip"
        cached_file.write_bytes(b"cached content")

        result = download_embedded_python("3.13.0", cache_dir)

        assert result == cached_file
        assert result.read_bytes() == b"cached content"

    def test_download_embedded_python_invalid_version(self, tmp_path):
        """Should raise error for invalid Python version."""
        from build_distribution import download_embedded_python

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        with pytest.raises(ValueError, match="Invalid Python version"):
            download_embedded_python("invalid", cache_dir)


class TestExtractEmbeddedPython:
    """Test extracting embeddable package to target directory."""

    def test_extract_embedded_python_success(self, tmp_path):
        """Should extract all files to target directory."""
        from build_distribution import extract_embedded_python

        # Create test zip file
        zip_path = tmp_path / "python.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("python.exe", b"fake python")
            zf.writestr("python313.dll", b"fake dll")
            zf.writestr("python313.zip", b"fake stdlib")

        target_dir = tmp_path / "runtime"

        extract_embedded_python(zip_path, target_dir)

        assert target_dir.exists()
        assert (target_dir / "python.exe").exists()
        assert (target_dir / "python313.dll").exists()
        assert (target_dir / "python313.zip").exists()

    def test_extract_embedded_python_missing_zip(self, tmp_path):
        """Should raise error if zip file doesn't exist."""
        from build_distribution import extract_embedded_python

        zip_path = tmp_path / "nonexistent.zip"
        target_dir = tmp_path / "runtime"

        with pytest.raises(FileNotFoundError):
            extract_embedded_python(zip_path, target_dir)

    def test_extract_embedded_python_overwrites_existing(self, tmp_path):
        """Should overwrite existing files in target directory."""
        from build_distribution import extract_embedded_python

        # Create test zip file with new content
        zip_path = tmp_path / "python.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("python.exe", b"new python")

        target_dir = tmp_path / "runtime"
        target_dir.mkdir()
        (target_dir / "python.exe").write_bytes(b"old python")

        extract_embedded_python(zip_path, target_dir)

        assert (target_dir / "python.exe").read_bytes() == b"new python"


class TestCreateDirectoryStructure:
    """Test creating _internal directory structure."""

    def test_create_directory_structure_success(self, tmp_path):
        """Should create all required directories."""
        from build_distribution import create_directory_structure

        root_dir = tmp_path / "STT-Stenographer"

        paths = create_directory_structure(root_dir)

        # Check return value
        assert isinstance(paths, dict)
        assert "root" in paths
        assert "internal" in paths
        assert "runtime" in paths
        assert "lib" in paths
        assert "app" in paths
        assert "app_src" in paths
        assert "app_config" in paths
        assert "app_assets" in paths
        assert "models" in paths

        # Check directories exist
        assert root_dir.exists()
        assert (root_dir / "_internal").exists()
        assert (root_dir / "_internal" / "runtime").exists()
        assert (root_dir / "_internal" / "Lib" / "site-packages").exists()
        assert (root_dir / "_internal" / "app").exists()
        assert (root_dir / "_internal" / "app" / "src").exists()
        assert (root_dir / "_internal" / "app" / "config").exists()
        assert (root_dir / "_internal" / "app" / "assets").exists()
        assert (root_dir / "_internal" / "models").exists()

    def test_create_directory_structure_idempotent(self, tmp_path):
        """Should succeed even if directories already exist."""
        from build_distribution import create_directory_structure

        root_dir = tmp_path / "STT-Stenographer"

        # Create twice
        paths1 = create_directory_structure(root_dir)
        paths2 = create_directory_structure(root_dir)

        # Both should succeed and return same paths
        assert paths1 == paths2
        assert root_dir.exists()


class TestVerifySignatures:
    """Test verifying digital signatures of Python executables."""

    def test_verify_signatures_success(self, tmp_path):
        """Should verify python.exe and pythonw.exe are signed."""
        from build_distribution import verify_signatures

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create fake executables
        (runtime_dir / "python.exe").write_bytes(b"fake exe")
        (runtime_dir / "pythonw.exe").write_bytes(b"fake exe")

        # Mock signature verification
        with patch('build_distribution.verify_file_signature') as mock_verify:
            mock_verify.return_value = True

            result = verify_signatures(runtime_dir)

            assert result is True
            assert mock_verify.call_count == 2

    def test_verify_signatures_missing_files(self, tmp_path):
        """Should return False if executables don't exist."""
        from build_distribution import verify_signatures

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        result = verify_signatures(runtime_dir)

        assert result is False

    def test_verify_signatures_unsigned(self, tmp_path):
        """Should return False if executables are unsigned."""
        from build_distribution import verify_signatures

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        (runtime_dir / "python.exe").write_bytes(b"fake exe")
        (runtime_dir / "pythonw.exe").write_bytes(b"fake exe")

        # Mock signature verification to return False
        with patch('build_distribution.verify_file_signature') as mock_verify:
            mock_verify.return_value = False

            result = verify_signatures(runtime_dir)

            assert result is False


class TestCreatePthFile:
    """Test creating python313._pth configuration file."""

    def test_create_pth_file_success(self, tmp_path):
        """Should create _pth file with correct paths."""
        from build_distribution import create_pth_file

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        paths = [
            "_internal/runtime/python313.zip",
            "_internal/Lib/site-packages",
            "_internal/app",
            "import site"
        ]

        create_pth_file(runtime_dir, paths)

        pth_file = runtime_dir / "python313._pth"
        assert pth_file.exists()

        content = pth_file.read_text()
        assert "_internal/runtime/python313.zip" in content
        assert "_internal/Lib/site-packages" in content
        assert "_internal/app" in content
        assert "import site" in content

    def test_create_pth_file_overwrites_existing(self, tmp_path):
        """Should overwrite existing _pth file."""
        from build_distribution import create_pth_file

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create existing file
        pth_file = runtime_dir / "python313._pth"
        pth_file.write_text("old content")

        paths = ["_internal/app"]
        create_pth_file(runtime_dir, paths)

        content = pth_file.read_text()
        assert content == "_internal/app\n"
        assert "old content" not in content


class TestEnablePip:
    """Test enabling pip in embedded Python."""

    def test_enable_pip_success(self, tmp_path):
        """Should download get-pip.py and run it."""
        from build_distribution import enable_pip

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create fake python.exe
        python_exe = runtime_dir / "python.exe"
        python_exe.write_bytes(b"fake python")

        # Mock HTTP download and subprocess
        with patch('build_distribution.requests.get') as mock_get, \
             patch('build_distribution.subprocess.run') as mock_run:

            mock_response = Mock()
            mock_response.content = b"# get-pip.py script"
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            mock_run.return_value = Mock(returncode=0)

            enable_pip(runtime_dir)

            # Verify get-pip.py was downloaded (then cleaned up)
            get_pip = runtime_dir / "get-pip.py"
            assert not get_pip.exists()  # Should be cleaned up after install

            # Verify python was called to install pip
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "python.exe" in str(call_args[0])
            assert "get-pip.py" in str(call_args[1])

    def test_enable_pip_download_failure(self, tmp_path):
        """Should raise error if get-pip.py download fails."""
        from build_distribution import enable_pip

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        with patch('build_distribution.requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")

            with pytest.raises(Exception):
                enable_pip(runtime_dir)


class TestVerifyPip:
    """Test verifying pip is available."""

    def test_verify_pip_success(self, tmp_path):
        """Should return True if pip is available."""
        from build_distribution import verify_pip

        python_exe = tmp_path / "python.exe"

        with patch('build_distribution.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="pip 24.0")

            result = verify_pip(python_exe)

            assert result is True

    def test_verify_pip_not_available(self, tmp_path):
        """Should return False if pip is not available."""
        from build_distribution import verify_pip

        python_exe = tmp_path / "python.exe"

        with patch('build_distribution.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1)

            result = verify_pip(python_exe)

            assert result is False


class TestInstallDependencies:
    """Test installing dependencies into distribution."""

    def test_install_dependencies_success(self, tmp_path):
        """Should install packages from requirements.txt."""
        from build_distribution import install_dependencies

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        python_exe = runtime_dir / "python.exe"
        python_exe.write_bytes(b"fake python")

        target_dir = tmp_path / "Lib" / "site-packages"
        target_dir.mkdir(parents=True)

        requirements = tmp_path / "requirements.txt"
        requirements.write_text("requests\nnumpy")

        with patch('build_distribution.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="Successfully installed")

            result = install_dependencies(python_exe, target_dir, requirements)

            assert result is True
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "pip" in str(call_args)
            assert "install" in str(call_args)
            assert "--target" in str(call_args)

    def test_install_dependencies_failure(self, tmp_path):
        """Should return False if pip install fails."""
        from build_distribution import install_dependencies

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        python_exe = runtime_dir / "python.exe"
        python_exe.write_bytes(b"fake python")

        target_dir = tmp_path / "Lib" / "site-packages"
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("nonexistent-package")

        with patch('build_distribution.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr="ERROR: Package not found")

            result = install_dependencies(python_exe, target_dir, requirements)

            assert result is False


class TestVerifyNativeLibraries:
    """Test verifying native libraries (DLLs) are present."""

    def test_verify_native_libraries_all_present(self, tmp_path):
        """Should return True if all critical DLLs exist."""
        from build_distribution import verify_native_libraries

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create ONNX runtime DLLs
        onnx_dir = site_packages / "onnxruntime" / "capi"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "onnxruntime.dll").write_bytes(b"fake dll")
        (onnx_dir / "onnxruntime_providers_shared.dll").write_bytes(b"fake dll")

        # Create sounddevice DLLs
        sd_dir = site_packages / "_sounddevice_data"
        sd_dir.mkdir(parents=True)
        (sd_dir / "portaudio_x64.dll").write_bytes(b"fake dll")

        result = verify_native_libraries(site_packages)

        assert result["onnxruntime"] is True
        assert result["sounddevice"] is True

    def test_verify_native_libraries_missing(self, tmp_path):
        """Should return False for missing libraries."""
        from build_distribution import verify_native_libraries

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        result = verify_native_libraries(site_packages)

        assert result["onnxruntime"] is False
        assert result["sounddevice"] is False


class TestImports:
    """Test verifying Python packages can be imported."""

    def test_imports_success(self, tmp_path):
        """Should return True if all modules import successfully."""
        from build_distribution import test_imports

        python_exe = tmp_path / "python.exe"

        with patch('build_distribution.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = test_imports(python_exe, ["numpy", "onnxruntime"])

            assert result["numpy"] is True
            assert result["onnxruntime"] is True
            assert mock_run.call_count == 2

    def test_imports_failure(self, tmp_path):
        """Should return False if module import fails."""
        from build_distribution import test_imports

        python_exe = tmp_path / "python.exe"

        with patch('build_distribution.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1)

            result = test_imports(python_exe, ["nonexistent"])

            assert result["nonexistent"] is False


class TestCopyApplicationCode:
    """Test copying application code to distribution."""

    def test_copy_application_code_success(self, tmp_path):
        """Should copy all .py files from src/ to _internal/app/."""
        from build_distribution import copy_application_code

        # Create source structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("# Init")
        (src_dir / "module1.py").write_text("def func(): pass")
        (src_dir / "module2.py").write_text("class MyClass: pass")

        # Create main.py
        main_py = tmp_path / "main.py"
        main_py.write_text("print('main')")

        # Create distribution structure
        dist_dir = tmp_path / "dist" / "STT-Stenographer"
        app_dir = dist_dir / "_internal" / "app"
        app_dir.mkdir(parents=True)

        result = copy_application_code(src_dir, main_py, app_dir)

        assert result is True
        assert (app_dir / "src" / "__init__.py").exists()
        assert (app_dir / "src" / "module1.py").exists()
        assert (app_dir / "src" / "module2.py").exists()
        assert (app_dir / "main.py").exists()

    def test_copy_application_code_missing_source(self, tmp_path):
        """Should return False if source directory doesn't exist."""
        from build_distribution import copy_application_code

        src_dir = tmp_path / "nonexistent"
        main_py = tmp_path / "main.py"
        app_dir = tmp_path / "app"
        app_dir.mkdir()

        result = copy_application_code(src_dir, main_py, app_dir)

        assert result is False

    def test_copy_application_code_preserves_structure(self, tmp_path):
        """Should preserve directory structure when copying."""
        from build_distribution import copy_application_code

        # Create nested source structure
        src_dir = tmp_path / "src"
        sub_dir = src_dir / "submodule"
        sub_dir.mkdir(parents=True)
        (src_dir / "__init__.py").write_text("")
        (sub_dir / "nested.py").write_text("nested_func()")

        main_py = tmp_path / "main.py"
        main_py.write_text("import src")

        app_dir = tmp_path / "app"
        app_dir.mkdir()

        result = copy_application_code(src_dir, main_py, app_dir)

        assert result is True
        assert (app_dir / "src" / "submodule" / "nested.py").exists()


class TestCompileToPyc:
    """Test compiling .py files to .pyc bytecode."""

    def test_compile_to_pyc_success(self, tmp_path):
        """Should compile all .py files in app directory to .pyc."""
        from build_distribution import compile_to_pyc

        app_dir = tmp_path / "app"
        app_dir.mkdir()
        (app_dir / "module.py").write_text("def test(): pass")

        src_dir = app_dir / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")
        (src_dir / "code.py").write_text("x = 1")

        python_exe = tmp_path / "python.exe"

        with patch('build_distribution.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = compile_to_pyc(python_exe, app_dir)

            assert result is True
            assert mock_run.called

    def test_compile_to_pyc_failure(self, tmp_path):
        """Should return False if compilation fails."""
        from build_distribution import compile_to_pyc

        app_dir = tmp_path / "app"
        app_dir.mkdir()
        python_exe = tmp_path / "python.exe"

        with patch('build_distribution.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr="Syntax error")

            result = compile_to_pyc(python_exe, app_dir)

            assert result is False

    def test_compile_to_pyc_cleans_up_files(self, tmp_path):
        """Should remove .py files and __pycache__ after compilation."""
        from build_distribution import compile_to_pyc
        import sys

        app_dir = tmp_path / "app"
        app_dir.mkdir()
        (app_dir / "module.py").write_text("def test(): pass")

        src_dir = app_dir / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("")
        (src_dir / "code.py").write_text("x = 1")

        # Use real Python to compile
        python_exe = Path(sys.executable)

        result = compile_to_pyc(python_exe, app_dir)

        assert result is True

        # Verify .py files are removed
        assert not (app_dir / "module.py").exists()
        assert not (src_dir / "__init__.py").exists()
        assert not (src_dir / "code.py").exists()

        # Verify .pyc files exist
        assert (app_dir / "module.pyc").exists()
        assert (src_dir / "__init__.pyc").exists()
        assert (src_dir / "code.pyc").exists()

        # Verify __pycache__ is removed
        assert not (src_dir / "__pycache__").exists()


class TestCreateLauncher:
    """Test creating Windows launcher shortcut."""

    def test_create_launcher_success(self, tmp_path):
        """Should create launcher shortcut with icon."""
        from build_distribution import create_launcher

        build_dir = tmp_path / "STT-Stenographer"
        build_dir.mkdir()

        # Create required files
        icon_file = build_dir / "icon.ico"
        icon_file.write_bytes(b"ICO")

        runtime_dir = build_dir / "_internal" / "runtime"
        runtime_dir.mkdir(parents=True)
        python_exe = runtime_dir / "pythonw.exe"
        python_exe.write_bytes(b"EXE")

        app_dir = build_dir / "_internal" / "app"
        app_dir.mkdir(parents=True)
        main_pyc = app_dir / "main.pyc"
        main_pyc.write_bytes(b"PYC")

        result = create_launcher(build_dir)

        assert result is True
        # Shortcut should be created
        shortcut = build_dir / "STT - Stenographer.lnk"
        assert shortcut.exists()

    def test_create_launcher_missing_files(self, tmp_path):
        """Should return False if required files are missing."""
        from build_distribution import create_launcher

        build_dir = tmp_path / "STT-Stenographer"
        build_dir.mkdir()

        result = create_launcher(build_dir)

        assert result is False


class TestCopyTkinter:
    """Test copying tkinter module to embedded Python."""

    def test_copy_tkinter_success(self, tmp_path):
        """Should copy tkinter module from system Python."""
        from build_distribution import copy_tkinter_to_distribution
        import sys

        # Create fake system Python structure
        system_python = tmp_path / "system_python"
        system_lib = system_python / "Lib"
        system_lib.mkdir(parents=True)

        # Create fake tkinter module
        tkinter_dir = system_lib / "tkinter"
        tkinter_dir.mkdir()
        (tkinter_dir / "__init__.py").write_text("# tkinter")
        (tkinter_dir / "constants.py").write_text("# constants")

        # Create fake _tkinter.pyd and required DLLs
        system_dlls = system_python / "DLLs"
        system_dlls.mkdir()
        (system_dlls / "_tkinter.pyd").write_bytes(b"fake pyd")
        (system_dlls / "zlib1.dll").write_bytes(b"fake zlib")

        # Create distribution runtime directory
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Mock sys.base_prefix to point to fake system Python
        with patch('sys.base_prefix', str(system_python)):
            result = copy_tkinter_to_distribution(runtime_dir)

            assert result is True
            # Verify tkinter module copied
            assert (runtime_dir / "Lib" / "tkinter" / "__init__.py").exists()
            assert (runtime_dir / "Lib" / "tkinter" / "constants.py").exists()
            # Verify _tkinter.pyd copied
            assert (runtime_dir / "_tkinter.pyd").exists()
            # Verify zlib1.dll copied
            assert (runtime_dir / "zlib1.dll").exists()

    def test_copy_tkinter_missing_source(self, tmp_path):
        """Should return False if system Python doesn't have tkinter."""
        from build_distribution import copy_tkinter_to_distribution
        import sys

        # Create fake system Python WITHOUT tkinter
        system_python = tmp_path / "system_python"
        system_python.mkdir()

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        with patch('sys.base_prefix', str(system_python)):
            result = copy_tkinter_to_distribution(runtime_dir)

            assert result is False

    def test_verify_tkinter_import(self, tmp_path):
        """Should verify tkinter can be imported."""
        from build_distribution import verify_tkinter
        import sys

        python_exe = Path(sys.executable)

        # Use real Python to test tkinter import
        result = verify_tkinter(python_exe)

        # Should succeed if system Python has tkinter
        assert result is True

    def test_copy_tkinter_includes_zlib_dll(self, tmp_path):
        """Should copy zlib1.dll which is required for _tkinter.pyd in Python 3.12+."""
        from build_distribution import copy_tkinter_to_distribution

        # Create fake system Python structure
        system_python = tmp_path / "system_python"
        system_lib = system_python / "Lib"
        system_lib.mkdir(parents=True)

        # Create fake tkinter module
        tkinter_dir = system_lib / "tkinter"
        tkinter_dir.mkdir()
        (tkinter_dir / "__init__.py").write_text("# tkinter")

        # Create fake DLLs directory with all required files
        system_dlls = system_python / "DLLs"
        system_dlls.mkdir()
        (system_dlls / "_tkinter.pyd").write_bytes(b"fake pyd")
        (system_dlls / "tcl86t.dll").write_bytes(b"fake tcl")
        (system_dlls / "tk86t.dll").write_bytes(b"fake tk")
        (system_dlls / "zlib1.dll").write_bytes(b"fake zlib")  # Critical for Python 3.12+

        # Create distribution runtime directory
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Mock sys.base_prefix to point to fake system Python
        with patch('sys.base_prefix', str(system_python)):
            result = copy_tkinter_to_distribution(runtime_dir)

            assert result is True
            # Verify zlib1.dll was copied (critical for _tkinter.pyd to load)
            assert (runtime_dir / "zlib1.dll").exists()
            # Verify other DLLs copied
            assert (runtime_dir / "tcl86t.dll").exists()
            assert (runtime_dir / "tk86t.dll").exists()

    def test_copy_tkinter_fails_without_zlib_dll(self, tmp_path):
        """Should fail if zlib1.dll is missing (required for Python 3.12+)."""
        from build_distribution import copy_tkinter_to_distribution

        # Create fake system Python structure
        system_python = tmp_path / "system_python"
        system_lib = system_python / "Lib"
        system_lib.mkdir(parents=True)

        # Create fake tkinter module
        tkinter_dir = system_lib / "tkinter"
        tkinter_dir.mkdir()
        (tkinter_dir / "__init__.py").write_text("# tkinter")

        # Create fake DLLs directory WITHOUT zlib1.dll
        system_dlls = system_python / "DLLs"
        system_dlls.mkdir()
        (system_dlls / "_tkinter.pyd").write_bytes(b"fake pyd")
        (system_dlls / "tcl86t.dll").write_bytes(b"fake tcl")
        (system_dlls / "tk86t.dll").write_bytes(b"fake tk")
        # zlib1.dll is MISSING

        # Create distribution runtime directory
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Mock sys.base_prefix to point to fake system Python
        with patch('sys.base_prefix', str(system_python)):
            result = copy_tkinter_to_distribution(runtime_dir)

            # Should fail because zlib1.dll is critical
            assert result is False


class TestCopyAssetsAndConfig:
    """Test copying assets and configuration files."""

    def test_copy_assets_and_config_success(self, tmp_path):
        """Should copy config files and icon to distribution."""
        from build_distribution import copy_assets_and_config

        # Create source files
        project_root = tmp_path / "project"
        project_root.mkdir()

        config_dir = project_root / "config"
        config_dir.mkdir()
        (config_dir / "stt_config.json").write_text('{"test": true}')

        icon_file = project_root / "icon.ico"
        icon_file.write_bytes(b"ICO_DATA")

        # Create distribution structure
        build_dir = tmp_path / "dist"
        build_dir.mkdir()
        app_dir = build_dir / "_internal" / "app"
        app_dir.mkdir(parents=True)

        result = copy_assets_and_config(project_root, build_dir, app_dir)

        assert result is True
        assert (app_dir / "config" / "stt_config.json").exists()
        assert (build_dir / "icon.ico").exists()

    def test_copy_assets_and_config_missing_files(self, tmp_path):
        """Should handle missing config/icon files gracefully."""
        from build_distribution import copy_assets_and_config

        project_root = tmp_path / "project"
        project_root.mkdir()

        build_dir = tmp_path / "dist"
        build_dir.mkdir()
        app_dir = build_dir / "_internal" / "app"
        app_dir.mkdir(parents=True)

        # Should not fail even if files are missing
        result = copy_assets_and_config(project_root, build_dir, app_dir)

        # Returns True even if optional files are missing
        assert result is True
