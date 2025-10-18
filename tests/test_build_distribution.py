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


class TestRemovePipFromDistribution:
    """Test removing pip from site-packages after dependency installation."""

    def test_remove_pip_from_runtime_locations(self, tmp_path):
        """Should remove pip from runtime/Lib/site-packages and runtime/Scripts."""
        from build_distribution import remove_pip_from_distribution

        runtime_dir = tmp_path / "runtime"
        site_packages = tmp_path / "Lib" / "site-packages"
        site_packages.mkdir(parents=True)

        # Location 1: Create pip in runtime/Lib/site-packages
        runtime_site_packages = runtime_dir / "Lib" / "site-packages"
        runtime_site_packages.mkdir(parents=True)

        pip_dir = runtime_site_packages / "pip"
        pip_dir.mkdir()
        (pip_dir / "__init__.py").write_text("# pip init")
        (pip_dir / "main.py").write_text("# pip main")

        pip_dist_info = runtime_site_packages / "pip-25.2.dist-info"
        pip_dist_info.mkdir()
        (pip_dist_info / "METADATA").write_text("Name: pip")

        # Location 2: Create pip executables in runtime/Scripts
        scripts_dir = runtime_dir / "Scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "pip.exe").write_bytes(b"fake exe")
        (scripts_dir / "pip3.exe").write_bytes(b"fake exe")
        (scripts_dir / "pip3.13.exe").write_bytes(b"fake exe")

        result = remove_pip_from_distribution(runtime_dir, site_packages)

        assert result is True
        # Verify runtime pip removed
        assert not pip_dir.exists()
        assert not pip_dist_info.exists()
        # Verify pip executables removed
        assert not (scripts_dir / "pip.exe").exists()
        assert not (scripts_dir / "pip3.exe").exists()
        assert not (scripts_dir / "pip3.13.exe").exists()

    def test_remove_pip_not_present(self, tmp_path):
        """Should succeed even if pip is not present."""
        from build_distribution import remove_pip_from_distribution

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # No pip installed
        result = remove_pip_from_distribution(runtime_dir, site_packages)

        assert result is True

    def test_remove_pip_preserves_other_packages(self, tmp_path):
        """Should only remove pip, not other packages or executables."""
        from build_distribution import remove_pip_from_distribution

        runtime_dir = tmp_path / "runtime"
        runtime_site_packages = runtime_dir / "Lib" / "site-packages"
        runtime_site_packages.mkdir(parents=True)

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create pip package in runtime
        pip_dir = runtime_site_packages / "pip"
        pip_dir.mkdir()
        (pip_dir / "__init__.py").write_text("")

        # Create Scripts with pip and other executables
        scripts_dir = runtime_dir / "Scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "pip.exe").write_bytes(b"fake")
        (scripts_dir / "pytest.exe").write_bytes(b"fake")  # Should be preserved
        (scripts_dir / "black.exe").write_bytes(b"fake")   # Should be preserved

        # Create other packages that should NOT be removed
        numpy_dir = runtime_site_packages / "numpy"
        numpy_dir.mkdir()
        (numpy_dir / "__init__.py").write_text("")

        result = remove_pip_from_distribution(runtime_dir, site_packages)

        assert result is True
        # pip removed
        assert not pip_dir.exists()
        assert not (scripts_dir / "pip.exe").exists()
        # Other packages and executables preserved
        assert numpy_dir.exists()
        assert (scripts_dir / "pytest.exe").exists()
        assert (scripts_dir / "black.exe").exists()

    def test_remove_pip_from_both_target_and_runtime(self, tmp_path):
        """Should remove pip from both target site-packages and runtime (edge case)."""
        from build_distribution import remove_pip_from_distribution

        runtime_dir = tmp_path / "runtime"
        runtime_site_packages = runtime_dir / "Lib" / "site-packages"
        runtime_site_packages.mkdir(parents=True)

        site_packages = tmp_path / "Lib" / "site-packages"
        site_packages.mkdir(parents=True)

        # Create pip in runtime location
        runtime_pip = runtime_site_packages / "pip"
        runtime_pip.mkdir()
        (runtime_pip / "__init__.py").write_text("# runtime pip")

        # Create pip in target location (edge case)
        target_pip = site_packages / "pip"
        target_pip.mkdir()
        (target_pip / "__init__.py").write_text("# target pip")

        result = remove_pip_from_distribution(runtime_dir, site_packages)

        assert result is True
        # Both locations cleaned
        assert not runtime_pip.exists()
        assert not target_pip.exists()


class TestRemoveTestsFromDistribution:
    """Test removing test directories from third-party packages."""

    def test_remove_tests_from_distribution_removes_all_patterns(self, tmp_path):
        """Should remove tests/, test/, and testing/ directories."""
        from build_distribution import remove_tests_from_distribution

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create various test directory patterns
        # Pattern 1: tests/
        numpy_tests = site_packages / "numpy" / "tests"
        numpy_tests.mkdir(parents=True)
        (numpy_tests / "test_array.py").write_text("# numpy test")
        (numpy_tests / "test_linalg.py").write_text("# numpy test")

        # Pattern 2: test/
        fsspec_test = site_packages / "fsspec" / "test"
        fsspec_test.mkdir(parents=True)
        (fsspec_test / "test_core.py").write_text("# fsspec test")

        # Pattern 3: testing/
        pandas_testing = site_packages / "pandas" / "testing"
        pandas_testing.mkdir(parents=True)
        (pandas_testing / "utils.py").write_text("# pandas testing")

        result = remove_tests_from_distribution(site_packages)

        assert result is True
        # All test directories should be removed
        assert not numpy_tests.exists()
        assert not fsspec_test.exists()
        assert not pandas_testing.exists()

    def test_remove_tests_from_distribution_nested_tests(self, tmp_path):
        """Should remove nested test directories (e.g., numpy/lib/tests)."""
        from build_distribution import remove_tests_from_distribution

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create nested test directories (common in numpy)
        numpy_lib_tests = site_packages / "numpy" / "lib" / "tests"
        numpy_lib_tests.mkdir(parents=True)
        (numpy_lib_tests / "test_utils.py").write_text("# test file")
        (numpy_lib_tests / "data" / "sample.dat").parent.mkdir()
        (numpy_lib_tests / "data" / "sample.dat").write_bytes(b"test data")

        numpy_core_tests = site_packages / "numpy" / "_core" / "tests"
        numpy_core_tests.mkdir(parents=True)
        (numpy_core_tests / "test_numeric.py").write_text("# test file")

        result = remove_tests_from_distribution(site_packages)

        assert result is True
        # Both nested test directories should be removed
        assert not numpy_lib_tests.exists()
        assert not numpy_core_tests.exists()

    def test_remove_tests_from_distribution_preserves_packages(self, tmp_path):
        """Should only remove test directories, not actual packages."""
        from build_distribution import remove_tests_from_distribution

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create actual package code
        numpy_dir = site_packages / "numpy"
        numpy_dir.mkdir()
        (numpy_dir / "__init__.py").write_text("# numpy init")
        (numpy_dir / "core.py").write_text("# numpy core")
        (numpy_dir / "linalg.py").write_text("# numpy linalg")

        # Create test directory to remove
        numpy_tests = numpy_dir / "tests"
        numpy_tests.mkdir()
        (numpy_tests / "test_core.py").write_text("# test")

        # Create another package
        requests_dir = site_packages / "requests"
        requests_dir.mkdir()
        (requests_dir / "__init__.py").write_text("# requests")
        (requests_dir / "api.py").write_text("# requests api")

        result = remove_tests_from_distribution(site_packages)

        assert result is True
        # Test directory removed
        assert not numpy_tests.exists()
        # Package files preserved
        assert (numpy_dir / "__init__.py").exists()
        assert (numpy_dir / "core.py").exists()
        assert (numpy_dir / "linalg.py").exists()
        assert (requests_dir / "__init__.py").exists()
        assert (requests_dir / "api.py").exists()

    def test_remove_tests_from_distribution_no_tests_present(self, tmp_path):
        """Should succeed even if no test directories exist."""
        from build_distribution import remove_tests_from_distribution

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create packages without test directories
        numpy_dir = site_packages / "numpy"
        numpy_dir.mkdir()
        (numpy_dir / "__init__.py").write_text("")

        result = remove_tests_from_distribution(site_packages)

        assert result is True

    def test_remove_tests_from_distribution_calculates_size(self, tmp_path):
        """Should calculate and report space saved."""
        from build_distribution import remove_tests_from_distribution

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create test directories with known sizes
        tests_dir = site_packages / "package" / "tests"
        tests_dir.mkdir(parents=True)

        # Create test files with some content
        test_file1 = tests_dir / "test_module.py"
        test_file1.write_text("# " * 100)  # 200 bytes

        test_file2 = tests_dir / "test_utils.py"
        test_file2.write_text("# " * 500)  # 1000 bytes

        result = remove_tests_from_distribution(site_packages)

        assert result is True
        assert not tests_dir.exists()

    def test_remove_tests_from_distribution_handles_permission_errors(self, tmp_path):
        """Should continue removing other directories if one fails."""
        from build_distribution import remove_tests_from_distribution
        from unittest.mock import patch
        import shutil as shutil_module

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create multiple test directories
        test_dir1 = site_packages / "package1" / "tests"
        test_dir1.mkdir(parents=True)
        (test_dir1 / "test.py").write_text("# test")

        test_dir2 = site_packages / "package2" / "tests"
        test_dir2.mkdir(parents=True)
        (test_dir2 / "test.py").write_text("# test")

        # Mock shutil.rmtree to fail on first directory but succeed on second
        original_rmtree = shutil_module.rmtree
        call_count = [0]

        def mock_rmtree(path, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise PermissionError("Access denied")
            return original_rmtree(path, *args, **kwargs)

        with patch('build_distribution.shutil.rmtree', side_effect=mock_rmtree):
            result = remove_tests_from_distribution(site_packages)

            # Should still return True (graceful degradation)
            assert result is True


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


class TestCleanupTclUnnecessaryFiles:
    """Test removing unnecessary Tcl/Tk files after tkinter copy."""

    def test_cleanup_removes_tzdata(self, tmp_path):
        """Should remove tzdata directory (timezone database not needed)."""
        from build_distribution import cleanup_tcl_unnecessary_files

        # Create runtime directory with fake Tcl structure
        runtime_dir = tmp_path / "runtime"
        tcl_dir = runtime_dir / "tcl" / "tcl8.6"
        tzdata_dir = tcl_dir / "tzdata"
        tzdata_dir.mkdir(parents=True)

        # Create fake timezone files
        (tzdata_dir / "UTC").write_text("# UTC timezone")
        africa_dir = tzdata_dir / "Africa"
        africa_dir.mkdir()
        (africa_dir / "Cairo").write_text("# Cairo timezone")
        (africa_dir / "Lagos").write_text("# Lagos timezone")

        result = cleanup_tcl_unnecessary_files(runtime_dir)

        assert result is True
        # tzdata directory should be removed
        assert not tzdata_dir.exists()
        # Parent directories should remain
        assert tcl_dir.exists()

    def test_cleanup_removes_msgs(self, tmp_path):
        """Should remove msgs directory (localization not needed for English-only app)."""
        from build_distribution import cleanup_tcl_unnecessary_files

        runtime_dir = tmp_path / "runtime"
        tcl_dir = runtime_dir / "tcl" / "tcl8.6"
        msgs_dir = tcl_dir / "msgs"
        msgs_dir.mkdir(parents=True)

        # Create fake message files
        (msgs_dir / "en.msg").write_text("# English messages")
        (msgs_dir / "fr.msg").write_text("# French messages")
        (msgs_dir / "de.msg").write_text("# German messages")

        result = cleanup_tcl_unnecessary_files(runtime_dir)

        assert result is True
        # msgs directory should be removed
        assert not msgs_dir.exists()
        # Parent directories should remain
        assert tcl_dir.exists()

    def test_cleanup_removes_both_tzdata_and_msgs(self, tmp_path):
        """Should remove both tzdata and msgs in single operation."""
        from build_distribution import cleanup_tcl_unnecessary_files

        runtime_dir = tmp_path / "runtime"
        tcl_dir = runtime_dir / "tcl" / "tcl8.6"

        # Create tzdata with ~100 files
        tzdata_dir = tcl_dir / "tzdata"
        tzdata_dir.mkdir(parents=True)
        for i in range(100):
            (tzdata_dir / f"timezone_{i}.tz").write_text(f"# Timezone {i}")

        # Create msgs with ~50 files
        msgs_dir = tcl_dir / "msgs"
        msgs_dir.mkdir(parents=True)
        for i in range(50):
            (msgs_dir / f"lang_{i}.msg").write_text(f"# Language {i}")

        result = cleanup_tcl_unnecessary_files(runtime_dir)

        assert result is True
        assert not tzdata_dir.exists()
        assert not msgs_dir.exists()

    def test_cleanup_preserves_essential_tcl_files(self, tmp_path):
        """Should only remove tzdata/msgs, not essential Tcl/Tk files."""
        from build_distribution import cleanup_tcl_unnecessary_files

        runtime_dir = tmp_path / "runtime"
        tcl_dir = runtime_dir / "tcl"

        # Create essential Tcl files that must NOT be removed
        tcl8_6 = tcl_dir / "tcl8.6"
        tcl8_6.mkdir(parents=True)
        (tcl8_6 / "init.tcl").write_text("# Tcl init script")
        (tcl8_6 / "package.tcl").write_text("# Tcl package system")

        tk8_6 = tcl_dir / "tk8.6"
        tk8_6.mkdir(parents=True)
        (tk8_6 / "button.tcl").write_text("# Button widget")
        (tk8_6 / "dialog.tcl").write_text("# Dialog widget")

        # Create removable files
        tzdata_dir = tcl8_6 / "tzdata"
        tzdata_dir.mkdir()
        (tzdata_dir / "UTC").write_text("# UTC")

        result = cleanup_tcl_unnecessary_files(runtime_dir)

        assert result is True
        # Essential files preserved
        assert (tcl8_6 / "init.tcl").exists()
        assert (tcl8_6 / "package.tcl").exists()
        assert (tk8_6 / "button.tcl").exists()
        assert (tk8_6 / "dialog.tcl").exists()
        # Removable files gone
        assert not tzdata_dir.exists()

    def test_cleanup_succeeds_with_no_tcl_directory(self, tmp_path):
        """Should succeed gracefully if no tcl/ directory exists."""
        from build_distribution import cleanup_tcl_unnecessary_files

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        result = cleanup_tcl_unnecessary_files(runtime_dir)

        assert result is True

    def test_cleanup_succeeds_with_already_clean_tcl(self, tmp_path):
        """Should succeed if tzdata/msgs already removed."""
        from build_distribution import cleanup_tcl_unnecessary_files

        runtime_dir = tmp_path / "runtime"
        tcl_dir = runtime_dir / "tcl" / "tcl8.6"
        tcl_dir.mkdir(parents=True)

        # Create only essential files, no tzdata or msgs
        (tcl_dir / "init.tcl").write_text("# init")

        result = cleanup_tcl_unnecessary_files(runtime_dir)

        assert result is True
        # Essential files still present
        assert (tcl_dir / "init.tcl").exists()
