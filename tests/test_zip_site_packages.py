"""
Tests for zip_site_packages() - Package .pyc files into .zip archives.

Tests cover:
- Creating .zip archives for each package
- Preserving native binaries (.pyd, .dll) outside .zip
- Removing source .pyc files after zipping
- Import compatibility from .zip archives
- Handling single-file modules
- Preserving data folders
"""
import pytest
from pathlib import Path
import zipfile
import sys
from unittest.mock import patch


class TestZipSitePackages:
    """Test packaging .pyc files into .zip archives."""

    def test_zip_site_packages_creates_archives(self, tmp_path):
        """Should create .zip archives for each package directory."""
        from build_distribution import zip_site_packages

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create fake package with .pyc files
        numpy_dir = site_packages / "numpy"
        numpy_dir.mkdir()
        (numpy_dir / "__init__.pyc").write_bytes(b"PYC" * 100)
        (numpy_dir / "core.pyc").write_bytes(b"PYC" * 100)

        subdir = numpy_dir / "linalg"
        subdir.mkdir()
        (subdir / "__init__.pyc").write_bytes(b"PYC" * 100)
        (subdir / "linalg.pyc").write_bytes(b"PYC" * 100)

        result = zip_site_packages(site_packages)

        assert result is True
        # Verify .zip archive created
        numpy_zip = site_packages / "numpy.zip"
        assert numpy_zip.exists()
        assert numpy_zip.is_file()

        # Verify .zip contains all .pyc files
        with zipfile.ZipFile(numpy_zip, 'r') as zf:
            names = zf.namelist()
            assert "numpy/__init__.pyc" in names
            assert "numpy/core.pyc" in names
            assert "numpy/linalg/__init__.pyc" in names
            assert "numpy/linalg/linalg.pyc" in names

    def test_zip_preserves_native_binaries(self, tmp_path):
        """Should skip packages with native binaries (keep entire package uncompressed)."""
        from build_distribution import zip_site_packages

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create package with .pyc and native binaries
        numpy_dir = site_packages / "numpy"
        numpy_dir.mkdir()
        (numpy_dir / "__init__.pyc").write_bytes(b"PYC" * 100)
        (numpy_dir / "_core.cp313-win_amd64.pyd").write_bytes(b"PYD" * 100)

        linalg_dir = numpy_dir / "linalg"
        linalg_dir.mkdir()
        (linalg_dir / "lapack_lite.cp313-win_amd64.pyd").write_bytes(b"PYD" * 100)

        result = zip_site_packages(site_packages)

        assert result is True

        # Verify package with native extensions is NOT zipped
        numpy_zip = site_packages / "numpy.zip"
        assert not numpy_zip.exists()

        # Verify original package directory and files remain intact
        assert numpy_dir.exists()
        assert (numpy_dir / "__init__.pyc").exists()
        assert (numpy_dir / "_core.cp313-win_amd64.pyd").exists()
        assert (linalg_dir / "lapack_lite.cp313-win_amd64.pyd").exists()

    def test_zip_removes_source_pyc(self, tmp_path):
        """Should remove original .pyc files after successful zipping."""
        from build_distribution import zip_site_packages

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create package with .pyc files
        requests_dir = site_packages / "requests"
        requests_dir.mkdir()
        init_pyc = requests_dir / "__init__.pyc"
        api_pyc = requests_dir / "api.pyc"
        init_pyc.write_bytes(b"PYC" * 100)
        api_pyc.write_bytes(b"PYC" * 100)

        result = zip_site_packages(site_packages)

        assert result is True

        # Verify original .pyc files removed
        assert not init_pyc.exists()
        assert not api_pyc.exists()

        # Verify package directory removed (since only .pyc files were inside)
        assert not requests_dir.exists()

        # Verify .zip exists with the files
        requests_zip = site_packages / "requests.zip"
        assert requests_zip.exists()

    def test_zip_import_compatibility(self, tmp_path):
        """Should create .zip archives that Python can import from."""
        from build_distribution import zip_site_packages

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create real compilable Python code
        test_package = site_packages / "testpkg"
        test_package.mkdir()

        # Write .py files first
        (test_package / "__init__.py").write_text("VERSION = '1.0'")
        (test_package / "utils.py").write_text("def greet(): return 'hello'")

        # Compile to .pyc using real Python
        import py_compile
        import importlib.util

        py_compile.compile(test_package / "__init__.py",
                          test_package / "__init__.pyc",
                          doraise=True)
        py_compile.compile(test_package / "utils.py",
                          test_package / "utils.pyc",
                          doraise=True)

        # Remove .py files (distribution has only .pyc)
        (test_package / "__init__.py").unlink()
        (test_package / "utils.py").unlink()

        result = zip_site_packages(site_packages)

        assert result is True

        # Add .zip to sys.path and try importing
        testpkg_zip = site_packages / "testpkg.zip"
        assert testpkg_zip.exists()

        sys.path.insert(0, str(testpkg_zip))
        try:
            # Import from .zip
            import testpkg
            import testpkg.utils

            assert testpkg.VERSION == '1.0'
            assert testpkg.utils.greet() == 'hello'
        finally:
            # Cleanup
            sys.path.remove(str(testpkg_zip))
            if 'testpkg' in sys.modules:
                del sys.modules['testpkg']
            if 'testpkg.utils' in sys.modules:
                del sys.modules['testpkg.utils']

    def test_zip_handles_single_file_modules(self, tmp_path):
        """Should keep single-file modules as .pyc (no zipping needed)."""
        from build_distribution import zip_site_packages

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create single .pyc file (not a package directory)
        sounddevice_pyc = site_packages / "sounddevice.pyc"
        sounddevice_pyc.write_bytes(b"PYC" * 100)

        typing_extensions_pyc = site_packages / "typing_extensions.pyc"
        typing_extensions_pyc.write_bytes(b"PYC" * 100)

        result = zip_site_packages(site_packages)

        assert result is True

        # Verify single-file modules kept as .pyc (no .zip created)
        assert sounddevice_pyc.exists()
        assert typing_extensions_pyc.exists()

        # No .zip files should be created for single-file modules
        assert not (site_packages / "sounddevice.zip").exists()
        assert not (site_packages / "typing_extensions.zip").exists()

    def test_zip_preserves_data_folders(self, tmp_path):
        """Should preserve data folders like numpy.libs/, _sounddevice_data/."""
        from build_distribution import zip_site_packages

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create package with data folder
        numpy_dir = site_packages / "numpy"
        numpy_dir.mkdir()
        (numpy_dir / "__init__.pyc").write_bytes(b"PYC" * 100)

        # Create .libs data folder (contains DLLs)
        libs_dir = site_packages / "numpy.libs"
        libs_dir.mkdir()
        (libs_dir / "openblas.dll").write_bytes(b"DLL" * 100)
        (libs_dir / "libgfortran.dll").write_bytes(b"DLL" * 100)

        # Create _data folder
        data_dir = site_packages / "_sounddevice_data"
        data_dir.mkdir()
        (data_dir / "portaudio_x64.dll").write_bytes(b"DLL" * 100)

        result = zip_site_packages(site_packages)

        assert result is True

        # Verify data folders still exist
        assert libs_dir.exists()
        assert libs_dir.is_dir()
        assert (libs_dir / "openblas.dll").exists()
        assert (libs_dir / "libgfortran.dll").exists()

        assert data_dir.exists()
        assert data_dir.is_dir()
        assert (data_dir / "portaudio_x64.dll").exists()

    def test_zip_preserves_dist_info_metadata(self, tmp_path):
        """Should preserve .dist-info folders for packages like hf_xet."""
        from build_distribution import zip_site_packages

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create package
        hf_xet_dir = site_packages / "hf_xet"
        hf_xet_dir.mkdir()
        (hf_xet_dir / "__init__.pyc").write_bytes(b"PYC" * 100)

        # Create .dist-info metadata (needed for runtime detection)
        dist_info = site_packages / "hf_xet-1.1.10.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text("Name: hf-xet\nVersion: 1.1.10")
        (dist_info / "RECORD").write_text("hf_xet/__init__.py,sha256=abc,123")

        result = zip_site_packages(site_packages)

        assert result is True

        # Verify .dist-info folder still exists
        assert dist_info.exists()
        assert dist_info.is_dir()
        assert (dist_info / "METADATA").exists()
        assert (dist_info / "RECORD").exists()

    def test_zip_handles_mixed_content_packages(self, tmp_path):
        """Should skip packages with native extensions entirely."""
        from build_distribution import zip_site_packages

        site_packages = tmp_path / "site-packages"
        site_packages.mkdir()

        # Create package with native extensions
        onnx_dir = site_packages / "onnxruntime"
        onnx_dir.mkdir()
        (onnx_dir / "__init__.pyc").write_bytes(b"PYC" * 100)
        (onnx_dir / "backend.pyc").write_bytes(b"PYC" * 100)

        # Native binary
        (onnx_dir / "_onnxruntime.pyd").write_bytes(b"PYD" * 100)

        # Subdirectory with DLLs
        capi_dir = onnx_dir / "capi"
        capi_dir.mkdir()
        (capi_dir / "__init__.pyc").write_bytes(b"PYC" * 100)
        (capi_dir / "onnxruntime.dll").write_bytes(b"DLL" * 1000)
        (capi_dir / "onnxruntime_providers_shared.dll").write_bytes(b"DLL" * 1000)

        # Create pure Python package (no native extensions)
        requests_dir = site_packages / "requests"
        requests_dir.mkdir()
        (requests_dir / "__init__.pyc").write_bytes(b"PYC" * 100)
        (requests_dir / "api.pyc").write_bytes(b"PYC" * 100)

        result = zip_site_packages(site_packages)

        assert result is True

        # Verify package with native extensions is NOT zipped
        onnx_zip = site_packages / "onnxruntime.zip"
        assert not onnx_zip.exists()

        # Verify all files in onnxruntime remain on filesystem
        assert onnx_dir.exists()
        assert (onnx_dir / "__init__.pyc").exists()
        assert (onnx_dir / "backend.pyc").exists()
        assert (onnx_dir / "_onnxruntime.pyd").exists()
        assert capi_dir.exists()
        assert (capi_dir / "__init__.pyc").exists()
        assert (capi_dir / "onnxruntime.dll").exists()
        assert (capi_dir / "onnxruntime_providers_shared.dll").exists()

        # Verify pure Python package WAS zipped
        requests_zip = site_packages / "requests.zip"
        assert requests_zip.exists()
        assert not requests_dir.exists()  # Directory removed after zipping
