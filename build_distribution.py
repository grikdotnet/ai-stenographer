"""
Windows distribution builder for STT application.

Creates portable Windows distribution using:
- Python embeddable package (with signed executables from system Python)
- Pre-compiled bytecode (.pyc files)
- Pre-installed dependencies
- Custom launcher with icon

Code Signing Strategy:
- Downloads embeddable Python package (unsigned executables)
- Replaces python.exe and pythonw.exe with properly signed versions from system Python
- This avoids Windows SmartScreen warnings on user machines
- System Python is signed by Python Software Foundation

Distribution structure:
AI-Stenographer/
├── AI - Stenographer.lnk
├── README.txt
├── LICENSE.txt
└── _internal/
    ├── runtime/        # Python executables (signed)
    ├── Lib/            # Dependencies
    ├── app/            # Application code (.pyc)
    │   ├── src/        # Source code with domain-based organization
    │   │   ├── sound/           # Audio processing (AudioSource, SoundPreProcessor)
    │   │   ├── asr/             # Speech recognition (Recognizer, AdaptiveWindower, VAD)
    │   │   ├── postprocessing/  # Text processing (TextMatcher, TextNormalizer)
    │   │   ├── gui/             # GUI components (TextDisplayWidget, TextFormatter)
    │   │   ├── controllers/     # MVC controllers (PauseController, InsertionController)
    │   │   └── quickentry/      # Quick entry feature (global hotkey, popup)
    │   ├── config/     # Configuration files
    │   └── assets/     # Static assets
    └── models/         # Downloaded at runtime
"""
import sys
import requests
import zipfile
import shutil
from pathlib import Path
from typing import Dict
import re
import subprocess

# Import LicenseCollector for automated license collection
from src.LicenseCollector import LicenseCollector


# Python version to download
PYTHON_VERSION = "3.12.10"
PYTHON_ARCH = "amd64"  # 64-bit Windows


def download_embedded_python(version: str, cache_dir: Path) -> Path:
    """
    Downloads Python embeddable package from python.org.

    Args:
        version: Python version string (e.g., "3.13.0")
        cache_dir: Directory to cache downloaded files

    Returns:
        Path to downloaded ZIP file

    Raises:
        ValueError: If version format is invalid
        requests.RequestException: If download fails
    """
    # Validate version format (X.Y.Z)
    if not re.match(r'^\d+\.\d+\.\d+$', version):
        raise ValueError(f"Invalid Python version format: {version}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Construct filename and check cache
    filename = f"python-{version}-embed-{PYTHON_ARCH}.zip"
    cached_file = cache_dir / filename

    if cached_file.exists():
        print(f"Using cached Python package: {cached_file}")
        return cached_file

    # Download from python.org
    # Format: https://www.python.org/ftp/python/3.13.0/python-3.13.0-embed-amd64.zip
    url = f"https://www.python.org/ftp/python/{version}/{filename}"

    print(f"Downloading Python {version} embeddable package...")
    print(f"URL: {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Write to cache
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(cached_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"Progress: {percent:.1f}%", end='\r')

    print(f"\nDownloaded: {cached_file}")
    return cached_file


def extract_embedded_python(zip_path: Path, target_dir: Path) -> None:
    """
    Extracts Python embeddable package to target directory.

    Args:
        zip_path: Path to Python embeddable ZIP file
        target_dir: Directory to extract files to

    Raises:
        FileNotFoundError: If zip_path doesn't exist
        zipfile.BadZipFile: If ZIP file is corrupted
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Python package not found: {zip_path}")

    print(f"Extracting Python to {target_dir}...")

    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)

    print(f"Extracted {len(list(target_dir.iterdir()))} files")


def create_directory_structure(root_dir: Path) -> Dict[str, Path]:
    """
    Creates _internal directory structure for distribution.

    Directory structure:
    root/
    ├── _internal/
    │   ├── runtime/
    │   ├── Lib/site-packages/
    │   ├── app/
    │   │   ├── src/
    │   │   ├── config/
    │   │   └── assets/
    │   └── models/

    Args:
        root_dir: Root distribution directory (e.g., AI-Stenographer/)

    Returns:
        Dictionary mapping location names to Path objects
    """
    print(f"Creating directory structure in {root_dir}...")

    # Create directories
    internal_dir = root_dir / "_internal"
    runtime_dir = internal_dir / "runtime"
    lib_dir = internal_dir / "Lib" / "site-packages"
    app_dir = internal_dir / "app"
    app_src_dir = app_dir / "src"
    app_config_dir = app_dir / "config"
    app_assets_dir = app_dir / "assets"
    models_dir = internal_dir / "models"

    # Create all directories
    for directory in [runtime_dir, lib_dir, app_src_dir, app_config_dir,
                      app_assets_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "root": root_dir,
        "internal": internal_dir,
        "runtime": runtime_dir,
        "lib": lib_dir,
        "app": app_dir,
        "app_src": app_src_dir,
        "app_config": app_config_dir,
        "app_assets": app_assets_dir,
        "models": models_dir
    }

    print(f"Created {len(paths)} directory locations")
    return paths


def verify_file_signature(file_path: Path) -> bool:
    """
    Verifies digital signature of Windows executable.

    Uses PowerShell Get-AuthenticodeSignature cmdlet.

    Args:
        file_path: Path to executable file

    Returns:
        True if file is signed and signature is valid
    """
    if not file_path.exists():
        return False

    try:
        # Use PowerShell to verify signature
        cmd = [
            "powershell",
            "-Command",
            f"(Get-AuthenticodeSignature '{file_path}').Status -eq 'Valid'"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        # PowerShell returns "True" or "False" as text
        is_valid = result.stdout.strip().lower() == "true"

        # Debug output for troubleshooting
        if not is_valid:
            # Get actual signature status for debugging
            status_cmd = [
                "powershell",
                "-Command",
                f"(Get-AuthenticodeSignature '{file_path}').Status"
            ]
            status_result = subprocess.run(status_cmd, capture_output=True, text=True, timeout=10)
            actual_status = status_result.stdout.strip()
            print(f"  [DEBUG] {file_path.name}: Expected 'Valid', got '{actual_status}'")

        return is_valid

    except Exception as e:
        print(f"Warning: Could not verify signature for {file_path}: {e}")
        return False


def copy_signed_executables_from_system(runtime_dir: Path) -> bool:
    """
    Replaces unsigned embedded Python executables with signed system Python executables.

    The embeddable Python package from python.org contains unsigned executables.
    This function copies properly signed python.exe and pythonw.exe from the
    system Python installation to avoid Windows SmartScreen warnings.

    Strategy:
    - Keep all files from embeddable package (python312.dll, stdlib, etc.)
    - Replace only python.exe and pythonw.exe with signed versions
    - Verify both source and destination signatures

    Args:
        runtime_dir: Directory containing Python runtime files

    Returns:
        True if signed executables were successfully copied and verified
    """
    print("Replacing unsigned executables with signed versions from system Python...")

    try:
        # Locate system Python (use sys.base_prefix to get actual installation, not venv)
        system_python = Path(sys.base_prefix)

        # Check version compatibility (must match minor version)
        system_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        target_version = PYTHON_VERSION.rsplit('.', 1)[0]  # e.g., "3.12.8" -> "3.12"

        if system_version != target_version:
            print(f"  [SKIP] System Python {system_version} does not match target {target_version}")
            print(f"  Continuing with unsigned executables from embeddable package")
            return False

        # Files to replace
        executables = ["python.exe", "pythonw.exe"]

        for exe_name in executables:
            src_exe = system_python / exe_name
            dest_exe = runtime_dir / exe_name

            # Check source exists
            if not src_exe.exists():
                print(f"  [ERROR] System Python {exe_name} not found: {src_exe}")
                return False

            # Verify source is signed
            if not verify_file_signature(src_exe):
                print(f"  [WARNING] System Python {exe_name} is not signed")
                print(f"  Continuing anyway, but SmartScreen warnings may still occur")

            # Check destination exists (from embeddable package)
            if not dest_exe.exists():
                print(f"  [ERROR] Embedded {exe_name} not found: {dest_exe}")
                return False

            # Replace unsigned executable with signed version
            shutil.copy2(src_exe, dest_exe)
            print(f"  Copied signed {exe_name} from system Python")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to copy signed executables: {e}")
        return False


def verify_signatures(runtime_dir: Path) -> bool:
    """
    Verifies digital signatures of Python executables.

    Checks:
    - python.exe
    - pythonw.exe
    - python3XX.dll

    Args:
        runtime_dir: Directory containing Python runtime files

    Returns:
        True if all files exist and are properly signed
    """
    print("Verifying digital signatures of Python executables...")

    files_to_check = [
        runtime_dir / "python.exe",
        runtime_dir / "pythonw.exe"
    ]

    # Check if files exist
    for file_path in files_to_check:
        if not file_path.exists():
            print(f"Error: {file_path.name} not found")
            return False

    # Verify signatures (Windows-specific)
    if sys.platform == "win32":
        for file_path in files_to_check:
            if verify_file_signature(file_path):
                print(f"[OK] {file_path.name} - Valid signature")
            else:
                print(f"[FAIL] {file_path.name} - No signature or invalid")
                return False
    else:
        print("Warning: Signature verification skipped (not on Windows)")

    return True


def main():
    """Main build script entry point."""
    print("=" * 60)
    print("AI Distribution Builder")
    print("=" * 60)

    # Setup paths
    project_root = Path(__file__).parent
    cache_dir = project_root / ".cache"
    dist_dir = project_root / "dist"
    build_dir = dist_dir / "AI-Stenographer"

    # Clean previous build
    if build_dir.exists():
        print(f"Cleaning previous build: {build_dir}")

        def handle_remove_readonly(func, path, exc):
            """Error handler for Windows readonly/locked files."""
            import stat
            import os

            # Try to change permissions and retry
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except:
                # If still fails, skip this file
                print(f"  [SKIP] Cannot remove: {Path(path).name}")

        try:
            shutil.rmtree(build_dir, onerror=handle_remove_readonly)
        except Exception as e:
            print(f"  Warning: Cleanup had errors: {e}")
            print(f"  Continuing anyway...")

    # Step 1: Download Python embeddable package
    try:
        zip_path = download_embedded_python(PYTHON_VERSION, cache_dir)
    except Exception as e:
        print(f"\nError downloading Python: {e}")
        return 1

    # Step 2: Create directory structure
    try:
        paths = create_directory_structure(build_dir)
    except Exception as e:
        print(f"\nError creating directories: {e}")
        return 1

    # Step 3: Extract Python to runtime directory
    try:
        extract_embedded_python(zip_path, paths["runtime"])
    except Exception as e:
        print(f"\nError extracting Python: {e}")
        return 1

    # Step 4: Replace unsigned executables with signed versions from system Python
    if not copy_signed_executables_from_system(paths["runtime"]):
        print("\nWarning: Could not copy signed executables from system Python")
        print("Continuing with unsigned executables from embeddable package")
        print("This may cause SmartScreen warnings on user machines")

    # Step 5: Verify signatures (should now pass with signed executables)
    if not verify_signatures(paths["runtime"]):
        print("\nWarning: Python executables are not properly signed")
        print("This may cause SmartScreen warnings on user machines")

    # Step 6: Copy tkinter module from system Python
    if not copy_tkinter_to_distribution(paths["runtime"]):
        print("\nWarning: Failed to copy tkinter module")
        print("Tkinter will not be available in the distribution")
        print("Continuing with build...")

    # Step 6a: Remove unnecessary Tcl/Tk files
    if not cleanup_tcl_unnecessary_files(paths["runtime"]):
        print("\nWarning: Failed to cleanup Tcl/Tk files")
        print("Build will continue, but distribution size will be larger")

    # Step 7: Create python312._pth configuration
    # Paths are relative to python.exe location (_internal/runtime/)
    try:
        pth_paths = [
            "python312.zip",            # In same dir as python.exe
            "Lib",                      # For tkinter module (in runtime/Lib/)
            "../Lib/site-packages",     # Up one, then to Lib/site-packages
            "../app",                   # Up one, then to app
            "import site"
        ]
        create_pth_file(paths["runtime"], pth_paths)
    except Exception as e:
        print(f"\nError creating _pth file: {e}")
        return 1

    # Step 8: Enable pip in embedded Python
    try:
        enable_pip(paths["runtime"])
    except Exception as e:
        print(f"\nError enabling pip: {e}")
        return 1

    # Step 9: Verify pip is available
    python_exe = paths["runtime"] / "python.exe"
    if not verify_pip(python_exe):
        print("\nWarning: Pip verification failed")
        print("Dependency installation may not work")

    # Step 10: Verify tkinter is importable
    if not verify_tkinter(python_exe):
        print("\nWarning: Tkinter verification failed")
        print("GUI functionality may not work")

    # Step 11: Collect third-party licenses
    if not collect_third_party_licenses(project_root):
        print("\nError: License collection failed")
        print("LICENSES/ folder and THIRD_PARTY_NOTICES.txt are required for distribution")
        return 1

    # Step 12: Copy legal documents
    try:
        copy_legal_documents(project_root, build_dir)
    except Exception as e:
        print(f"\nError copying legal documents: {e}")
        return 1

    # Step 13: Install dependencies
    requirements_file = project_root / "requirements.txt"
    if not install_dependencies(python_exe, paths["lib"], requirements_file):
        print("\nError: Dependency installation failed")
        return 1

    # Step 13a: Remove pip from distribution (not needed at runtime)
    if not remove_pip_from_distribution(paths["runtime"], paths["lib"]):
        print("\nWarning: Failed to remove pip from distribution")
        print("Build will continue, but distribution size will be larger")

    # Step 13b: Remove test directories from third-party packages
    if not remove_tests_from_distribution(paths["lib"]):
        print("\nWarning: Failed to remove test directories from distribution")
        print("Build will continue, but distribution size will be larger")

    # Step 14: Verify native libraries
    lib_results = verify_native_libraries(paths["lib"])
    if not all(lib_results.values()):
        print("\nWarning: Some native libraries missing")
        print("Application may not work correctly")

    # Step 15: Test critical imports
    critical_modules = ["numpy", "onnxruntime", "sounddevice", "onnx_asr", "tkinter", "pynput"]
    import_results = test_imports(python_exe, critical_modules)
    failed_imports = [m for m, success in import_results.items() if not success]

    if failed_imports:
        print(f"\nWarning: Failed to import: {', '.join(failed_imports)}")
        print("Application may not work correctly")

    # Step 16: Copy application code
    src_dir = project_root / "src"
    main_py = project_root / "main.py"
    if not copy_application_code(src_dir, main_py, paths["app"]):
        print("\nError: Failed to copy application code")
        return 1

    # Step 17: Compile to bytecode
    if not compile_to_pyc(python_exe, paths["app"]):
        print("\nError: Failed to compile code to bytecode")
        return 1

    # Step 18: Compile third-party packages to bytecode
    if not compile_site_packages(python_exe, paths["lib"]):
        print("\nError: Failed to compile third-party packages")
        return 1

    # Step 19: Package .pyc files into .zip archives (reduce filesystem overhead)
    if not zip_site_packages(paths["lib"]):
        print("\nError: Failed to package .pyc files into .zip archives")
        return 1

    # Step 20: Clean up redundant package metadata
    if not cleanup_package_metadata(paths["lib"]):
        print("\nError: Failed to clean package metadata")
        return 1

    # Step 21: Copy bundled Silero VAD model
    if not copy_bundled_silero_vad(project_root, paths["models"]):
        print("\nWarning: Failed to copy bundled Silero VAD model")
        print("Silero VAD will need to be downloaded on first run")

    # Step 22: Copy assets and configuration
    if not copy_assets_and_config(project_root, build_dir, paths["app"]):
        print("\nError: Failed to copy assets and configuration")
        return 1

    # Step 23: Create README documentation
    if not create_readme(build_dir):
        print("\nError: Failed to create README")
        return 1

    # Step 24: Create launcher shortcut
    if not create_launcher(build_dir):
        print("\nError: Failed to create launcher shortcut")
        return 1

    print("\n" + "=" * 60)
    print("Build completed successfully!")
    print(f"Build directory: {build_dir}")
    print(f"\nTo run the application:")
    print(f'  Double-click: "{build_dir}\\AI - Stenographer.lnk"')
    print("=" * 60)

    return 0


def create_pth_file(runtime_dir: Path, paths: list[str]) -> None:
    """
    Creates python312._pth file to configure module search paths.

    The _pth file tells embedded Python where to find modules without
    needing environment variables or registry settings.

    Args:
        runtime_dir: Directory containing Python runtime
        paths: List of paths to add to sys.path (relative to exe location)

    Example paths:
        _internal/runtime/python312.zip
        _internal/Lib/site-packages
        _internal/app
        import site
    """
    print("Creating python312._pth configuration...")

    pth_file = runtime_dir / "python312._pth"

    content = "\n".join(paths) + "\n"
    pth_file.write_text(content, encoding='utf-8')

    print(f"Created {pth_file}")
    print(f"Configured {len(paths)} module search paths")


def enable_pip(runtime_dir: Path) -> None:
    """
    Enables pip in embedded Python by downloading and running get-pip.py.

    Embedded Python doesn't include pip by default. This downloads the
    official get-pip.py installer and runs it.

    Args:
        runtime_dir: Directory containing Python runtime

    Raises:
        requests.RequestException: If download fails
        subprocess.CalledProcessError: If pip installation fails
    """
    print("Enabling pip in embedded Python...")

    python_exe = runtime_dir / "python.exe"
    get_pip_script = runtime_dir / "get-pip.py"

    # Download get-pip.py
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    print(f"Downloading get-pip.py from {get_pip_url}...")

    response = requests.get(get_pip_url)
    response.raise_for_status()

    get_pip_script.write_bytes(response.content)
    print(f"Downloaded get-pip.py ({len(response.content)} bytes)")

    # Run get-pip.py
    print(f"Running get-pip.py with {python_exe.name}...")
    result = subprocess.run(
        [str(python_exe), str(get_pip_script)],
        capture_output=True,
        text=True,
        cwd=runtime_dir
    )

    if result.returncode != 0:
        print(f"Error installing pip:\n{result.stderr}")
        raise RuntimeError(f"Pip installation failed with code {result.returncode}")

    print("Pip installed successfully")

    # Clean up get-pip.py
    get_pip_script.unlink()


def verify_pip(python_exe: Path) -> bool:
    """
    Verifies that pip is available in Python installation.

    Args:
        python_exe: Path to python.exe

    Returns:
        True if pip is available and working
    """
    print(f"Verifying pip is available...")

    try:
        result = subprocess.run(
            [str(python_exe), "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print(f"Pip version: {result.stdout.strip()}")
            return True
        else:
            print(f"Pip not available (exit code {result.returncode})")
            return False

    except Exception as e:
        print(f"Error verifying pip: {e}")
        return False


def copy_tkinter_to_distribution(runtime_dir: Path, allow_fallback: bool = True) -> bool:
    """
    Copies tkinter module and Tcl/Tk libraries from system Python.

    Python embeddable package includes Tcl/Tk DLLs but NOT:
    - tkinter module itself
    - tcl/tk library files (_tkinter.pyd depends on these)

    This copies from system Python to make GUI functionality available.

    Copies:
    - Lib/tkinter/ directory → runtime/Lib/tkinter/
    - DLLs/_tkinter.pyd → runtime/_tkinter.pyd
    - tcl/ directory → runtime/tcl/ (Tcl/Tk library files)
    - DLLs/zlib1.dll → runtime/zlib1.dll (required for Python 3.12+)

    Args:
        runtime_dir: Path to embedded Python runtime directory
        allow_fallback: If True, attempt to find matching Python installation when
                       version mismatch occurs. Set to False in tests to prevent
                       finding real system Python installations.

    Returns:
        True if tkinter was successfully copied, False otherwise
    """
    print("Copying tkinter module and Tcl/Tk libraries from system Python...")

    try:
        # Locate system Python (use sys.base_prefix to get actual installation, not venv)
        system_python = Path(sys.base_prefix)

        # Check version compatibility (must match minor version)
        system_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        target_version = PYTHON_VERSION.rsplit('.', 1)[0]  # e.g., "3.12.10" -> "3.12"

        if system_version != target_version:
            if allow_fallback:
                # Try to find matching Python installation
                target_major_minor = target_version.replace('.', '')  # "3.12" -> "312"
                python_install_path = Path(f"C:/Python{target_major_minor}")

                if python_install_path.exists():
                    print(f"  [INFO] System Python {system_version} != target {target_version}")
                    print(f"  [INFO] Using {python_install_path} for tkinter extraction")
                    system_python = python_install_path
                else:
                    print(f"  [WARNING] System Python {system_version} does not match target {target_version}")
                    print(f"  [WARNING] C:/Python{target_major_minor} not found")
                    print(f"  [ERROR] Cannot extract tkinter - version mismatch")
                    return False
            else:
                # Fallback disabled (for testing)
                print(f"  [ERROR] System Python {system_version} does not match target {target_version}")
                return False

        # 1. Copy tkinter module
        system_tkinter = system_python / "Lib" / "tkinter"
        if not system_tkinter.exists():
            print(f"  [ERROR] tkinter not found: {system_tkinter}")
            print(f"  System Python must have tkinter installed")
            return False

        dest_lib = runtime_dir / "Lib"
        dest_lib.mkdir(parents=True, exist_ok=True)

        dest_tkinter = dest_lib / "tkinter"
        if dest_tkinter.exists():
            shutil.rmtree(dest_tkinter)

        shutil.copytree(system_tkinter, dest_tkinter)
        file_count = sum(1 for _ in dest_tkinter.rglob("*") if _.is_file())
        print(f"  Copied tkinter module ({file_count} files)")

        # 2. Copy _tkinter.pyd binary
        system_tkinter_pyd = system_python / "DLLs" / "_tkinter.pyd"
        if not system_tkinter_pyd.exists():
            print(f"  [ERROR] _tkinter.pyd not found: {system_tkinter_pyd}")
            return False

        dest_pyd = runtime_dir / "_tkinter.pyd"
        shutil.copy2(system_tkinter_pyd, dest_pyd)
        print(f"  Copied _tkinter.pyd ({dest_pyd.stat().st_size} bytes)")

        # 3. Copy tcl/ directory (Tcl/Tk library files required by _tkinter.pyd)
        system_tcl = system_python / "tcl"
        if system_tcl.exists():
            dest_tcl = runtime_dir / "tcl"
            if dest_tcl.exists():
                shutil.rmtree(dest_tcl)

            shutil.copytree(system_tcl, dest_tcl)
            tcl_size_mb = sum(f.stat().st_size for f in dest_tcl.rglob("*") if f.is_file()) / (1024 * 1024)
            print(f"  Copied tcl/ directory ({tcl_size_mb:.1f}MB)")
        else:
            print(f"  [WARNING] tcl/ directory not found: {system_tcl}")
            print(f"  Tkinter may not work without Tcl/Tk library files")

        # 4. Copy Tcl/Tk DLLs (tcl86t.dll, tk86t.dll, zlib1.dll)
        # Python 3.12+ requires zlib1.dll for _tkinter.pyd to load
        # Some Python embedded versions don't include these
        system_dlls = system_python / "DLLs"
        required_dlls = ["tcl86t.dll", "tk86t.dll", "zlib1.dll"]
        copied_dlls = []

        for dll_name in required_dlls:
            dll_src = system_dlls / dll_name
            if dll_src.exists():
                dll_dest = runtime_dir / dll_name
                if not dll_dest.exists():  # Don't overwrite if already present
                    shutil.copy2(dll_src, dll_dest)
                    copied_dlls.append(dll_name)
            elif dll_name == "zlib1.dll":
                # zlib1.dll is CRITICAL for _tkinter.pyd in Python 3.12+
                print(f"  [ERROR] {dll_name} not found: {dll_src}")
                print(f"  _tkinter.pyd will fail to load without zlib1.dll")
                return False

        if copied_dlls:
            print(f"  Copied required DLLs: {', '.join(copied_dlls)}")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to copy tkinter: {e}")
        return False


def cleanup_tcl_unnecessary_files(runtime_dir: Path) -> bool:
    """
    Removes unnecessary Tcl/Tk files from distribution.

    The application uses only basic tkinter widgets (Label, Text, Button)
    which don't require timezone data or advanced Tcl libraries.

    Removes:
    - tcl/tcl8.6/tzdata/ (~600 files, 2.1MB) - Timezone database not needed
    - tcl/tcl8.6/msgs/ - Localization messages (app is English-only)
    - tcl/tcl8/8.6/encoding/ (partial) - Keep only utf-8, unicode, and common encodings

    Preserves:
    - Core Tcl/Tk init scripts (required for tkinter to work)
    - Basic widgets and dialogs

    Args:
        runtime_dir: Path to embedded Python runtime directory

    Returns:
        True if cleanup succeeded, False otherwise
    """
    print("Removing unnecessary Tcl/Tk files...")

    try:
        tcl_dir = runtime_dir / "tcl"
        if not tcl_dir.exists():
            print("  [SKIP] No tcl/ directory found")
            return True

        removed_items = []
        total_size = 0

        # Remove timezone data (not used by basic tkinter widgets)
        tzdata_dir = tcl_dir / "tcl8.6" / "tzdata"
        if tzdata_dir.exists():
            tzdata_size = sum(f.stat().st_size for f in tzdata_dir.rglob("*") if f.is_file())
            file_count = sum(1 for _ in tzdata_dir.rglob("*") if _.is_file())
            total_size += tzdata_size
            shutil.rmtree(tzdata_dir)
            removed_items.append(f"tzdata/ ({file_count} files)")

        # Remove localization messages (app is English-only)
        msgs_dir = tcl_dir / "tcl8.6" / "msgs"
        if msgs_dir.exists():
            msgs_size = sum(f.stat().st_size for f in msgs_dir.rglob("*") if f.is_file())
            file_count = sum(1 for _ in msgs_dir.rglob("*") if _.is_file())
            total_size += msgs_size
            shutil.rmtree(msgs_dir)
            removed_items.append(f"msgs/ ({file_count} files)")

        if removed_items:
            size_mb = total_size / (1024 * 1024)
            print(f"  Removed: {', '.join(removed_items)}")
            print(f"  Space saved: {size_mb:.1f}MB")
        else:
            print("  Nothing to remove (already clean)")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to cleanup Tcl/Tk files: {e}")
        return False


def verify_tkinter(python_exe: Path) -> bool:
    """
    Verifies that tkinter module can be imported.

    Args:
        python_exe: Path to python.exe

    Returns:
        True if tkinter imports successfully
    """
    print("Verifying tkinter is importable...")

    try:
        result = subprocess.run(
            [str(python_exe), "-c", "import tkinter"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print(f"  [OK] tkinter imports successfully")
            return True
        else:
            print(f"  [FAIL] tkinter import failed:")
            print(f"  {result.stderr.strip()}")
            return False

    except Exception as e:
        print(f"  [ERROR] Error verifying tkinter: {e}")
        return False

def collect_third_party_licenses(project_root: Path) -> bool:
    """
    Collects third-party license files for distribution.

    Uses LicenseCollector to automatically gather licenses from all runtime
    dependencies and create THIRD_PARTY_NOTICES.txt.

    This ensures licenses are always up-to-date with installed dependencies.

    Args:
        project_root: Project root directory

    Returns:
        True if license collection succeeded, False otherwise
    """
    print("Collecting third-party licenses...")

    try:
        # Create LicenseCollector with default output directory
        collector = LicenseCollector(output_dir=str(project_root / "LICENSES"))

        # Run the collection process
        collector.collect_all_licenses()
        collector.create_python_license_entry()
        collector.generate_notices_file()
        collector.generate_readme()

        print(f"  Collected {len(collector.collected_licenses)} licenses")
        return True

    except Exception as e:
        print(f"  [ERROR] License collection failed: {e}")
        return False


def copy_legal_documents(project_root: Path, build_dir: Path) -> None:
    """
    Copies legal documents to distribution root.

    Copies:
    - LICENSE.txt (main license)
    - EULA.txt (end user license agreement)
    - THIRD_PARTY_NOTICES.txt (attribution)
    - LICENSES/ folder (third-party licenses)

    Args:
        project_root: Project root directory
        build_dir: Build root directory (AI-Stenographer/)
    """
    print("Copying legal documents...")

    # Copy legal text files to root
    legal_files = ["LICENSE.txt", "EULA.txt", "THIRD_PARTY_NOTICES.txt"]
    copied = 0

    for legal_file in legal_files:
        src = project_root / legal_file
        if src.exists():
            shutil.copy2(src, build_dir / legal_file)
            print(f"  Copied {legal_file}")
            copied += 1
        else:
            print(f"  Warning: {legal_file} not found")

    # Copy LICENSES/ folder
    licenses_src = project_root / "LICENSES"
    licenses_dest = build_dir / "LICENSES"

    if licenses_src.exists():
        if licenses_dest.exists():
            shutil.rmtree(licenses_dest)
        shutil.copytree(licenses_src, licenses_dest)
        license_count = len(list(licenses_dest.glob("*/")))
        print(f"  Copied LICENSES/ folder ({license_count} packages)")
    else:
        print(f"  Warning: LICENSES/ folder not found")

    print(f"Legal documents: {copied} files + LICENSES/ folder")


def install_dependencies(python_exe: Path, target_dir: Path, requirements: Path) -> bool:
    """
    Installs Python dependencies from requirements.txt.

    Uses pip to install packages into target directory (site-packages).

    Args:
        python_exe: Path to python.exe in embedded Python
        target_dir: Target directory for packages (Lib/site-packages)
        requirements: Path to requirements.txt

    Returns:
        True if installation succeeded, False otherwise
    """
    print(f"Installing dependencies from {requirements.name}...")

    if not requirements.exists():
        print(f"Error: {requirements} not found")
        return False

    target_dir.mkdir(parents=True, exist_ok=True)

    # Run pip install with --target flag
    cmd = [
        str(python_exe),
        "-m", "pip", "install",
        "-r", str(requirements),
        "--target", str(target_dir),
        "--no-warn-script-location",
        "--disable-pip-version-check"
    ]

    print(f"Running: pip install -r {requirements.name} --target {target_dir.name}/")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=requirements.parent
    )

    if result.returncode == 0:
        print("Dependencies installed successfully")
        # Show installed packages summary
        if "Successfully installed" in result.stdout:
            print(result.stdout.split("Successfully installed")[1].strip())
        return True
    else:
        print(f"Error installing dependencies:")
        print(result.stderr)
        return False


def remove_tests_from_distribution(site_packages_dir: Path) -> bool:
    """
    Removes test directories from third-party packages.

    Many Python packages include their unit tests when installed, which are
    completely unnecessary for a frozen distribution. This removes common
    test directory patterns to save significant space (~30MB+).

    Removes:
    - */tests/ directories (most common pattern)
    - */test/ directories (alternate pattern)
    - */testing/ directories (used by some packages)

    Args:
        site_packages_dir: Directory containing third-party packages (_internal/Lib/site-packages/)

    Returns:
        True if successful, False on error
    """
    print("Removing test directories from third-party packages...")

    try:
        removed_count = 0
        total_size = 0
        test_patterns = ["tests", "test", "testing"]

        for pattern in test_patterns:
            # Find all directories matching the test pattern
            # Use rglob to find them recursively in packages
            test_dirs = list(site_packages_dir.rglob(pattern))

            for test_dir in test_dirs:
                # Only remove if it's actually a directory
                if not test_dir.is_dir():
                    continue

                # Calculate size before removal
                try:
                    dir_size = sum(f.stat().st_size for f in test_dir.rglob("*") if f.is_file())
                    total_size += dir_size

                    # Remove the directory
                    shutil.rmtree(test_dir)
                    removed_count += 1
                except Exception as e:
                    # Skip if we can't remove (permission issues, etc.)
                    print(f"  [SKIP] Could not remove {test_dir.relative_to(site_packages_dir)}: {e}")

        if removed_count > 0:
            size_mb = total_size / (1024 * 1024)
            print(f"  Removed {removed_count} test directories")
            print(f"  Space saved: {size_mb:.1f}MB")
        else:
            print("  No test directories found (already clean)")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to remove tests: {e}")
        return False


def remove_pip_from_distribution(runtime_dir: Path, site_packages_dir: Path) -> bool:
    """
    Removes pip from distribution after dependency installation.

    Pip is only needed during build to install dependencies. At runtime, the
    application never uses pip. Removing it saves significant space (857 files).

    Pip is installed in two locations by get-pip.py:
    1. runtime/Lib/site-packages/pip/ - the pip package
    2. runtime/Scripts/ - pip executables (pip.exe, pip3.exe, etc.)

    This function removes pip from BOTH locations.

    Args:
        runtime_dir: Directory containing Python runtime (_internal/runtime/)
        site_packages_dir: Directory containing third-party packages (_internal/Lib/site-packages/)

    Returns:
        True if successful (even if pip wasn't found), False on error
    """
    print("Removing pip from distribution (not needed at runtime)...")

    try:
        removed_items = []
        total_size = 0

        # Location 1: Remove pip from runtime/Lib/site-packages/
        runtime_site_packages = runtime_dir / "Lib" / "site-packages"
        if runtime_site_packages.exists():
            # Remove pip package directory
            pip_dir = runtime_site_packages / "pip"
            if pip_dir.exists() and pip_dir.is_dir():
                pip_size = sum(f.stat().st_size for f in pip_dir.rglob("*") if f.is_file())
                total_size += pip_size
                file_count = sum(1 for _ in pip_dir.rglob("*") if _.is_file())
                shutil.rmtree(pip_dir)
                removed_items.append(f"runtime/pip/ ({file_count} files)")

            # Remove pip metadata folders (pip-*.dist-info)
            pip_dist_info = list(runtime_site_packages.glob("pip-*.dist-info"))
            for dist_info in pip_dist_info:
                if dist_info.is_dir():
                    dist_info_size = sum(f.stat().st_size for f in dist_info.rglob("*") if f.is_file())
                    total_size += dist_info_size
                    shutil.rmtree(dist_info)
                    removed_items.append(f"runtime/{dist_info.name}/")

        # Location 2: Remove pip executables from runtime/Scripts/
        scripts_dir = runtime_dir / "Scripts"
        if scripts_dir.exists():
            pip_exes = list(scripts_dir.glob("pip*.exe"))
            for pip_exe in pip_exes:
                if pip_exe.is_file():
                    exe_size = pip_exe.stat().st_size
                    total_size += exe_size
                    pip_exe.unlink()
                    removed_items.append(f"Scripts/{pip_exe.name}")

        # Location 3: Remove pip from target site-packages (if it exists there)
        # This shouldn't normally happen, but check just in case
        pip_dir_target = site_packages_dir / "pip"
        if pip_dir_target.exists() and pip_dir_target.is_dir():
            pip_size = sum(f.stat().st_size for f in pip_dir_target.rglob("*") if f.is_file())
            total_size += pip_size
            file_count = sum(1 for _ in pip_dir_target.rglob("*") if _.is_file())
            shutil.rmtree(pip_dir_target)
            removed_items.append(f"target/pip/ ({file_count} files)")

        pip_dist_info_target = list(site_packages_dir.glob("pip-*.dist-info"))
        for dist_info in pip_dist_info_target:
            if dist_info.is_dir():
                dist_info_size = sum(f.stat().st_size for f in dist_info.rglob("*") if f.is_file())
                total_size += dist_info_size
                shutil.rmtree(dist_info)
                removed_items.append(f"target/{dist_info.name}/")

        if removed_items:
            size_mb = total_size / (1024 * 1024)
            print(f"  Removed: {', '.join(removed_items)}")
            print(f"  Space saved: {size_mb:.1f}MB")
        else:
            print("  Pip not found (already removed or not installed)")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to remove pip: {e}")
        return False


def verify_native_libraries(site_packages: Path) -> dict[str, bool]:
    """
    Verifies that native libraries (DLLs) are present.

    Checks for critical DLLs:
    - ONNX Runtime: onnxruntime.dll, onnxruntime_providers_shared.dll, DirectML.dll
    - Sounddevice: portaudio DLL

    Args:
        site_packages: Path to site-packages directory

    Returns:
        Dictionary mapping library name to presence boolean
    """
    print("Verifying native libraries...")

    results = {}

    # Check ONNX Runtime DLLs
    onnx_capi = site_packages / "onnxruntime" / "capi"
    onnx_dll = onnx_capi / "onnxruntime.dll"
    onnx_providers_dll = onnx_capi / "onnxruntime_providers_shared.dll"
    directml_dll = onnx_capi / "DirectML.dll"

    results["onnxruntime"] = onnx_dll.exists() and onnx_providers_dll.exists()
    results["directml"] = directml_dll.exists()

    if results["onnxruntime"]:
        print(f"  [OK] ONNX Runtime DLLs found")
    else:
        print(f"  [MISSING] ONNX Runtime DLLs not found")

    if results["directml"]:
        print(f"  [OK] DirectML.dll found (GPU acceleration available)")
    else:
        print(f"  [WARNING] DirectML.dll not found (GPU acceleration unavailable)")
    
    # Check sounddevice/PortAudio DLL
    sd_data = site_packages / "_sounddevice_data"
    portaudio_dll = None

    if sd_data.exists():
        # Check for various PortAudio DLL names
        # sounddevice-0.5.1+ uses portaudio-binaries subdirectory
        dll_patterns = [
            "portaudio_x64.dll",
            "portaudio.dll",
            "_portaudio.pyd",
            "portaudio-binaries/libportaudio64bit.dll",
            "portaudio-binaries/libportaudio64bit-asio.dll"
        ]
        for dll_name in dll_patterns:
            dll_path = sd_data / dll_name
            if dll_path.exists():
                portaudio_dll = dll_path
                break
    
    results["sounddevice"] = portaudio_dll is not None
    
    if results["sounddevice"]:
        print(f"  [OK] PortAudio DLL found: {portaudio_dll.name}")
    else:
        print(f"  [MISSING] PortAudio DLL not found in _sounddevice_data/")
    
    return results


def test_imports(python_exe: Path, modules: list[str]) -> dict[str, bool]:
    """
    Tests that Python modules can be imported.
    
    Runs python -c "import <module>" for each module to verify
    they are correctly installed and importable.
    
    Args:
        python_exe: Path to python.exe
        modules: List of module names to test
    
    Returns:
        Dictionary mapping module name to import success boolean
    """
    print(f"Testing imports for {len(modules)} modules...")
    
    results = {}
    
    for module in modules:
        cmd = [str(python_exe), "-c", f"import {module}"]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        results[module] = result.returncode == 0
        
        if results[module]:
            print(f"  [OK] {module}")
        else:
            print(f"  [FAIL] {module}: {result.stderr.strip()}")
    
    return results


def copy_application_code(src_dir: Path, main_py: Path, app_dir: Path) -> bool:
    """
    Copies application code to distribution app directory.

    Copies:
    - src/ directory with all subdirectories and files
    - main.py file

    Args:
        src_dir: Source directory containing application code
        main_py: Path to main.py file
        app_dir: Target app directory in distribution

    Returns:
        True if successful, False otherwise
    """
    print("Copying application code...")

    try:
        if not src_dir.exists():
            print(f"  [ERROR] Source directory not found: {src_dir}")
            return False

        if not main_py.exists():
            print(f"  [ERROR] main.py not found: {main_py}")
            return False

        target_src = app_dir / "src"
        if target_src.exists():
            shutil.rmtree(target_src)

        shutil.copytree(src_dir, target_src)
        print(f"  Copied src/ ({sum(1 for _ in src_dir.rglob('*.py'))} Python files)")

        target_main = app_dir / "main.py"
        shutil.copy2(main_py, target_main)
        print(f"  Copied main.py")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to copy code: {e}")
        return False


def copy_bundled_silero_vad(project_root: Path, models_dir: Path) -> bool:
    """
    Copies bundled Silero VAD model to distribution.

    The Silero VAD model is bundled with the repository to avoid downloading
    it on first run. This copies it from the project's models/ directory to
    the distribution's _internal/models/ directory.

    Args:
        project_root: Project root directory
        models_dir: Target models directory in distribution (_internal/models/)

    Returns:
        True if successful, False otherwise
    """
    print("Copying bundled Silero VAD model...")

    try:
        silero_src = project_root / "models" / "silero_vad"
        silero_dest = models_dir / "silero_vad"

        if not silero_src.exists():
            print(f"  [WARNING] Silero VAD not found: {silero_src}")
            print(f"  Silero VAD will need to be downloaded on first run")
            return False

        # Check that the model file exists
        silero_model = silero_src / "silero_vad.onnx"
        if not silero_model.exists():
            print(f"  [WARNING] silero_vad.onnx not found: {silero_model}")
            return False

        # Copy the entire silero_vad directory
        if silero_dest.exists():
            shutil.rmtree(silero_dest)

        shutil.copytree(silero_src, silero_dest)
        model_size_mb = silero_model.stat().st_size / (1024 * 1024)
        print(f"  Copied Silero VAD model ({model_size_mb:.1f}MB)")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to copy Silero VAD: {e}")
        return False


def copy_assets_and_config(project_root: Path, build_dir: Path, app_dir: Path) -> bool:
    """
    Copies assets and configuration files to distribution.

    Copies:
    - config/ directory to _internal/app/config/

    Args:
        project_root: Project root directory
        build_dir: Build root directory (AI-Stenographer/)
        app_dir: App directory (_internal/app/)

    Returns:
        True if successful (even if some files are missing)
    """
    print("Copying assets and configuration...")

    try:
        config_src = project_root / "config"
        if config_src.exists():
            config_dest = app_dir / "config"
            if config_dest.exists():
                shutil.rmtree(config_dest)
            shutil.copytree(config_src, config_dest)
            config_files = len(list(config_dest.rglob("*.*")))
            print(f"  Copied config/ ({config_files} files)")
        else:
            print(f"  [SKIP] config/ directory not found")


        steno_src = project_root / "stenographer.gif"
        if steno_src.exists():
            steno_dest = build_dir / "stenographer.gif"
            shutil.copy2(steno_src, steno_dest)
            print(f"  Copied stenographer.gif")
        else:
            print(f"  [SKIP] stenographer.gif not found")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to copy assets: {e}")
        return False


def compile_to_pyc(python_exe: Path, app_dir: Path) -> bool:
    """
    Compiles all .py files in app directory to .pyc bytecode.

    Uses compileall module to compile all Python files to bytecode,
    then removes source .py files and __pycache__ directories.
    This improves startup time and provides code protection.

    Args:
        python_exe: Path to python.exe
        app_dir: Directory containing Python files to compile

    Returns:
        True if successful, False otherwise
    """
    print("Compiling Python files to bytecode...")

    try:
        # Use compileall module to compile all .py files
        # -b flag creates .pyc files alongside .py files (not in __pycache__)
        cmd = [
            str(python_exe),
            "-m", "compileall",
            "-b",  # Write bytecode to .pyc files (same directory as .py)
            "-q",  # Quiet mode
            str(app_dir)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"  [ERROR] Compilation failed:")
            print(f"  {result.stderr}")
            return False

        # Count compiled files before cleanup
        pyc_files = list(app_dir.rglob('*.pyc'))
        pyc_count = len(pyc_files)
        print(f"  Compiled {pyc_count} .pyc files")

        # Remove source .py files (keep only .pyc)
        py_files = list(app_dir.rglob('*.py'))
        for py_file in py_files:
            py_file.unlink()
        print(f"  Removed {len(py_files)} .py source files")

        # Remove __pycache__ directories
        pycache_dirs = list(app_dir.rglob('__pycache__'))
        for pycache_dir in pycache_dirs:
            shutil.rmtree(pycache_dir)
        if pycache_dirs:
            print(f"  Removed {len(pycache_dirs)} __pycache__ directories")

        return True

    except Exception as e:
        print(f"  [ERROR] Compilation error: {e}")
        return False


def compile_site_packages(python_exe: Path, site_packages_dir: Path) -> bool:
    """
    Compiles all .py files in site-packages to .pyc bytecode and removes sources.

    This reduces distribution size and provides consistency with application code.
    Compiles all third-party packages to bytecode-only format.

    Args:
        python_exe: Path to python.exe
        site_packages_dir: Directory containing third-party packages (Lib/site-packages/)

    Returns:
        True if successful, False otherwise
    """
    print("Compiling third-party packages to bytecode...")

    try:
        # Use compileall module to compile all .py files
        # -b flag creates .pyc files alongside .py files (not in __pycache__)
        cmd = [
            str(python_exe),
            "-m", "compileall",
            "-b",  # Write bytecode to .pyc files (same directory as .py)
            "-q",  # Quiet mode
            str(site_packages_dir)
        ]

        print(f"  Compiling site-packages...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout for large packages
        )

        if result.returncode != 0:
            print(f"  [ERROR] Compilation failed:")
            print(f"  {result.stderr}")
            return False

        # Count compiled files before cleanup
        pyc_files = list(site_packages_dir.rglob('*.pyc'))
        pyc_count = len(pyc_files)
        print(f"  Compiled {pyc_count} .pyc files")

        # Calculate size before cleanup
        py_files = list(site_packages_dir.rglob('*.py'))
        size_before = sum(f.stat().st_size for f in py_files) / (1024 * 1024)

        # Remove source .py files (keep only .pyc)
        for py_file in py_files:
            py_file.unlink()
        print(f"  Removed {len(py_files)} .py source files ({size_before:.1f}MB)")

        # Remove __pycache__ directories
        pycache_dirs = list(site_packages_dir.rglob('__pycache__'))
        for pycache_dir in pycache_dirs:
            shutil.rmtree(pycache_dir)
        if pycache_dirs:
            print(f"  Removed {len(pycache_dirs)} __pycache__ directories")

        return True

    except Exception as e:
        print(f"  [ERROR] Compilation error: {e}")
        return False


def cleanup_package_metadata(site_packages_dir: Path) -> bool:
    """
    Removes redundant package metadata folders from site-packages.

    After license collection to LICENSES/ folder, the following are redundant:
    - *.dist-info/ folders (contain licenses, metadata, RECORD files)
    - *.egg-info/ folders (legacy package metadata)

    These folders are not needed for runtime and can be safely removed to
    reduce distribution size.

    EXCEPTION: Some packages use importlib.metadata to detect optional dependencies,
    so we preserve metadata for packages that are runtime-checked:
    - hf_xet (checked by huggingface_hub)

    Args:
        site_packages_dir: Directory containing third-party packages (Lib/site-packages/)

    Returns:
        True if successful, False otherwise
    """
    print("Cleaning up redundant package metadata...")

    # Packages whose metadata must be preserved for runtime detection
    PRESERVE_METADATA = {"hf_xet"}

    try:
        # Find all metadata folders
        dist_info_dirs = list(site_packages_dir.glob("*.dist-info"))
        egg_info_dirs = list(site_packages_dir.glob("*.egg-info"))

        # Filter out preserved packages
        def should_preserve(metadata_dir: Path) -> bool:
            """Check if metadata folder should be preserved"""
            dir_name = metadata_dir.name
            # Extract package name from metadata folder (e.g., "hf_xet-1.1.10.dist-info" -> "hf_xet")
            package_name = dir_name.split("-")[0]
            return package_name in PRESERVE_METADATA

        dist_info_dirs = [d for d in dist_info_dirs if not should_preserve(d)]
        egg_info_dirs = [e for e in egg_info_dirs if not should_preserve(e)]

        all_metadata_dirs = dist_info_dirs + egg_info_dirs

        if not all_metadata_dirs:
            print("  No metadata folders found (already clean)")
            return True

        # Calculate size before cleanup
        total_size = 0
        for metadata_dir in all_metadata_dirs:
            if metadata_dir.is_dir():
                total_size += sum(f.stat().st_size for f in metadata_dir.rglob("*") if f.is_file())
            elif metadata_dir.is_file():
                total_size += metadata_dir.stat().st_size

        size_mb = total_size / (1024 * 1024)

        # Remove metadata directories
        removed_count = 0
        for metadata_dir in all_metadata_dirs:
            try:
                if metadata_dir.is_dir():
                    shutil.rmtree(metadata_dir)
                elif metadata_dir.is_file():
                    metadata_dir.unlink()
                removed_count += 1
            except Exception as e:
                print(f"  [WARNING] Failed to remove {metadata_dir.name}: {e}")

        print(f"  Removed {removed_count} metadata folders ({size_mb:.1f}MB)")
        print(f"  - {len(dist_info_dirs)} .dist-info folders")
        print(f"  - {len(egg_info_dirs)} .egg-info folders")
        if PRESERVE_METADATA:
            print(f"  Preserved metadata for: {', '.join(sorted(PRESERVE_METADATA))}")

        return True

    except Exception as e:
        print(f"  [ERROR] Cleanup error: {e}")
        return False


def create_readme(build_dir: Path) -> bool:
    """
    Creates README.txt with user instructions.

    Args:
        build_dir: Build root directory (AI-Stenographer/)

    Returns:
        True if successful, False otherwise
    """
    print("Creating README.txt...")

    try:
        readme_content = """AI Stenographer - Real-time Speech-to-Text
===============================================

QUICK START
-----------
1. Double-click "AI - Stenographer.lnk" to launch
2. On first run, AI models will download automatically (~2GB)
3. Grant microphone permission when prompted
4. Start speaking - text appears in real-time!

REQUIREMENTS
------------
- Windows 10/11 (64-bit)
- Working microphone
- Internet connection (first run only for model download)
- ~3GB free disk space (models + application)

FEATURES
--------
- Real-time speech recognition
- High accuracy with Parakeet AI model
- Voice activity detection (only transcribes speech)
- No cloud processing - everything runs locally
- Privacy-focused - no data sent to external servers

USAGE
-----
- The application window shows transcribed text in real-time
- Press Ctrl+C in the window or close it to stop
- Models are downloaded once and reused on subsequent runs
- Models are stored in: ./models/

TROUBLESHOOTING
---------------
"Models downloading..." on first launch
  → Normal behavior. Wait for ~2GB download to complete
  → Requires internet connection

"Microphone not detected"
  → Check Windows sound settings
  → Ensure microphone is enabled and not muted
  → Grant microphone permission to the application

"Application won't start"
  → Check Windows Event Viewer for errors
  → Ensure antivirus isn't blocking the application
  → Try running as Administrator

"Transcription is inaccurate"
  → Speak clearly and at moderate pace
  → Ensure microphone is close enough
  → Check for background noise
  → Verify microphone quality in Windows settings

ADVANCED OPTIONS
----------------
Run from command line for verbose output:
  _internal\\runtime\\python.exe _internal\\app\\main.pyc -v

Custom windowing (reduce duplication):
  _internal\\runtime\\python.exe _internal\\app\\main.pyc --window=2.0 --step=1.5

PRIVACY & DATA
--------------
- All processing happens on your computer
- No internet connection required after initial model download
- No audio or text is sent to external servers
- Models are stored locally in ./models/

TECHNICAL DETAILS
-----------------
- AI Model: NVIDIA Parakeet TDT 0.6B (ONNX)
- VAD: Silero Voice Activity Detector
- Python: 3.13.0 (embedded)
- License: See LICENSE.txt

SUPPORT
-------
For issues, questions, or contributions:
- Check LICENSES/ folder for third-party licenses
- See LICENSE.txt for application license
- See EULA.txt for terms of use

VERSION
-------
Build Date: """ + Path(__file__).stat().st_mtime.__str__()[:10] + """
Platform: Windows x64
Python: 3.13.0

===============================================
"""

        readme_path = build_dir / "README.txt"
        readme_path.write_text(readme_content, encoding='utf-8')

        print(f"  Created README.txt")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to create README: {e}")
        return False


def zip_site_packages(site_packages_dir: Path) -> bool:
    """
    Packages .pyc files into .zip archives to reduce filesystem overhead.

    Creates one .zip archive per package/module containing all .pyc files,
    while preserving native binaries (.pyd, .dll) and data folders outside.

    Strategy:
    - Package directories (e.g., numpy/) → numpy.zip with all .pyc files
    - Single .pyc files (e.g., sounddevice.pyc) → sounddevice.zip
    - Native binaries (.pyd, .dll) stay uncompressed outside .zip
    - Data folders (*.libs/, *_data/) stay uncompressed
    - Metadata (.dist-info/) preserved for runtime detection

    Args:
        site_packages_dir: Path to site-packages directory

    Returns:
        True if successful, False otherwise

    Example reduction:
        Before: 2,997 .pyc files across ~40 packages
        After: ~40 .zip files + ~300 native binaries
    """
    print("Packaging .pyc files into .zip archives...")

    try:
        # Folders to preserve (contain native libraries or runtime data)
        DATA_FOLDER_PATTERNS = {'.libs', '_data', '.dist-info', '.egg-info'}

        def is_data_folder(path: Path) -> bool:
            """Check if folder should be preserved uncompressed."""
            return any(pattern in path.name for pattern in DATA_FOLDER_PATTERNS)

        def has_native_extensions(package_dir: Path) -> bool:
            """Check if package contains native extensions (.pyd, .dll)."""
            return any(package_dir.rglob('*.pyd')) or any(package_dir.rglob('*.dll'))

        # Find all top-level items (packages and single-file modules)
        top_level_items = []

        for item in site_packages_dir.iterdir():
            # Skip data folders and metadata
            if is_data_folder(item):
                continue

            # Skip already created .zip files (in case of re-run)
            if item.suffix == '.zip':
                continue

            # Skip special files
            if item.name.startswith('.') or item.suffix == '.pth':
                continue

            top_level_items.append(item)

        if not top_level_items:
            print("  No packages found to zip")
            return True

        zipped_count = 0
        pyc_count = 0

        for item in top_level_items:
            if item.is_dir():
                package_name = item.name

                # Skip packages with native extensions - they must stay on filesystem
                if has_native_extensions(item):
                    continue

                # Package directory - zip all .pyc files
                zip_path = site_packages_dir / f"{package_name}.zip"

                # Collect all .pyc files recursively
                pyc_files = list(item.rglob('*.pyc'))

                if not pyc_files:
                    # No .pyc files, skip this package
                    continue

                # Count non-.pyc files to see if package will remain after zipping
                all_files = list(item.rglob('*'))
                non_pyc_files = [f for f in all_files if f.is_file() and f.suffix != '.pyc']

                if non_pyc_files:
                    # Package has non-.pyc content (e.g., py.typed, templates/)
                    # Can't zip because directory must stay on filesystem
                    # Zipping would cause import conflicts (dir vs zip)
                    continue

                # Package is pure .pyc - safe to zip and remove directory
                # Create .zip archive
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for pyc_file in pyc_files:
                        # Store with relative path from site-packages
                        arcname = pyc_file.relative_to(site_packages_dir)
                        zf.write(pyc_file, arcname)

                # Remove .pyc files after successful archiving
                for pyc_file in pyc_files:
                    pyc_file.unlink()
                    pyc_count += 1

                # Clean up empty directories
                # Walk bottom-up to remove empty subdirectories first
                for subdir in sorted(item.rglob('*'), reverse=True):
                    if subdir.is_dir() and not any(subdir.iterdir()):
                        subdir.rmdir()

                # Package directory should now be empty - remove it
                if not any(item.iterdir()):
                    item.rmdir()

                zipped_count += 1

            elif item.is_file() and item.suffix == '.pyc':
                # Single-file module - keep as-is (.pyc files can be imported directly)
                # No need to zip single files - they're already small and importable
                pass

        print(f"  Created {zipped_count} .zip archives from {pyc_count} .pyc files")

        # Create sitecustomize.py to add .zip files to sys.path
        if zipped_count > 0:
            sitecustomize_content = '''"""
Site customization for AI-Stenographer distribution.

Automatically adds .zip archives in site-packages to sys.path for zipimport.
"""
import sys
from pathlib import Path

# Find all .zip files in site-packages and add them to sys.path
site_packages = Path(__file__).parent
for zip_file in site_packages.glob('*.zip'):
    zip_path = str(zip_file)
    if zip_path not in sys.path:
        sys.path.insert(0, zip_path)
'''
            sitecustomize_path = site_packages_dir / "sitecustomize.py"
            sitecustomize_path.write_text(sitecustomize_content, encoding='utf-8')
            print(f"  Created sitecustomize.py for zipimport support")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to zip site-packages: {e}")
        return False


def create_launcher(build_dir: Path) -> bool:
    """
    Creates Windows shortcut (.lnk) to launch the application.

    The launcher:
    - Sets working directory to build_dir (so ./models/ resolves correctly)
    - Runs pythonw.exe (no console window)
    - Executes _internal/app/main.pyc

    Args:
        build_dir: Build root directory (AI-Stenographer/)

    Returns:
        True if successful, False otherwise
    """
    print("Creating launcher shortcut...")

    try:
        import win32com.client

        # Verify required files exist
        python_exe = build_dir / "_internal" / "runtime" / "pythonw.exe"
        main_pyc = build_dir / "_internal" / "app" / "main.pyc"

        if not python_exe.exists():
            print(f"  [ERROR] pythonw.exe not found: {python_exe}")
            return False

        if not main_pyc.exists():
            print(f"  [ERROR] main.pyc not found: {main_pyc}")
            return False

        # Create shortcut
        # Use cmd.exe for portable relative paths, start /B to detach process
        # /C executes command and terminates
        # Using pythonw.exe (not python.exe) ensures no console window appears
        shortcut_path = build_dir / "AI - Stenographer.lnk"
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(str(shortcut_path))
        shortcut.TargetPath = r"%windir%\system32\cmd.exe"
        shortcut.Arguments = r'/C "start /B _internal\runtime\pythonw.exe _internal\app\main.pyc"'
        shortcut.WorkingDirectory = ''
        shortcut.IconLocation = r"%SystemRoot%\System32\imageres.dll,364"
        shortcut.Description = "AI Stenographer - Real-time Speech-to-Text"
        shortcut.WindowStyle = 1
        shortcut.save()

        print(f"  Created launcher: {shortcut_path.name}")
        return True

    except ImportError:
        print(f"  [ERROR] pywin32 not installed")
        print(f"  Run: pip install pywin32")
        return False
    except Exception as e:
        print(f"  [ERROR] Failed to create launcher: {e}")
        return False


if __name__ == "__main__":
    sys.exit(main())
