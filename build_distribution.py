"""
Windows distribution builder for STT application.

Creates portable Windows distribution using:
- Python embeddable package (signed executables)
- Pre-compiled bytecode (.pyc files)
- Pre-installed dependencies
- Custom launcher with icon

Distribution structure:
STT-Stenographer/
├── STT - Stenographer.lnk
├── README.txt
├── LICENSE.txt
├── icon.ico
└── _internal/
    ├── runtime/        # Python executables
    ├── Lib/            # Dependencies
    ├── app/            # Application code (.pyc)
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


# Python version to download
PYTHON_VERSION = "3.13.5"
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
        root_dir: Root distribution directory (e.g., STT-Stenographer/)

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
        return result.stdout.strip().lower() == "true"

    except Exception as e:
        print(f"Warning: Could not verify signature for {file_path}: {e}")
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
    print("STT Distribution Builder - Steps 1-8")
    print("=" * 60)

    # Setup paths
    project_root = Path(__file__).parent
    cache_dir = project_root / ".cache"
    dist_dir = project_root / "dist"
    build_dir = dist_dir / "STT-Stenographer"

    # Clean previous build
    if build_dir.exists():
        print(f"Cleaning previous build: {build_dir}")
        shutil.rmtree(build_dir)

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

    # Step 4: Verify signatures
    if not verify_signatures(paths["runtime"]):
        print("\nWarning: Python executables are not properly signed")
        print("This may cause SmartScreen warnings on user machines")

    # Step 5: Copy tkinter module from system Python
    if not copy_tkinter_to_distribution(paths["runtime"]):
        print("\nError: Failed to copy tkinter module")
        print("Tkinter is required for GUI functionality")
        return 1

    # Step 6: Create python313._pth configuration
    # Paths are relative to python.exe location (_internal/runtime/)
    try:
        pth_paths = [
            "python313.zip",            # In same dir as python.exe
            "Lib",                      # For tkinter module (in runtime/Lib/)
            "../Lib/site-packages",     # Up one, then to Lib/site-packages
            "../app",                   # Up one, then to app
            "import site"
        ]
        create_pth_file(paths["runtime"], pth_paths)
    except Exception as e:
        print(f"\nError creating _pth file: {e}")
        return 1

    # Step 7: Enable pip in embedded Python
    try:
        enable_pip(paths["runtime"])
    except Exception as e:
        print(f"\nError enabling pip: {e}")
        return 1

    # Step 8: Verify pip is available
    python_exe = paths["runtime"] / "python.exe"
    if not verify_pip(python_exe):
        print("\nWarning: Pip verification failed")
        print("Dependency installation may not work")

    # Step 9: Verify tkinter is importable
    if not verify_tkinter(python_exe):
        print("\nWarning: Tkinter verification failed")
        print("GUI functionality may not work")

    # Step 10: Copy legal documents
    try:
        copy_legal_documents(project_root, build_dir)
    except Exception as e:
        print(f"\nError copying legal documents: {e}")
        return 1

    # Step 11: Install dependencies
    requirements_file = project_root / "requirements.txt"
    if not install_dependencies(python_exe, paths["lib"], requirements_file):
        print("\nError: Dependency installation failed")
        return 1

    # Step 12: Verify native libraries
    lib_results = verify_native_libraries(paths["lib"])
    if not all(lib_results.values()):
        print("\nWarning: Some native libraries missing")
        print("Application may not work correctly")

    # Step 13: Test critical imports
    critical_modules = ["numpy", "onnxruntime", "sounddevice", "onnx_asr", "tkinter"]
    import_results = test_imports(python_exe, critical_modules)
    failed_imports = [m for m, success in import_results.items() if not success]

    if failed_imports:
        print(f"\nWarning: Failed to import: {', '.join(failed_imports)}")
        print("Application may not work correctly")

    # Step 14: Copy application code
    src_dir = project_root / "src"
    main_py = project_root / "main.py"
    if not copy_application_code(src_dir, main_py, paths["app"]):
        print("\nError: Failed to copy application code")
        return 1

    # Step 15: Compile to bytecode
    if not compile_to_pyc(python_exe, paths["app"]):
        print("\nError: Failed to compile code to bytecode")
        return 1

    # Step 16: Copy assets and configuration
    if not copy_assets_and_config(project_root, build_dir, paths["app"]):
        print("\nError: Failed to copy assets and configuration")
        return 1

    # Step 17: Create README documentation
    if not create_readme(build_dir):
        print("\nError: Failed to create README")
        return 1

    # Step 18: Create launcher shortcut
    if not create_launcher(build_dir):
        print("\nError: Failed to create launcher shortcut")
        return 1

    print("\n" + "=" * 60)
    print("Build completed successfully!")
    print(f"Build directory: {build_dir}")
    print(f"\nTo run the application:")
    print(f'  Double-click: "{build_dir}\\STT - Stenographer.lnk"')
    print("=" * 60)

    return 0


def create_pth_file(runtime_dir: Path, paths: list[str]) -> None:
    """
    Creates python313._pth file to configure module search paths.

    The _pth file tells embedded Python where to find modules without
    needing environment variables or registry settings.

    Args:
        runtime_dir: Directory containing Python runtime
        paths: List of paths to add to sys.path (relative to exe location)

    Example paths:
        _internal/runtime/python313.zip
        _internal/Lib/site-packages
        _internal/app
        import site
    """
    print("Creating python313._pth configuration...")

    pth_file = runtime_dir / "python313._pth"

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


def copy_tkinter_to_distribution(runtime_dir: Path) -> bool:
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

    Returns:
        True if tkinter was successfully copied, False otherwise
    """
    print("Copying tkinter module and Tcl/Tk libraries from system Python...")

    try:
        # Locate system Python (use sys.base_prefix to get actual installation, not venv)
        system_python = Path(sys.base_prefix)

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
        build_dir: Build root directory (STT-Stenographer/)
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


def verify_native_libraries(site_packages: Path) -> dict[str, bool]:
    """
    Verifies that native libraries (DLLs) are present.
    
    Checks for critical DLLs:
    - ONNX Runtime: onnxruntime.dll, onnxruntime_providers_shared.dll
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
    
    results["onnxruntime"] = onnx_dll.exists() and onnx_providers_dll.exists()
    
    if results["onnxruntime"]:
        print(f"  [OK] ONNX Runtime DLLs found")
    else:
        print(f"  [MISSING] ONNX Runtime DLLs not found")
    
    # Check sounddevice/PortAudio DLL
    sd_data = site_packages / "_sounddevice_data"
    portaudio_dll = None
    
    if sd_data.exists():
        # Check for various PortAudio DLL names
        for dll_name in ["portaudio_x64.dll", "portaudio.dll", "_portaudio.pyd"]:
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
        # Verify source exists
        if not src_dir.exists():
            print(f"  [ERROR] Source directory not found: {src_dir}")
            return False

        if not main_py.exists():
            print(f"  [ERROR] main.py not found: {main_py}")
            return False

        # Copy src/ directory
        target_src = app_dir / "src"
        if target_src.exists():
            shutil.rmtree(target_src)

        shutil.copytree(src_dir, target_src)
        print(f"  Copied src/ ({sum(1 for _ in src_dir.rglob('*.py'))} Python files)")

        # Copy main.py
        target_main = app_dir / "main.py"
        shutil.copy2(main_py, target_main)
        print(f"  Copied main.py")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to copy code: {e}")
        return False


def copy_assets_and_config(project_root: Path, build_dir: Path, app_dir: Path) -> bool:
    """
    Copies assets and configuration files to distribution.

    Copies:
    - config/ directory to _internal/app/config/
    - icon.ico to build root

    Args:
        project_root: Project root directory
        build_dir: Build root directory (STT-Stenographer/)
        app_dir: App directory (_internal/app/)

    Returns:
        True if successful (even if some files are missing)
    """
    print("Copying assets and configuration...")

    try:
        # Copy config directory
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

        # Copy icon to root
        icon_src = project_root / "icon.ico"
        if icon_src.exists():
            icon_dest = build_dir / "icon.ico"
            shutil.copy2(icon_src, icon_dest)
            print(f"  Copied icon.ico")
        else:
            print(f"  [SKIP] icon.ico not found")

        # Copy stenographer.jpg to root (for loading window)
        steno_src = project_root / "stenographer.jpg"
        if steno_src.exists():
            steno_dest = build_dir / "stenographer.jpg"
            shutil.copy2(steno_src, steno_dest)
            print(f"  Copied stenographer.jpg")
        else:
            print(f"  [SKIP] stenographer.jpg not found")

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


def create_readme(build_dir: Path) -> bool:
    """
    Creates README.txt with user instructions.

    Args:
        build_dir: Build root directory (STT-Stenographer/)

    Returns:
        True if successful, False otherwise
    """
    print("Creating README.txt...")

    try:
        readme_content = """STT Stenographer - Real-time Speech-to-Text
===============================================

QUICK START
-----------
1. Double-click "STT - Stenographer.lnk" to launch
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


def create_launcher(build_dir: Path) -> bool:
    """
    Creates Windows shortcut (.lnk) to launch the application.

    The launcher:
    - Sets working directory to build_dir (so ./models/ resolves correctly)
    - Runs pythonw.exe (no console window)
    - Executes _internal/app/main.pyc
    - Uses icon.ico for the shortcut

    Args:
        build_dir: Build root directory (STT-Stenographer/)

    Returns:
        True if successful, False otherwise
    """
    print("Creating launcher shortcut...")

    try:
        import win32com.client

        # Verify required files exist
        python_exe = build_dir / "_internal" / "runtime" / "pythonw.exe"
        main_pyc = build_dir / "_internal" / "app" / "main.pyc"
        icon_file = build_dir / "icon.ico"

        if not python_exe.exists():
            print(f"  [ERROR] pythonw.exe not found: {python_exe}")
            return False

        if not main_pyc.exists():
            print(f"  [ERROR] main.pyc not found: {main_pyc}")
            return False

        # Create shortcut
        shortcut_path = build_dir / "STT - Stenographer.lnk"

        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(str(shortcut_path))

        # Set target to pythonw.exe (no console window)
        shortcut.TargetPath = str(python_exe)

        # Set arguments to run main.pyc
        shortcut.Arguments = r"_internal\app\main.pyc"

        # Set working directory to build root (so ./models/ and ./stenographer.jpg resolve)
        shortcut.WorkingDirectory = str(build_dir)

        # Set icon if it exists
        if icon_file.exists():
            shortcut.IconLocation = str(icon_file)

        # Set description
        shortcut.Description = "STT Stenographer - Real-time Speech-to-Text"

        # Save shortcut
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
