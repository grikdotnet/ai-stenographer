"""
MSIX distribution builder for AI Stenographer (Microsoft Store).

Creates MSIX package using:
- Python embeddable package (with signed executables)
- Pre-compiled bytecode (.pyc files)
- Pre-installed dependencies
- Native launcher executable (AIStenographer.exe)
- AppxManifest.xml with Store metadata
- Store assets (PNG icons)

Key Differences from Portable Distribution:
- Uses native .exe launcher instead of .lnk shortcut
- Includes AppxManifest.xml for Store compliance
- Includes Store assets (icons)
- Models stored in AppData (not _internal)
- Entire package signed with certificate

MSIX Package Structure:
AI.Stenographer_1.0.0.0_x64/
├── AppxManifest.xml           # Store package manifest
├── AIStenographer.exe          # Native launcher
├── Assets/                     # Store icons
│   ├── Square44x44Logo.png
│   ├── Square150x150Logo.png
│   └── StoreLogo.png
├── _internal/
│   ├── runtime/               # Python 3.13.5 embeddable
│   ├── Lib/site-packages/     # Dependencies
│   ├── app/                   # Application bytecode
│   └── models/                # Empty (download on first run)
├── README.txt
├── LICENSE.txt
├── EULA.txt
├── PRIVACY_POLICY.txt
└── THIRD_PARTY_NOTICES.txt
"""
import sys
import shutil
from pathlib import Path
from typing import Dict
import subprocess

# Import shared build functions from portable distribution builder
sys.path.insert(0, str(Path(__file__).parent.parent))
from build_distribution import (
    download_embedded_python,
    extract_embedded_python,
    create_directory_structure,
    copy_signed_executables_from_system,
    verify_signatures,
    copy_tkinter_to_distribution,
    cleanup_tcl_unnecessary_files,
    verify_tkinter,
    create_pth_file,
    enable_pip,
    verify_pip,
    collect_third_party_licenses,
    install_dependencies,
    remove_pip_from_distribution,
    remove_tests_from_distribution,
    verify_native_libraries,
    test_imports,
    copy_application_code,
    compile_to_pyc,
    compile_site_packages,
    zip_site_packages,
    cleanup_package_metadata,
    copy_assets_and_config,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Build type
BUILD_TYPE = "store"

# Python version
PYTHON_VERSION = "3.13.5"

# MSIX package metadata (must match AppxManifest.xml)
APP_VERSION = "1.5.1.0"
PUBLISHER_NAME = "CN=3F8691C4-05D3-45C7-AB1E-113776D7E567"
PACKAGE_NAME = "GrigoriKochanov.AIStenographer"

# Skip signing for Store submission (Microsoft re-signs with trusted certificate)
SKIP_SIGNING = True

# Certificate configuration
CERT_PASSWORD = "test123"  # Test certificate password
TIMESTAMP_SERVER = "http://timestamp.digicert.com"

# Windows SDK paths (auto-detected, can be overridden)
DEFAULT_SDK_PATHS = [
    r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64",
    r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22000.0\x64",
    r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64",
    r"C:\Program Files (x86)\Windows Kits\10\App Certification Kit",
]


# ============================================================================
# WINDOWS SDK DETECTION
# ============================================================================

def find_windows_sdk() -> Path:
    """
    Locates Windows SDK tools (MakeAppx.exe, SignTool.exe).

    Searches common installation paths for Windows SDK 10.
    Required tools:
    - MakeAppx.exe - Creates MSIX packages
    - SignTool.exe - Signs packages with certificates

    Returns:
        Path to SDK bin directory containing tools

    Raises:
        RuntimeError: If SDK tools not found
    """
    print("Locating Windows SDK tools...")

    for sdk_path in DEFAULT_SDK_PATHS:
        sdk_dir = Path(sdk_path)
        if not sdk_dir.exists():
            continue

        makeappx_exe = sdk_dir / "MakeAppx.exe"
        signtool_exe = sdk_dir / "SignTool.exe"

        if makeappx_exe.exists() and signtool_exe.exists():
            print(f"  Found Windows SDK: {sdk_dir}")
            print(f"  MakeAppx.exe: {makeappx_exe}")
            print(f"  SignTool.exe: {signtool_exe}")
            return sdk_dir

    # SDK not found
    raise RuntimeError(
        "Windows SDK not found. Please install Windows SDK 10.\n"
        "Download: https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/\n"
        "Required tools: MakeAppx.exe, SignTool.exe\n"
        f"Searched paths:\n" + "\n".join(f"  - {p}" for p in DEFAULT_SDK_PATHS)
    )


# ============================================================================
# MSIX-SPECIFIC FILE OPERATIONS
# ============================================================================

def copy_msix_specific_files(staging_dir: Path, msix_dir: Path, project_root: Path) -> bool:
    """
    Copies MSIX-specific files to staging directory.

    Files copied:
    - AppxManifest.xml (package manifest)
    - AIStenographer.exe (native launcher)
    - Assets/*.png (Store icons, 5 files)
    - PRIVACY_POLICY.txt (Store requirement)

    Args:
        staging_dir: MSIX staging directory root
        msix_dir: Source msix/ directory
        project_root: Project root directory

    Returns:
        True if successful, False otherwise
    """
    print("Copying MSIX-specific files...")

    try:
        # 1. Copy AppxManifest.xml
        manifest_src = msix_dir / "AppxManifest.xml"
        manifest_dest = staging_dir / "AppxManifest.xml"

        if not manifest_src.exists():
            print(f"  [ERROR] AppxManifest.xml not found: {manifest_src}")
            return False

        shutil.copy2(manifest_src, manifest_dest)
        print(f"  Copied AppxManifest.xml")

        # 2. Copy native launcher
        launcher_src = msix_dir / "launcher" / "AIStenographer.exe"
        launcher_dest = staging_dir / "AIStenographer.exe"

        if not launcher_src.exists():
            print(f"  [ERROR] AIStenographer.exe not found: {launcher_src}")
            print(f"  Please build the launcher first using msix/launcher/build_launcher.py")
            return False

        shutil.copy2(launcher_src, launcher_dest)
        print(f"  Copied AIStenographer.exe ({launcher_src.stat().st_size} bytes)")

        # 3. Copy Store assets
        assets_src = msix_dir / "Assets"
        assets_dest = staging_dir / "Assets"

        if not assets_src.exists():
            print(f"  [ERROR] Assets directory not found: {assets_src}")
            return False

        if assets_dest.exists():
            shutil.rmtree(assets_dest)

        shutil.copytree(assets_src, assets_dest)

        # Verify all required assets exist
        required_assets = [
            "Square44x44Logo.png",
            "Square150x150Logo.png",
            "StoreLogo.png",
        ]

        missing_assets = []
        for asset_name in required_assets:
            asset_path = assets_dest / asset_name
            if not asset_path.exists():
                missing_assets.append(asset_name)

        if missing_assets:
            print(f"  [ERROR] Missing required assets: {', '.join(missing_assets)}")
            return False

        print(f"  Copied Assets/ ({len(required_assets)} PNG files)")

        # 4. Copy PRIVACY_POLICY.txt (Store requirement)
        privacy_src = project_root / "PRIVACY_POLICY.txt"
        privacy_dest = staging_dir / "PRIVACY_POLICY.txt"

        if not privacy_src.exists():
            print(f"  [WARNING] PRIVACY_POLICY.txt not found: {privacy_src}")
            print(f"  This file is required for Microsoft Store submission")
            return False

        shutil.copy2(privacy_src, privacy_dest)
        print(f"  Copied PRIVACY_POLICY.txt")

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to copy MSIX files: {e}")
        return False


def create_store_readme(staging_dir: Path) -> bool:
    """
    Creates Store-specific README.txt with installation instructions.

    Store README differences from portable:
    - Mentions AppData paths for models/config
    - Notes MSIX installation process
    - No launcher shortcut instructions

    Args:
        staging_dir: MSIX staging directory root

    Returns:
        True if successful, False otherwise
    """
    print("Creating Store-specific README.txt...")

    try:
        readme_content = """AI Stenographer - Real-time Speech-to-Text (Microsoft Store)
===============================================================

MICROSOFT STORE VERSION
-----------------------
This version is packaged for Microsoft Store distribution.

Models and configuration are stored in:
  %LOCALAPPDATA%\\AI-Stenographer\\

This ensures proper sandboxing and user data isolation.

QUICK START
-----------
1. Launch "AI Stenographer" from Start Menu
2. On first run, AI models will download automatically (~2GB)
3. Grant microphone permission when prompted
4. Start speaking - text appears in real-time!

REQUIREMENTS
------------
- Windows 10 version 1809 (October 2018 Update) or later
- Windows 11 (all versions)
- Working microphone
- Internet connection (first run only for model download)
- ~3GB free disk space in AppData folder

FEATURES
--------
- Real-time speech recognition
- High accuracy with Parakeet AI model
- Voice activity detection (only transcribes speech)
- No cloud processing - everything runs locally
- Privacy-focused - no data sent to external servers

DATA STORAGE
------------
The Store version stores data in your AppData folder:

Models (~2GB):
  %LOCALAPPDATA%\\AI-Stenographer\\models\\

Configuration:
  %LOCALAPPDATA%\\AI-Stenographer\\config\\

Logs:
  %LOCALAPPDATA%\\AI-Stenographer\\logs\\

This location is automatically cleaned when you uninstall the app.

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
  → Restart the application
  → Try reinstalling from Microsoft Store

"Transcription is inaccurate"
  → Speak clearly and at moderate pace
  → Ensure microphone is close enough
  → Check for background noise
  → Verify microphone quality in Windows settings

PRIVACY & DATA
--------------
- All processing happens on your computer
- No internet connection required after initial model download
- No audio or text is sent to external servers
- Models are stored locally in AppData

For detailed privacy information, see PRIVACY_POLICY.txt

TECHNICAL DETAILS
-----------------
- AI Model: NVIDIA Parakeet TDT 0.6B (ONNX)
- VAD: Silero Voice Activity Detector
- Python: 3.13.5 (embedded)
- Package: MSIX

SUPPORT
-------
For issues, questions, or contributions:
- Microsoft Store: Use "Rate and review" to report issues
- See LICENSE.txt for application license
- See EULA.txt for terms of use
- See THIRD_PARTY_NOTICES.txt for attribution

VERSION
-------
Package Version: """ + APP_VERSION + """
Platform: Windows x64
Python: """ + PYTHON_VERSION + """

===============================================================
"""

        readme_path = staging_dir / "README.txt"
        readme_path.write_text(readme_content, encoding='utf-8')

        print(f"  Created Store README.txt")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to create README: {e}")
        return False


# ============================================================================
# MSIX PACKAGE CREATION
# ============================================================================

def validate_package_structure(staging_dir: Path) -> bool:
    """
    Validates MSIX package structure before creating .msix file.

    Checks for:
    - AppxManifest.xml
    - AIStenographer.exe
    - Assets/*.png (5 files)
    - _internal/runtime/pythonw.exe
    - _internal/app/main.pyc

    Args:
        staging_dir: MSIX staging directory root

    Returns:
        True if structure is valid, False otherwise
    """
    print("Validating MSIX package structure...")

    required_files = {
        "AppxManifest.xml": staging_dir / "AppxManifest.xml",
        "AIStenographer.exe": staging_dir / "AIStenographer.exe",
        "pythonw.exe": staging_dir / "_internal" / "runtime" / "pythonw.exe",
        "main.pyc": staging_dir / "_internal" / "app" / "main.pyc",
    }

    # Check required files
    missing_files = []
    for file_name, file_path in required_files.items():
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        print(f"  [ERROR] Missing required files: {', '.join(missing_files)}")
        return False

    print(f"  [OK] Required files present")

    # Check Assets directory
    assets_dir = staging_dir / "Assets"
    if not assets_dir.exists():
        print(f"  [ERROR] Assets directory not found")
        return False

    required_assets = [
        "Square44x44Logo.png",
        "Square150x150Logo.png",
        "StoreLogo.png",
    ]

    missing_assets = []
    for asset_name in required_assets:
        asset_path = assets_dir / asset_name
        if not asset_path.exists():
            missing_assets.append(asset_name)

    if missing_assets:
        print(f"  [ERROR] Missing assets: {', '.join(missing_assets)}")
        return False

    print(f"  [OK] Store assets present ({len(required_assets)} files)")

    return True


def create_msix_package(staging_dir: Path, output_path: Path, sdk_dir: Path) -> bool:
    """
    Creates MSIX package using MakeAppx.exe.

    Uses Windows SDK MakeAppx.exe tool to create .msix package from
    staging directory.

    Args:
        staging_dir: MSIX staging directory root
        output_path: Output .msix file path
        sdk_dir: Windows SDK bin directory

    Returns:
        True if successful, False otherwise
    """
    print(f"Creating MSIX package: {output_path.name}")

    try:
        makeappx_exe = sdk_dir / "MakeAppx.exe"

        if not makeappx_exe.exists():
            print(f"  [ERROR] MakeAppx.exe not found: {makeappx_exe}")
            return False

        # Validate structure before packaging
        if not validate_package_structure(staging_dir):
            return False

        # Build MakeAppx command - use shell=True to properly handle Windows paths
        # Quote paths to handle spaces in directory names
        staging_str = str(staging_dir).replace("/", "\\")
        output_str = str(output_path).replace("/", "\\")
        makeappx_str = str(makeappx_exe).replace("/", "\\")

        cmd = f'"{makeappx_str}" pack /d "{staging_str}" /p "{output_str}" /nv /o'

        print(f"  Running MakeAppx.exe...")
        print(f"  Command: {cmd}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            timeout=300  # 5 minutes timeout
        )

        if result.returncode != 0:
            print(f"  [ERROR] MakeAppx.exe failed (exit code {result.returncode}):")
            if result.stdout:
                print(f"  stdout: {result.stdout}")
            if result.stderr:
                print(f"  stderr: {result.stderr}")
            return False

        # Verify output file exists
        if not output_path.exists():
            print(f"  [ERROR] Output file not created: {output_path}")
            return False

        # Report package size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Package created: {output_path.name} ({size_mb:.1f}MB)")

        return True

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] MakeAppx.exe timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"  [ERROR] Failed to create MSIX package: {e}")
        return False


def sign_msix_package(msix_path: Path, cert_path: Path, sdk_dir: Path) -> bool:
    """
    Signs MSIX package using SignTool.exe.

    Uses Windows SDK SignTool.exe to sign package with test certificate.
    Includes timestamp server for signature validity after cert expiry.

    Args:
        msix_path: Path to .msix package to sign
        cert_path: Path to .pfx certificate file
        sdk_dir: Windows SDK bin directory

    Returns:
        True if successful, False otherwise
    """
    print(f"Signing MSIX package: {msix_path.name}")

    try:
        signtool_exe = sdk_dir / "SignTool.exe"

        if not signtool_exe.exists():
            print(f"  [ERROR] SignTool.exe not found: {signtool_exe}")
            return False

        if not cert_path.exists():
            print(f"  [ERROR] Certificate not found: {cert_path}")
            print(f"  Please run msix/create_test_certificate.ps1 first")
            return False

        # Build SignTool command - use shell=True for proper Windows path handling
        signtool_str = str(signtool_exe).replace("/", "\\")
        cert_str = str(cert_path).replace("/", "\\")
        msix_str = str(msix_path).replace("/", "\\")

        cmd = (
            f'"{signtool_str}" sign /fd SHA256 /a '
            f'/f "{cert_str}" /p "{CERT_PASSWORD}" '
            f'/t "{TIMESTAMP_SERVER}" "{msix_str}"'
        )

        print(f"  Running SignTool.exe...")
        print(f"  Certificate: {cert_path.name}")
        print(f"  Timestamp server: {TIMESTAMP_SERVER}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"  [ERROR] SignTool.exe failed (exit code {result.returncode}):")
            if result.stdout:
                print(f"  stdout: {result.stdout}")
            if result.stderr:
                print(f"  stderr: {result.stderr}")
            print(f"\nTroubleshooting:")
            print(f"  - Verify certificate is valid")
            print(f"  - Check timestamp server connectivity")
            print(f"  - Ensure certificate password is correct")
            return False

        print(f"  Package signed successfully")

        # Verify signature
        print(f"  Verifying signature...")
        verify_cmd = f'"{signtool_str}" verify /pa "{msix_str}"'

        verify_result = subprocess.run(
            verify_cmd,
            capture_output=True,
            text=True,
            shell=True,
            timeout=30
        )

        if verify_result.returncode != 0:
            # Check if it's just the test certificate not being trusted (expected)
            # Normalize whitespace (tabs, newlines) to spaces for matching
            stderr_normalized = " ".join((verify_result.stderr or "").lower().split())
            is_untrusted_root = "root certificate which is not trusted" in stderr_normalized

            if is_untrusted_root:
                print(f"  [OK] Package signed (test certificate not in trusted store)")
                print(f"  Note: Install certificate before sideloading:")
                print(f"    Import-Certificate -FilePath \"msix\\AIStenographer_TestCert.cer\" -CertStoreLocation \"Cert:\\LocalMachine\\Root\"")
            else:
                print(f"  [WARNING] Signature verification failed:")
                if verify_result.stdout:
                    print(f"  stdout: {verify_result.stdout}")
                if verify_result.stderr:
                    print(f"  stderr: {verify_result.stderr}")
                return False
        else:
            print(f"  [OK] Signature verified (certificate trusted)")

        return True

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] SignTool.exe timed out")
        return False
    except Exception as e:
        print(f"  [ERROR] Failed to sign package: {e}")
        return False


def copy_legal_documents_msix(project_root: Path, staging_dir: Path) -> None:
    """
    Copies legal documents to MSIX staging directory.

    Copies:
    - LICENSE.txt (main license)
    - EULA.txt (end user license agreement)
    - PRIVACY_POLICY.txt (privacy policy - Store requirement)
    - THIRD_PARTY_NOTICES.txt (attribution)
    - LICENSES/ folder (third-party licenses)

    Args:
        project_root: Project root directory
        staging_dir: MSIX staging directory root
    """
    print("Copying legal documents...")

    # Copy legal text files to root
    legal_files = ["LICENSE.txt", "EULA.txt", "PRIVACY_POLICY.txt", "THIRD_PARTY_NOTICES.txt"]
    copied = 0

    for legal_file in legal_files:
        src = project_root / legal_file
        if src.exists():
            shutil.copy2(src, staging_dir / legal_file)
            print(f"  Copied {legal_file}")
            copied += 1
        else:
            print(f"  Warning: {legal_file} not found")

    # Copy LICENSES/ folder
    licenses_src = project_root / "LICENSES"
    licenses_dest = staging_dir / "LICENSES"

    if licenses_src.exists():
        if licenses_dest.exists():
            shutil.rmtree(licenses_dest)
        shutil.copytree(licenses_src, licenses_dest)
        license_count = len(list(licenses_dest.glob("*/")))
        print(f"  Copied LICENSES/ folder ({license_count} packages)")
    else:
        print(f"  Warning: LICENSES/ folder not found")

    print(f"Legal documents: {copied} files + LICENSES/ folder")


# ============================================================================
# MAIN BUILD FUNCTION
# ============================================================================

def main():
    """Main MSIX build script entry point."""
    print("=" * 80)
    print("AI Stenographer - MSIX Distribution Builder (Microsoft Store)")
    print("=" * 80)

    # Setup paths
    project_root = Path(__file__).parent.parent
    msix_dir = project_root / "msix"
    cache_dir = project_root / ".cache"
    dist_dir = project_root / "dist"
    staging_dir = dist_dir / f"{PACKAGE_NAME}_{APP_VERSION}_x64"

    # Find Windows SDK tools
    try:
        sdk_dir = find_windows_sdk()
    except RuntimeError as e:
        print(f"\n{e}")
        return 1

    # Clean previous build
    if staging_dir.exists():
        print(f"\nCleaning previous build: {staging_dir}")

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
            shutil.rmtree(staging_dir, onerror=handle_remove_readonly)
        except Exception as e:
            print(f"  Warning: Cleanup had errors: {e}")
            print(f"  Continuing anyway...")

    # Step 1: Download Python embeddable package
    print("\n" + "=" * 80)
    print("STEP 1: Download Python Embeddable Package")
    print("=" * 80)
    try:
        zip_path = download_embedded_python(PYTHON_VERSION, cache_dir)
    except Exception as e:
        print(f"\nError downloading Python: {e}")
        return 1

    # Step 2: Create directory structure
    print("\n" + "=" * 80)
    print("STEP 2: Create Directory Structure")
    print("=" * 80)
    try:
        paths = create_directory_structure(staging_dir)
    except Exception as e:
        print(f"\nError creating directories: {e}")
        return 1

    # Step 3: Extract Python to runtime directory
    print("\n" + "=" * 80)
    print("STEP 3: Extract Python Runtime")
    print("=" * 80)
    try:
        extract_embedded_python(zip_path, paths["runtime"])
    except Exception as e:
        print(f"\nError extracting Python: {e}")
        return 1

    # Step 4: Replace unsigned executables with signed versions
    print("\n" + "=" * 80)
    print("STEP 4: Replace with Signed Python Executables")
    print("=" * 80)
    if not copy_signed_executables_from_system(paths["runtime"]):
        print("\nWarning: Could not copy signed executables from system Python")

    # Step 5: Verify signatures
    print("\n" + "=" * 80)
    print("STEP 5: Verify Python Executable Signatures")
    print("=" * 80)
    if not verify_signatures(paths["runtime"]):
        print("\nWarning: Python executables are not properly signed")

    # Step 6: Copy tkinter module
    print("\n" + "=" * 80)
    print("STEP 6: Copy Tkinter Module")
    print("=" * 80)
    if not copy_tkinter_to_distribution(paths["runtime"]):
        print("\nError: Failed to copy tkinter module")
        return 1

    # Step 6a: Remove unnecessary Tcl/Tk files
    if not cleanup_tcl_unnecessary_files(paths["runtime"]):
        print("\nWarning: Failed to cleanup Tcl/Tk files")

    # Step 7: Create python313._pth configuration
    print("\n" + "=" * 80)
    print("STEP 7: Configure Module Search Paths")
    print("=" * 80)
    try:
        pth_paths = [
            "python313.zip",
            "Lib",
            "../Lib/site-packages",
            "../app",
            "import site"
        ]
        create_pth_file(paths["runtime"], pth_paths)
    except Exception as e:
        print(f"\nError creating _pth file: {e}")
        return 1

    # Step 8: Enable pip
    print("\n" + "=" * 80)
    print("STEP 8: Enable Pip in Embedded Python")
    print("=" * 80)
    try:
        enable_pip(paths["runtime"])
    except Exception as e:
        print(f"\nError enabling pip: {e}")
        return 1

    # Step 9: Verify pip
    python_exe = paths["runtime"] / "python.exe"
    if not verify_pip(python_exe):
        print("\nWarning: Pip verification failed")

    # Step 10: Verify tkinter
    if not verify_tkinter(python_exe):
        print("\nWarning: Tkinter verification failed")

    # Step 11: Collect third-party licenses
    print("\n" + "=" * 80)
    print("STEP 11: Collect Third-Party Licenses")
    print("=" * 80)
    if not collect_third_party_licenses(project_root):
        print("\nError: License collection failed")
        return 1

    # Step 12: Install dependencies
    print("\n" + "=" * 80)
    print("STEP 12: Install Dependencies")
    print("=" * 80)
    requirements_file = project_root / "requirements.txt"
    if not install_dependencies(python_exe, paths["lib"], requirements_file):
        print("\nError: Dependency installation failed")
        return 1

    # Step 13: Remove pip from distribution
    if not remove_pip_from_distribution(paths["runtime"], paths["lib"]):
        print("\nWarning: Failed to remove pip from distribution")

    # Step 14: Remove test directories
    if not remove_tests_from_distribution(paths["lib"]):
        print("\nWarning: Failed to remove test directories")

    # Step 15: Verify native libraries
    print("\n" + "=" * 80)
    print("STEP 15: Verify Native Libraries")
    print("=" * 80)
    lib_results = verify_native_libraries(paths["lib"])
    if not all(lib_results.values()):
        print("\nWarning: Some native libraries missing")

    # Step 16: Test critical imports
    print("\n" + "=" * 80)
    print("STEP 16: Test Critical Imports")
    print("=" * 80)
    critical_modules = ["numpy", "onnxruntime", "sounddevice", "onnx_asr", "tkinter"]
    import_results = test_imports(python_exe, critical_modules)
    failed_imports = [m for m, success in import_results.items() if not success]

    if failed_imports:
        print(f"\nWarning: Failed to import: {', '.join(failed_imports)}")

    # Step 17: Copy application code
    print("\n" + "=" * 80)
    print("STEP 17: Copy Application Code")
    print("=" * 80)
    src_dir = project_root / "src"
    main_py = project_root / "main.py"
    if not copy_application_code(src_dir, main_py, paths["app"]):
        print("\nError: Failed to copy application code")
        return 1

    # Step 18: Compile application to bytecode
    print("\n" + "=" * 80)
    print("STEP 18: Compile Application to Bytecode")
    print("=" * 80)
    if not compile_to_pyc(python_exe, paths["app"]):
        print("\nError: Failed to compile code to bytecode")
        return 1

    # Step 19: Compile third-party packages
    print("\n" + "=" * 80)
    print("STEP 19: Compile Third-Party Packages")
    print("=" * 80)
    if not compile_site_packages(python_exe, paths["lib"]):
        print("\nError: Failed to compile third-party packages")
        return 1

    # Step 20: Package .pyc files into .zip archives
    print("\n" + "=" * 80)
    print("STEP 20: Package .pyc Files into .zip Archives")
    print("=" * 80)
    if not zip_site_packages(paths["lib"]):
        print("\nError: Failed to package .pyc files")
        return 1

    # Step 21: Clean up redundant package metadata
    print("\n" + "=" * 80)
    print("STEP 21: Clean Up Package Metadata")
    print("=" * 80)
    if not cleanup_package_metadata(paths["lib"]):
        print("\nError: Failed to clean package metadata")
        return 1

    # Step 22: Copy assets and configuration
    print("\n" + "=" * 80)
    print("STEP 22: Copy Assets and Configuration")
    print("=" * 80)
    if not copy_assets_and_config(project_root, staging_dir, paths["app"]):
        print("\nError: Failed to copy assets and configuration")
        return 1

    # Step 23: Copy MSIX-specific files
    print("\n" + "=" * 80)
    print("STEP 23: Copy MSIX-Specific Files")
    print("=" * 80)
    if not copy_msix_specific_files(staging_dir, msix_dir, project_root):
        print("\nError: Failed to copy MSIX-specific files")
        return 1

    # Step 24: Copy legal documents
    print("\n" + "=" * 80)
    print("STEP 24: Copy Legal Documents")
    print("=" * 80)
    try:
        copy_legal_documents_msix(project_root, staging_dir)
    except Exception as e:
        print(f"\nError copying legal documents: {e}")
        return 1

    # Step 25: Create Store-specific README
    print("\n" + "=" * 80)
    print("STEP 25: Create Store-Specific README")
    print("=" * 80)
    if not create_store_readme(staging_dir):
        print("\nError: Failed to create README")
        return 1

    # Step 26: Create MSIX package
    print("\n" + "=" * 80)
    print("STEP 26: Create MSIX Package")
    print("=" * 80)
    msix_output = dist_dir / f"{PACKAGE_NAME}_{APP_VERSION}_x64.msix"
    if not create_msix_package(staging_dir, msix_output, sdk_dir):
        print("\nError: Failed to create MSIX package")
        return 1

    # Step 27: Sign MSIX package (optional - skipped for Store submission)
    print("\n" + "=" * 80)
    print("STEP 27: Sign MSIX Package")
    print("=" * 80)
    if SKIP_SIGNING:
        print("  [SKIPPED] Signing disabled for Store submission")
        print("  Microsoft Store will re-sign with trusted certificate")
    else:
        cert_path = msix_dir / "AIStenographer_TestCert.pfx"
        if not sign_msix_package(msix_output, cert_path, sdk_dir):
            print("\nError: Failed to sign MSIX package")
            return 1

    # Build complete - print summary
    print("\n" + "=" * 80)
    print("BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nPackage: {msix_output}")
    print(f"Size: {msix_output.stat().st_size / (1024 * 1024):.1f}MB")
    print(f"Version: {APP_VERSION}")
    print(f"Publisher: {PUBLISHER_NAME}")
    print(f"\nStaging Directory: {staging_dir}")

    if SKIP_SIGNING:
        print(f"\nFor Microsoft Store submission:")
        print(f"  Upload {msix_output.name} to Partner Center")
        print(f"  Microsoft will sign the package during certification")
    else:
        cert_path = msix_dir / "AIStenographer_TestCert.pfx"
        print(f"\nTo install for testing:")
        print(f"  1. Install test certificate:")
        print(f"     Import-Certificate -FilePath \"{cert_path.with_suffix('.cer')}\" -CertStoreLocation \"Cert:\\LocalMachine\\Root\"")
        print(f"  2. Install MSIX package:")
        print(f"     Add-AppxPackage -Path \"{msix_output}\"")
        print(f"  3. Launch from Start Menu: \"AI Stenographer\"")
        print(f"\nNote: Test certificate is for development only.")
        print(f"Microsoft Store will re-sign with trusted certificate during submission.")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
