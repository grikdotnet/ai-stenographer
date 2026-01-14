#!/usr/bin/env python3
"""
Build script for AI Stenographer native launcher.
Compiles AIStenographer.cpp to AIStenographer.exe using MSVC.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """
    Compiles the native launcher using MSVC.

    Algorithm:
    1. Set up MSVC and Windows SDK paths
    2. Configure compiler environment
    3. Invoke cl.exe to compile the C++ source
    4. Clean up intermediate files

    Returns:
        0 on success, 1 on failure
    """
    print("=" * 60)
    print("AI Stenographer Native Launcher Build Script")
    print("=" * 60)
    print()

    # Configuration
    msvc_root = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207")
    msvc_bin = msvc_root / "bin/Hostx64/x64"
    msvc_include = msvc_root / "include"
    msvc_lib = msvc_root / "lib/x64"

    sdk_root = Path("C:/Program Files (x86)/Windows Kits/10")
    sdk_version = "10.0.26100.0"
    sdk_include = sdk_root / f"Include/{sdk_version}"
    sdk_lib = sdk_root / f"Lib/{sdk_version}"

    # Verify compiler exists
    cl_exe = msvc_bin / "cl.exe"
    if not cl_exe.exists():
        print(f"ERROR: MSVC compiler not found at: {cl_exe}")
        print("Please verify Visual Studio Build Tools 2022 is installed")
        return 1

    print("[1/3] Setting up build environment...")

    # Build environment variables
    import os
    env = os.environ.copy()
    env.update({
        "PATH": f"{msvc_bin};{env.get('PATH', '')}",
        "INCLUDE": f"{msvc_include};{sdk_include}/ucrt;{sdk_include}/shared;{sdk_include}/um",
        "LIB": f"{msvc_lib};{sdk_lib}/ucrt/x64;{sdk_lib}/um/x64",
        "TMP": str(Path.home() / "AppData/Local/Temp"),
        "TEMP": str(Path.home() / "AppData/Local/Temp")
    })

    print("[2/3] Compiling AIStenographer.cpp...")

    # Compiler command
    cmd = [
        str(cl_exe),
        "/nologo",           # Suppress copyright message
        "/EHsc",            # Exception handling model
        "/O2",              # Maximum optimization
        "/W3",              # Warning level 3
        "/D", "UNICODE",    # Unicode build
        "/D", "_UNICODE",   # Unicode build
        "/Fe:AIStenographer.exe",  # Output filename
        "AIStenographer.cpp",
        "/link",            # Linker options follow
        "/SUBSYSTEM:WINDOWS",  # Windows GUI application
        "shlwapi.lib",      # Shell API library
        "user32.lib"        # User interface library (MessageBox)
    ]

    # Run compilation
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True
        )

        # Print compiler output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print()
            print(f"ERROR: Compilation failed with error code {result.returncode}")
            return result.returncode

    except Exception as e:
        print(f"ERROR: Failed to run compiler: {e}")
        return 1

    print("[3/3] Cleaning up intermediate files...")

    # Clean up object file
    obj_file = Path(__file__).parent / "AIStenographer.obj"
    if obj_file.exists():
        obj_file.unlink()

    print()
    print("=" * 60)
    print("Build completed successfully!")
    print("Output: AIStenographer.exe")
    print("=" * 60)

    # Display file info
    exe_file = Path(__file__).parent / "AIStenographer.exe"
    if exe_file.exists():
        size = exe_file.stat().st_size
        print(f"File size: {size:,} bytes ({size / 1024:.1f} KB)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
