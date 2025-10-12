"""
Quick rebuild script for development.

Use this during development to update distribution without re-downloading
Python or reinstalling dependencies.

Only updates:
- Application code (recompiles .py -> .pyc)
- Config files
- Assets

Usage:
    python rebuild_quick.py
"""
import sys
import shutil
from pathlib import Path
import subprocess


def quick_rebuild():
    """Quick rebuild - only updates application code and assets."""
    print("=" * 60)
    print("Quick Rebuild - Application Code Only")
    print("=" * 60)

    project_root = Path(__file__).parent
    build_dir = project_root / "dist" / "STT-Stenographer"

    # Check if build exists
    if not build_dir.exists():
        print("Error: Build directory doesn't exist!")
        print(f"Expected: {build_dir}")
        print("\nRun 'python build_distribution.py' first for full build")
        return 1

    internal_dir = build_dir / "_internal"
    app_dir = internal_dir / "app"
    runtime_dir = internal_dir / "runtime"
    python_exe = runtime_dir / "python.exe"

    # Step 1: Remove old compiled code
    print("\n1. Removing old compiled code...")
    if app_dir.exists():
        shutil.rmtree(app_dir)
    app_dir.mkdir(parents=True)
    (app_dir / "src").mkdir()
    (app_dir / "config").mkdir()
    (app_dir / "assets").mkdir()
    print(f"   Cleaned: {app_dir}")

    # Step 2: Copy source code
    print("\n2. Copying source code...")
    src_dir = project_root / "src"
    dest_src_dir = app_dir / "src"

    copied = 0
    for py_file in src_dir.glob("*.py"):
        shutil.copy2(py_file, dest_src_dir / py_file.name)
        copied += 1

    # Copy main.py
    main_py = project_root / "main.py"
    if main_py.exists():
        shutil.copy2(main_py, app_dir / "main.py")
        copied += 1

    print(f"   Copied {copied} Python files")

    # Step 3: Compile to bytecode
    print("\n3. Compiling to bytecode...")
    result = subprocess.run(
        [str(python_exe), "-m", "compileall", "-b", "-f", str(app_dir)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"   Error compiling: {result.stderr}")
        return 1

    # Count compiled files
    pyc_files = list(app_dir.rglob("*.pyc"))
    print(f"   Compiled {len(pyc_files)} .pyc files")

    # Step 4: Remove source files
    print("\n4. Removing source files...")
    removed = 0
    for py_file in app_dir.rglob("*.py"):
        py_file.unlink()
        removed += 1
    print(f"   Removed {removed} .py files")

    # Step 5: Copy config
    print("\n5. Copying config files...")
    config_src = project_root / "config"
    config_dest = app_dir / "config"

    if config_src.exists():
        for config_file in config_src.glob("*"):
            if config_file.is_file():
                shutil.copy2(config_file, config_dest / config_file.name)
        print(f"   Copied config files")

    # Step 6: Copy assets
    print("\n6. Copying assets...")
    assets_dest = app_dir / "assets"

    stenographer_jpg = project_root / "stenographer.jpg"
    if stenographer_jpg.exists():
        shutil.copy2(stenographer_jpg, assets_dest / "stenographer.jpg")
        print(f"   Copied stenographer.jpg")

    # Step 7: Copy legal documents to root
    print("\n7. Copying legal documents...")
    legal_files = ["LICENSE.txt", "EULA.txt", "THIRD_PARTY_NOTICES.txt"]
    legal_copied = 0
    for legal_file in legal_files:
        src = project_root / legal_file
        if src.exists():
            shutil.copy2(src, build_dir / legal_file)
            legal_copied += 1
    print(f"   Copied {legal_copied} legal documents to root")

    # Step 8: Copy LICENSES/ folder to root
    licenses_src = project_root / "LICENSES"
    licenses_dest = build_dir / "LICENSES"
    if licenses_src.exists():
        if licenses_dest.exists():
            shutil.rmtree(licenses_dest)
        shutil.copytree(licenses_src, licenses_dest)
        license_count = len(list(licenses_dest.glob("*/")))
        print(f"   Copied LICENSES/ folder ({license_count} packages)")

    print("\n" + "=" * 60)
    print("Quick rebuild completed!")
    print(f"Build directory: {build_dir}")
    print("\nYou can now test with:")
    print(f'  cd "{build_dir}"')
    print('  _internal\\runtime\\pythonw.exe _internal\\app\\main.pyc')
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(quick_rebuild())
