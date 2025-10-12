"""
File watcher that automatically rebuilds when Python files change.

Watches for changes to:
- src/*.py
- main.py
- config/*
- stenographer.jpg

When changes detected, runs rebuild_quick.py automatically.

Usage:
    python watch_and_rebuild.py

Press Ctrl+C to stop watching.
"""
import sys
import time
from pathlib import Path
from datetime import datetime
import subprocess


def get_file_mtimes(project_root: Path) -> dict:
    """Get modification times of all watched files."""
    watched = {}

    # Watch src/*.py
    src_dir = project_root / "src"
    if src_dir.exists():
        for py_file in src_dir.glob("*.py"):
            watched[py_file] = py_file.stat().st_mtime

    # Watch main.py
    main_py = project_root / "main.py"
    if main_py.exists():
        watched[main_py] = main_py.stat().st_mtime

    # Watch config/*
    config_dir = project_root / "config"
    if config_dir.exists():
        for config_file in config_dir.glob("*"):
            if config_file.is_file():
                watched[config_file] = config_file.stat().st_mtime

    # Watch stenographer.jpg
    stenographer = project_root / "stenographer.jpg"
    if stenographer.exists():
        watched[stenographer] = stenographer.stat().st_mtime

    # Watch legal documents
    legal_files = ["LICENSE.txt", "EULA.txt", "THIRD_PARTY_NOTICES.txt"]
    for legal_file in legal_files:
        legal_path = project_root / legal_file
        if legal_path.exists():
            watched[legal_path] = legal_path.stat().st_mtime

    return watched


def watch_and_rebuild():
    """Watch files and rebuild on changes."""
    project_root = Path(__file__).parent

    print("=" * 60)
    print("File Watcher - Auto Rebuild")
    print("=" * 60)
    print("\nWatching for changes to:")
    print("  - src/*.py")
    print("  - main.py")
    print("  - config/*")
    print("  - stenographer.jpg")
    print("  - LICENSE.txt, EULA.txt, THIRD_PARTY_NOTICES.txt")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)

    last_mtimes = get_file_mtimes(project_root)
    print(f"\nWatching {len(last_mtimes)} files...")

    rebuild_script = project_root / "rebuild_quick.py"
    if not rebuild_script.exists():
        print(f"\nError: {rebuild_script} not found!")
        return 1

    try:
        while True:
            time.sleep(1)  # Check every second

            current_mtimes = get_file_mtimes(project_root)

            # Check for changes
            changed_files = []

            # Check modified files
            for file_path, mtime in current_mtimes.items():
                if file_path not in last_mtimes or last_mtimes[file_path] != mtime:
                    changed_files.append(file_path)

            # Check deleted files
            for file_path in last_mtimes:
                if file_path not in current_mtimes:
                    changed_files.append(file_path)

            if changed_files:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] Changes detected:")
                for file_path in changed_files:
                    print(f"  - {file_path.name}")

                print("\nRunning quick rebuild...")
                result = subprocess.run(
                    [sys.executable, str(rebuild_script)],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    print("✓ Rebuild successful")
                else:
                    print("✗ Rebuild failed:")
                    print(result.stdout)
                    print(result.stderr)

                last_mtimes = current_mtimes
                print(f"\nContinuing to watch {len(last_mtimes)} files...")

    except KeyboardInterrupt:
        print("\n\nStopped watching files")
        return 0


if __name__ == "__main__":
    sys.exit(watch_and_rebuild())
