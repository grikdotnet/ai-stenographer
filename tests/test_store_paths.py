"""
Test script to verify Store path resolution logic.

This script simulates different environments and displays the resolved paths.
"""
import sys
import os
from pathlib import Path

# Import PathResolver from src (one level up from tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.PathResolver import PathResolver


def __test_environment(name: str, env_vars: dict, script_path: Path, set_frozen: bool = False):
    """Test path resolution in a specific environment."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    # Set environment variables (skip 'frozen' which is not an env var)
    original_env = {}
    for key, value in env_vars.items():
        if key == 'frozen':
            continue  # Handle frozen separately
        original_env[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    # Mock sys.frozen if needed
    original_frozen = getattr(sys, 'frozen', None)
    if set_frozen:
        sys.frozen = True

    try:
        resolver = PathResolver(script_path)
        paths = resolver.paths

        print(f"\nEnvironment: {paths.environment}")
        print(f"\nResolved Paths:")
        print(f"  {'app_dir':15} = {paths.app_dir}")
        print(f"  {'internal_dir':15} = {paths.internal_dir}")
        print(f"  {'root_dir':15} = {paths.root_dir}")
        print(f"  {'models_dir':15} = {paths.models_dir}")
        print(f"  {'config_dir':15} = {paths.config_dir}")
        print(f"  {'assets_dir':15} = {paths.assets_dir}")
        print(f"  {'logs_dir':15} = {paths.logs_dir}")

        # Check if paths are writable
        print(f"\nWritability Check:")
        for key in ['models_dir', 'config_dir', 'logs_dir']:
            path = getattr(paths, key)
            try:
                path.mkdir(parents=True, exist_ok=True)
                test_file = path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                writable = "✓ Writable"
            except Exception as e:
                writable = f"✗ Not writable: {e}"
            print(f"  {key:15} = {writable}")

    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        # Restore sys.frozen
        if original_frozen is None and hasattr(sys, 'frozen'):
            delattr(sys, 'frozen')
        elif original_frozen is not None:
            sys.frozen = original_frozen


def main():
    """Run tests for all environments."""
    print("Path Resolution Test Suite")
    print("="*60)

    # Test 1: Development mode (current environment)
    _test_environment(
        "Development Mode",
        {},
        Path(__file__).parent / "main.py",
        set_frozen=False
    )

    # Test 2: Portable distribution mode
    portable_path = Path("C:/AI-Stenographer/_internal/app/main.pyc")
    _test_environment(
        "Portable Distribution Mode",
        {},
        portable_path,
        set_frozen=True
    )

    # Test 3: Microsoft Store mode
    store_path = Path("C:/Program Files/WindowsApps/AI.Stenographer_1.0.0.0_x64/_internal/app/main.pyc")
    _test_environment(
        "Microsoft Store Mode (MSIX)",
        {
            'MSIX_PACKAGE_IDENTITY': 'AI.Stenographer_1.0.0.0_x64__8wekyb3d8bbwe',
            'LOCALAPPDATA': str(Path.home() / "AppData" / "Local"),
        },
        store_path,
        set_frozen=True
    )

    print(f"\n{'='*60}")
    print("Test suite completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
