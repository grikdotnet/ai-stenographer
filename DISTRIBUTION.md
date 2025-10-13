# Distribution Guide

This guide explains how to package and distribute the Speech-to-Text application as a pre-compiled Python archive with full license compliance.

## Table of Contents

1. [Quick Start](#quick-start)
2. [License Compliance](#license-compliance)
3. [Packaging Methods](#packaging-methods)
4. [Distribution Structure](#distribution-structure)
5. [Pre-compilation with PyInstaller](#pre-compilation-with-pyinstaller)
6. [ZIP Archive Distribution](#zip-archive-distribution)
7. [Testing the Distribution](#testing-the-distribution)

---

## Quick Start

### Step 1: Collect Licenses

```bash
source venv/Scripts/activate
python scripts/collect_licenses.py
```

This creates:
- `LICENSES/` - Directory with all third-party license files
- `THIRD_PARTY_NOTICES.txt` - Attribution and compliance information

### Step 2: Create Distribution Package

Choose one of the methods below:
- **Method A:** PyInstaller (single executable)
- **Method B:** ZIP archive (pre-compiled .pyc files)
- **Method C:** Wheel distribution

---

## License Compliance

### What Must Be Included

✅ **Always include in your distribution:**

1. **LICENSES/** directory with all license files
2. **THIRD_PARTY_NOTICES.txt** with attribution
3. **README.txt** explaining license compliance
4. **About dialog or splash screen** showing attributions

### Package License Summary

All dependencies use permissive licenses that allow proprietary distribution:

| License Type | Packages | Requirements |
|--------------|----------|--------------|
| **MIT** | sounddevice, onnx-asr, onnxruntime, pytest, PyYAML, coloredlogs | Include license text |
| **BSD** | numpy, soundfile, scipy, PyTorch, torchaudio | Include copyright notice |
| **Apache 2.0** | huggingface_hub, requests | Include license + NOTICE files |
| **PSF** | Python, tkinter | Include Python license |
| **LGPL** | libsndfile (via soundfile) | Dynamic linking + allow replacement |
| **MPL 2.0** | tqdm (partial) | Include license for MPL files |

### LGPL Compliance (libsndfile)

The soundfile package depends on libsndfile (LGPL). To comply:

✅ **Automatic compliance:**
- soundfile uses **dynamic linking** via ctypes (compliant by default)
- No special action needed for basic distribution

✅ **Documentation required:**
- State that libsndfile is included (in THIRD_PARTY_NOTICES.txt)
- Provide link to libsndfile source: https://github.com/libsndfile/libsndfile
- Note that users can replace libsndfile if desired

❌ **Do not:**
- Statically link libsndfile without providing source
- Remove LGPL license from distribution
- Prevent users from replacing the library

---

## Packaging Methods

### Method A: PyInstaller (Recommended for End Users)

Creates a single executable or directory with all dependencies.

**Advantages:**
- Easy distribution (single .exe file)
- No Python installation required
- Automatic dependency bundling
- Built-in license collection

**Installation:**

```bash
pip install pyinstaller
```

**Create executable:**

```bash
# Single file (larger, slower startup)
pyinstaller --onefile --name STT_App main.py

# Directory bundle (faster, smaller executable)
pyinstaller --onedir --name STT_App main.py
```

**Include licenses:**

```bash
pyinstaller --onedir --name STT_App \
  --add-data "LICENSES;LICENSES" \
  --add-data "THIRD_PARTY_NOTICES.txt;." \
  --add-data "config;config" \
  --add-data "models;models" \
  main.py
```

**Output:** `dist/STT_App/` directory or `dist/STT_App.exe` file

### Method B: ZIP Archive with Pre-compiled Python

Creates a .zip with pre-compiled .pyc files and libraries.

**Advantages:**
- Smaller size than PyInstaller
- Faster startup
- Source code protection via .pyc files
- Easy updates (replace .pyc files)

**Steps:**

See [ZIP Archive Distribution](#zip-archive-distribution) section below.

### Method C: Python Wheel (.whl)

Creates an installable Python package.

**Advantages:**
- Standard Python distribution format
- Works with pip
- Easy version management

**Disadvantages:**
- Requires Python installation on target machine
- Source code visible

**Create wheel:**

```bash
pip install wheel
python setup.py bdist_wheel
```

---

## Distribution Structure

### PyInstaller Output Structure

```
STT_App/
├── STT_App.exe              # Main executable
├── _internal/               # Dependencies
│   ├── numpy/
│   ├── torch/
│   └── ...
├── LICENSES/                # Third-party licenses
│   ├── numpy/
│   │   └── LICENSE.txt
│   ├── sounddevice/
│   │   └── LICENSE.txt
│   └── ...
├── THIRD_PARTY_NOTICES.txt  # Attribution file
├── config/                  # Configuration files
│   └── stt_config.json
├── models/                  # ML models (or download on first run)
│   ├── parakeet/
│   └── silero_vad/
└── README.txt               # User documentation
```

### ZIP Archive Structure

```
stt_app_v1.0.zip
├── app/
│   ├── main.pyc             # Pre-compiled main script
│   ├── src/
│   │   ├── __pycache__/     # Pre-compiled modules
│   │   ├── pipeline.pyc
│   │   ├── AudioSource.pyc
│   │   └── ...
│   ├── config/
│   │   └── stt_config.json
│   └── requirements.txt
├── lib/                     # Bundled libraries
│   ├── numpy/
│   ├── torch/
│   └── ...
├── python/                  # Python runtime (optional)
│   ├── python.exe
│   ├── python311.dll
│   └── Lib/
├── LICENSES/                # All license files
├── THIRD_PARTY_NOTICES.txt
├── README.txt               # Installation instructions
└── run.bat                  # Launch script
```

---

## Pre-compilation with PyInstaller

### Complete PyInstaller Script

Create `build_dist.py`:

```python
"""Build distribution package with PyInstaller."""
import PyInstaller.__main__
import sys

PyInstaller.__main__.run([
    'main.py',
    '--name=STT_App',
    '--onedir',                          # Directory bundle (faster)
    '--windowed',                        # No console window (optional)
    '--icon=icon.ico',                   # Application icon

    # Add data files
    '--add-data=LICENSES;LICENSES',
    '--add-data=THIRD_PARTY_NOTICES.txt;.',
    '--add-data=config;config',
    '--add-data=models;models',          # If models are pre-downloaded

    # Optimize
    '--noconfirm',                       # Overwrite output without asking
    '--clean',                           # Clean cache before build

    # Hidden imports (if needed)
    '--hidden-import=numpy',
    '--hidden-import=onnxruntime',
    '--hidden-import=sounddevice',

    # Exclude unnecessary packages
    '--exclude-module=pytest',
    '--exclude-module=tkinter.test',
])
```

Run:

```bash
python build_dist.py
```

### Testing PyInstaller Build

```bash
cd dist/STT_App
./STT_App.exe
```

---

## ZIP Archive Distribution

### Step-by-Step ZIP Archive Creation

#### 1. Pre-compile Python Files

```bash
# Activate virtual environment
source venv/Scripts/activate

# Compile all .py files to .pyc
python -m compileall -b src/
python -m compileall -b main.py

# This creates .pyc files alongside .py files
# -b flag creates .pyc files with same name (removes .py)
```

#### 2. Create Distribution Directory

```bash
mkdir -p dist/stt_app
mkdir -p dist/stt_app/app
mkdir -p dist/stt_app/lib
```

#### 3. Copy Pre-compiled Files

```bash
# Copy compiled application code
cp -r src/__pycache__/*.pyc dist/stt_app/app/src/
cp main.pyc dist/stt_app/app/

# Copy configuration
cp -r config dist/stt_app/app/

# Copy licenses
cp -r LICENSES dist/stt_app/
cp THIRD_PARTY_NOTICES.txt dist/stt_app/

# Copy models (optional - or download on first run)
cp -r models dist/stt_app/app/
```

#### 4. Bundle Python Libraries

**Option A: Copy from venv (smaller)**

```bash
# Copy only runtime dependencies
python scripts/bundle_dependencies.py dist/stt_app/lib
```

**Option B: Use pip download (portable)**

```bash
pip download -r requirements.txt -d dist/stt_app/lib --no-binary :all:
```

#### 5. Create Launcher Script

Create `dist/stt_app/run.bat`:

```batch
@echo off
REM Speech-to-Text Application Launcher

REM Set Python path (assumes Python is in PATH)
SET PYTHONPATH=%~dp0lib;%~dp0app;%PYTHONPATH%

REM Run application
python app\main.pyc %*

pause
```

Create `dist/stt_app/run.sh` (for Linux/Mac):

```bash
#!/bin/bash
# Speech-to-Text Application Launcher

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set Python path
export PYTHONPATH="$DIR/lib:$DIR/app:$PYTHONPATH"

# Run application
python3 "$DIR/app/main.pyc" "$@"
```

#### 6. Create README for Users

Create `dist/stt_app/README.txt`:

```
Speech-to-Text Application
==========================

INSTALLATION
------------

1. Requirements:
   - Python 3.9 or later
   - Windows/Linux/macOS

2. Installation:
   - Extract this ZIP archive
   - No additional installation required

USAGE
-----

Windows:
   Double-click run.bat

   Or from command line:
   run.bat

Linux/macOS:
   chmod +x run.sh
   ./run.sh

   Or:
   python3 app/main.pyc

FIRST RUN
---------

On first run, the application will download required ML models (~2GB):
- Parakeet STT Model (~1.5GB)
- Silero VAD Model (~0.2GB)

This is a one-time download. Models are cached in ./models/

COMMAND LINE OPTIONS
--------------------

Verbose mode:
   run.bat -v

Custom windowing:
   run.bat --window=2.0 --step=1.0

LICENSE & ATTRIBUTION
---------------------

This software includes third-party libraries with their own licenses.
See THIRD_PARTY_NOTICES.txt and LICENSES/ directory for details.

SUPPORT
-------

For issues or questions, visit:
https://github.com/[your-repo]/issues
```

#### 7. Create ZIP Archive

```bash
cd dist
zip -r stt_app_v1.0.zip stt_app/

# Or on Windows (PowerShell):
# Compress-Archive -Path stt_app -DestinationPath stt_app_v1.0.zip
```

### Helper Script for ZIP Creation

Create `scripts/create_zip_dist.py`:

```python
"""Create ZIP archive distribution with pre-compiled Python."""
import os
import shutil
import zipfile
import compileall
from pathlib import Path


class ZipDistCreator:
    """Creates ZIP archive distribution."""

    def __init__(self, output_dir='dist/stt_app'):
        self.output_dir = Path(output_dir)
        self.app_dir = self.output_dir / 'app'
        self.lib_dir = self.output_dir / 'lib'

    def clean(self):
        """Remove old distribution."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compile_python_files(self):
        """Compile all Python files to .pyc."""
        print("Compiling Python files...")
        compileall.compile_dir('src', force=True, quiet=1, legacy=True)
        compileall.compile_file('main.py', force=True, quiet=1, legacy=True)

    def copy_application(self):
        """Copy compiled application files."""
        print("Copying application files...")

        # Create app directory structure
        self.app_dir.mkdir(parents=True, exist_ok=True)

        # Copy main.pyc
        shutil.copy2('main.pyc', self.app_dir / 'main.pyc')

        # Copy src/ with only .pyc files
        src_dest = self.app_dir / 'src'
        src_dest.mkdir(exist_ok=True)

        for pyc_file in Path('src').rglob('*.pyc'):
            rel_path = pyc_file.relative_to('src')
            dest_file = src_dest / rel_path.parent / pyc_file.name
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pyc_file, dest_file)

        # Copy config
        shutil.copytree('config', self.app_dir / 'config', dirs_exist_ok=True)

        # Copy models if they exist
        if Path('models').exists():
            shutil.copytree('models', self.app_dir / 'models', dirs_exist_ok=True)

    def copy_libraries(self):
        """Copy required libraries from venv."""
        print("Copying libraries...")
        # This would need pip or venv introspection
        # For now, we'll just note that libraries should be installed
        # on the target system or bundled separately

    def copy_licenses(self):
        """Copy license files."""
        print("Copying licenses...")
        shutil.copytree('LICENSES', self.output_dir / 'LICENSES', dirs_exist_ok=True)
        shutil.copy2('THIRD_PARTY_NOTICES.txt', self.output_dir / 'THIRD_PARTY_NOTICES.txt')

    def create_launcher(self):
        """Create launcher scripts."""
        print("Creating launcher scripts...")

        # Windows batch file
        bat_content = """@echo off
REM Speech-to-Text Application Launcher

SET PYTHONPATH=%~dp0lib;%~dp0app;%PYTHONPATH%
python app\\main.pyc %*
pause
"""
        (self.output_dir / 'run.bat').write_text(bat_content)

        # Linux/Mac shell script
        sh_content = """#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$DIR/lib:$DIR/app:$PYTHONPATH"
python3 "$DIR/app/main.pyc" "$@"
"""
        sh_file = self.output_dir / 'run.sh'
        sh_file.write_text(sh_content)
        os.chmod(sh_file, 0o755)

    def create_readme(self):
        """Create user README."""
        readme = """Speech-to-Text Application
==========================

See README.txt for full installation and usage instructions.
See THIRD_PARTY_NOTICES.txt for license information.
"""
        (self.output_dir / 'README.txt').write_text(readme)

    def create_zip(self):
        """Create final ZIP archive."""
        print("Creating ZIP archive...")

        zip_path = self.output_dir.parent / f"{self.output_dir.name}.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in self.output_dir.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(self.output_dir.parent)
                    zipf.write(file, arcname)

        print(f"\n✓ ZIP created: {zip_path}")
        return zip_path

    def build(self):
        """Build complete distribution."""
        print("\n" + "="*70)
        print("CREATING ZIP DISTRIBUTION")
        print("="*70 + "\n")

        self.clean()
        self.compile_python_files()
        self.copy_application()
        self.copy_licenses()
        self.create_launcher()
        self.create_readme()
        zip_path = self.create_zip()

        print("\n" + "="*70)
        print("DISTRIBUTION COMPLETE")
        print("="*70)
        print(f"\nOutput: {zip_path}")
        print(f"Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    creator = ZipDistCreator()
    creator.build()
```

---

## Testing the Distribution

### Test Checklist

✅ **Before release:**

1. **Extract archive to clean directory**
   ```bash
   mkdir test_install
   cd test_install
   unzip ../stt_app_v1.0.zip
   ```

2. **Run application**
   ```bash
   ./run.bat  # or ./run.sh
   ```

3. **Verify license compliance**
   - Check LICENSES/ directory exists
   - Check THIRD_PARTY_NOTICES.txt exists
   - Verify all packages are listed

4. **Test functionality**
   - Model download works
   - Audio capture works
   - Speech recognition works
   - GUI displays correctly

5. **Test on clean system**
   - Fresh Windows VM (no Python installed)
   - Fresh Linux VM
   - Verify all dependencies included

### Automated Testing

Create `scripts/test_distribution.py`:

```python
"""Test distribution package."""
import os
import zipfile
import subprocess
from pathlib import Path


def test_zip_structure(zip_path):
    """Test ZIP archive structure."""
    required_files = [
        'stt_app/LICENSES/',
        'stt_app/THIRD_PARTY_NOTICES.txt',
        'stt_app/app/main.pyc',
        'stt_app/run.bat',
    ]

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for required in required_files:
            assert any(required in name for name in names), f"Missing: {required}"

    print("✓ ZIP structure valid")


def test_licenses_complete():
    """Test that all licenses are collected."""
    licenses_dir = Path('LICENSES')
    required_packages = [
        'numpy', 'sounddevice', 'torch', 'onnxruntime',
        'pillow', 'scipy', 'requests', 'pyyaml'
    ]

    for package in required_packages:
        # Check for license file
        package_dir = licenses_dir / package
        license_file = package_dir / 'LICENSE.txt'
        assert license_file.exists(), f"Missing license for: {package}"

    print("✓ All licenses present")


if __name__ == '__main__':
    test_licenses_complete()
    # test_zip_structure('dist/stt_app_v1.0.zip')
    print("\n✓ All tests passed")
```

---

## Summary

### Recommended Approach

For most use cases, **PyInstaller** is recommended:

```bash
# 1. Collect licenses
python scripts/collect_licenses.py

# 2. Build with PyInstaller
pyinstaller --onedir --name STT_App \
  --add-data "LICENSES;LICENSES" \
  --add-data "THIRD_PARTY_NOTICES.txt;." \
  --add-data "config;config" \
  main.py

# 3. Test
cd dist/STT_App
./STT_App.exe

# 4. Package
cd ..
zip -r STT_App_v1.0.zip STT_App/
```

### File Sizes (Approximate)

- **PyInstaller single file:** ~800MB (slow startup)
- **PyInstaller directory:** ~600MB (fast startup)
- **ZIP with .pyc:** ~400MB (requires Python)
- **Source + requirements.txt:** ~50KB (requires pip install)

### License Compliance Summary

✅ **Always include:**
- LICENSES/ directory
- THIRD_PARTY_NOTICES.txt
- Attribution in About dialog

✅ **LGPL compliance (libsndfile):**
- Already compliant (dynamic linking)
- Just document usage in NOTICES

✅ **No source code disclosure required:**
- All licenses are permissive
- Keep your code proprietary
- Only include third-party notices

---

For questions or issues, refer to:
- Python Packaging Guide: https://packaging.python.org/
- PyInstaller Manual: https://pyinstaller.org/
- License texts in LICENSES/ directory
