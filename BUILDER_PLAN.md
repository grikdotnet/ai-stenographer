# Windows Distribution Build Plan (TDD)

## Goal
Create a portable Windows distribution of the STT application using Python embeddable package with pre-compiled bytecode, requiring no Python installation on user machines.

## Distribution Strategy

**Approach:** Embedded Python + Pre-compiled Bytecode
- **Embedded Python** (signed executables) - no installation needed
- **Pre-compiled bytecode** (.pyc files) - source code hidden
- **Pre-installed dependencies** - ready to run
- **Custom icon launcher** - professional appearance with icon.ico
- **Runtime model downloads** - models downloaded on first run
- **Clean root directory** - only user-facing files visible

## Final Distribution Structure

```
STT-Stenographer/                    # ROOT - USER FILES ONLY
├── STT - Stenographer.lnk           # Main launcher (uses icon.ico)
├── README.txt                       # User instructions (to be created)
├── LICENSE.txt                      # Software license (to be created)
├── icon.ico                         # ✓ Already exists (44KB, 6 icons)
│
└── _internal/                       # ALL TECHNICAL FILES HIDDEN
    ├── runtime/                     # Python runtime
    │   ├── python.exe               # ✓ Signed by PSF
    │   ├── pythonw.exe              # ✓ Signed by PSF (GUI mode, no console)
    │   ├── python313.dll            # ✓ Signed
    │   ├── python313.zip            # Standard library (imports from ZIP)
    │   ├── python313._pth           # Path configuration
    │   ├── _tkinter.pyd             # Tkinter binary (copied from system Python)
    │   ├── tk86t.dll                # ✓ Tcl/Tk DLL (included in embeddable package)
    │   ├── tcl86t.dll               # ✓ Tcl/Tk DLL (included in embeddable package)
    │   └── Lib/                     # Additional modules (tkinter)
    │       └── tkinter/             # Copied from system Python (~2MB)
    │
    ├── Lib/                         # Third-party dependencies
    │   └── site-packages/           # Pre-installed (numpy, onnxruntime, etc.)
    │
    ├── app/                         # YOUR APPLICATION (compiled .pyc only)
    │   ├── src/                     # Compiled bytecode
    │   │   ├── pipeline.pyc
    │   │   ├── Recognizer.pyc
    │   │   ├── VoiceActivityDetector.pyc
    │   │   ├── AudioSource.pyc
    │   │   ├── AdaptiveWindower.pyc
    │   │   ├── TextMatcher.pyc
    │   │   ├── TextNormalizer.pyc
    │   │   ├── GuiWindow.pyc
    │   │   ├── LoadingWindow.pyc
    │   │   ├── ModelManager.pyc
    │   │   ├── ModelDownloadDialog.pyc
    │   │   ├── GuiFactory.pyc
    │   │   ├── types.pyc
    │   │   └── __init__.pyc
    │   ├── config/
    │   │   └── stt_config.json
    │   ├── assets/
    │   │   └── stenographer.jpg     # ✓ Already exists (96KB)
    │   └── main.pyc                 # Compiled entry point
    │
    └── models/                      # AI models (downloaded at runtime)
        └── .gitkeep
```

**Distribution Size:**
- Compressed ZIP: ~122MB (includes tkinter)
- Extracted: ~182MB
- After first run (with models): ~2.2GB

## Implementation Steps (Test-Driven Development)

### Step 1: Create Build Script Foundation
**File:** `build_distribution.py`

**Tests to write:** `tests/test_build_distribution.py`
1. `test_download_embedded_python()` - Downloads Python 3.13.x embeddable package
2. `test_extract_embedded_python()` - Extracts to `_internal/runtime/`
3. `test_create_directory_structure()` - Creates all required folders
4. `test_verify_signed_executables()` - Confirms python.exe/pythonw.exe are signed

**Functions to implement:**
```python
def download_embedded_python(version: str, cache_dir: Path) -> Path
def extract_embedded_python(zip_path: Path, target_dir: Path) -> None
def create_directory_structure(root_dir: Path) -> dict[str, Path]
def verify_signatures(runtime_dir: Path) -> bool
```

---

### Step 2: Configure Embedded Python
**File:** `build_distribution.py` (continued)

**Tests to write:**
1. `test_create_pth_file()` - Creates correct python313._pth
2. `test_enable_pip()` - Enables pip in embedded Python
3. `test_pip_available()` - Verifies pip works

**Path configuration:** `_internal/runtime/python313._pth`
```
python313.zip
Lib
../Lib/site-packages
../app
import site
```

**Note:** `Lib/` is required for tkinter module (copied in Step 3).

**Functions to implement:**
```python
def create_pth_file(runtime_dir: Path, paths: list[str]) -> None
def enable_pip(runtime_dir: Path) -> None
def verify_pip(python_exe: Path) -> bool
```

---

### Step 3: Copy tkinter Module to Embedded Python
**File:** `build_distribution.py` (continued)

**Problem:** Python embeddable package does NOT include tkinter in `python313.zip`, even though Tcl/Tk DLLs are present. This causes `ModuleNotFoundError` when importing tkinter.

**Tests to write:**
1. `test_copy_tkinter_to_distribution()` - Copies tkinter from system Python
2. `test_tkinter_pyd_present()` - Verifies `_tkinter.pyd` binary is copied
3. `test_tkinter_importable()` - Tests `python.exe -c "import tkinter"`

**Functions to implement:**
```python
def copy_tkinter_to_distribution(runtime_dir: Path) -> bool
def verify_tkinter(python_exe: Path) -> bool
```

**Implementation details:**
- Copy `tkinter/` module from system Python's `Lib/tkinter/` → `_internal/runtime/Lib/tkinter/`
- Copy `_tkinter.pyd` from system Python's `DLLs/_tkinter.pyd` → `_internal/runtime/_tkinter.pyd`
- Tcl/Tk DLLs (`tk86t.dll`, `tcl86t.dll`) are already present in embeddable package
- Size increase: ~2MB

**Why this is needed:**
- LoadingWindow, ModelDownloadDialog, and GuiWindow all require tkinter
- Application cannot start without tkinter
- This is a **blocking issue** that prevents distribution from running

---

### Step 4: Install Dependencies
**File:** `build_distribution.py` (continued)

**Tests to write:**
1. `test_install_dependencies()` - Installs from requirements.txt
2. `test_verify_native_libs()` - Checks ONNX runtime DLLs, PortAudio
3. `test_dependency_import()` - Tests importing installed packages

**Functions to implement:**
```python
def install_dependencies(python_exe: Path, target_dir: Path, requirements: Path) -> bool
def verify_native_libraries(site_packages: Path) -> dict[str, bool]
def test_imports(python_exe: Path, modules: list[str]) -> dict[str, bool]
```

**Native libraries to verify:**
- `onnxruntime/capi/onnxruntime.dll`
- `onnxruntime/capi/onnxruntime_providers_shared.dll`
- `_sounddevice_data/portaudio*.dll`

---

### Step 5: Copy and Compile Application Code
**File:** `build_distribution.py` (continued)

**Tests to write:**
1. `test_copy_source_files()` - Copies src/ to _internal/app/
2. `test_compile_bytecode()` - Compiles all .py to .pyc (legacy format)
3. `test_remove_source_files()` - Removes all .py files
4. `test_pyc_import()` - Verifies .pyc files are importable without source

**Functions to implement:**
```python
def copy_application_code(src_dir: Path, dest_dir: Path) -> None
def compile_to_bytecode(app_dir: Path, legacy_format: bool = True) -> None
def remove_source_files(app_dir: Path) -> list[Path]
def verify_bytecode_imports(python_exe: Path, app_dir: Path) -> bool
```

**Compilation command:**
```bash
python -m compileall -b -f _internal/app/
# -b: legacy format (pyc alongside source, not __pycache__)
# -f: force recompilation
```

---

### Step 6: Copy Assets and Configuration
**File:** `build_distribution.py` (continued)

**Tests to write:**
1. `test_copy_config_files()` - Copies config/ to _internal/app/config/
2. `test_copy_assets()` - Copies stenographer.jpg to _internal/app/assets/
3. `test_create_models_placeholder()` - Creates empty models/ directory

**Functions to implement:**
```python
def copy_config_files(src_config: Path, dest_config: Path) -> None
def copy_assets(src_assets: list[Path], dest_assets: Path) -> None
def create_models_directory(models_dir: Path) -> None
```

---

### Step 7: Update Application Path Handling
**File:** `src/main.py` (modify before compilation)

**Tests to write:** `tests/test_main_paths.py`
1. `test_paths_from_internal_structure()` - Verifies paths resolve correctly
2. `test_assets_loading()` - Loads stenographer.jpg from correct location
3. `test_config_loading()` - Loads stt_config.json from correct location
4. `test_models_directory()` - Creates models in _internal/models/

**Code to add at top of main.py:**
```python
import sys
from pathlib import Path

# Determine if running from .py or .pyc
if hasattr(sys.modules['__main__'], '__file__'):
    SCRIPT_PATH = Path(sys.modules['__main__'].__file__).resolve()
else:
    SCRIPT_PATH = Path(__file__).resolve()

# Path structure: _internal/app/main.pyc
APP_DIR = SCRIPT_PATH.parent          # _internal/app/
INTERNAL_DIR = APP_DIR.parent         # _internal/
ROOT_DIR = INTERNAL_DIR.parent        # STT-Stenographer/

# Application paths
ASSETS_DIR = APP_DIR / "assets"
CONFIG_DIR = APP_DIR / "config"
MODELS_DIR = INTERNAL_DIR / "models"

# Ensure paths exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
```

**Update all path references:**
- `stenographer.jpg` → `ASSETS_DIR / "stenographer.jpg"`
- `config/stt_config.json` → `CONFIG_DIR / "stt_config.json"`
- `./models/` → `MODELS_DIR`

---

### Step 8: Create Windows Launcher Shortcut
**File:** `build_distribution.py` (continued)

**Tests to write:**
1. `test_create_shortcut()` - Creates .lnk file with correct target
2. `test_shortcut_icon()` - Sets icon to icon.ico
3. `test_shortcut_properties()` - Verifies target uses pythonw.exe

**Shortcut configuration:**
- **Name:** `STT - Stenographer.lnk`
- **Target:** `_internal\runtime\pythonw.exe _internal\app\main.pyc`
- **Start in:** (blank - uses shortcut location)
- **Icon:** `icon.ico`
- **Run:** Minimized (no console window)
- **Comment:** "Speech-to-Text Stenographer"

**Functions to implement:**
```python
def create_launcher_shortcut(
    root_dir: Path,
    python_exe: str,
    script: str,
    icon: Path
) -> Path
```

**Note:** Use `win32com.client` or `pywin32` to create .lnk files programmatically.

---

### Step 9: Create Documentation Files
**Files:** `README.txt`, `LICENSE.txt`

**Tests to write:**
1. `test_create_readme()` - Generates README.txt
2. `test_create_license()` - Generates LICENSE.txt
3. `test_documentation_presence()` - Verifies docs in distribution

**README.txt contents:**
```
STT - Stenographer
==================

Real-time Speech-to-Text application powered by AI.

QUICK START
-----------
1. Double-click "STT - Stenographer" to launch
2. On first run, AI models will be downloaded (~2GB, one-time)
3. Grant microphone access when prompted
4. Start speaking - text appears in real-time

REQUIREMENTS
------------
- Windows 10/11 (64-bit)
- Microphone
- 4GB free disk space (for AI models)
- Internet connection (first run only)

TROUBLESHOOTING
---------------
- If models fail to download, check internet connection
- For microphone issues, check Windows privacy settings
- Logs are stored in _internal/models/ folder

VERSION
-------
v1.0 - Initial release

SUPPORT
-------
Report issues at: [project URL or email]
```

**Functions to implement:**
```python
def create_readme(root_dir: Path, version: str) -> None
def create_license(root_dir: Path, license_type: str) -> None
```

---

### Step 10: Package Distribution
**File:** `build_distribution.py` (continued)

**Tests to write:**
1. `test_create_zip_archive()` - Creates distribution ZIP
2. `test_zip_contents()` - Verifies all files included
3. `test_zip_structure()` - Verifies correct directory structure

**Functions to implement:**
```python
def create_distribution_zip(
    source_dir: Path,
    output_path: Path,
    version: str
) -> Path
def verify_distribution(zip_path: Path) -> dict[str, bool]
```

**Output:** `dist/STT-Stenographer-v1.0.zip`

---

### Step 11: Create Build Helper Scripts
**File:** `build.bat` (Windows launcher)

```batch
@echo off
echo Building STT Distribution...
echo.

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.8+ to build the distribution
    pause
    exit /b 1
)

REM Run build script
python build_distribution.py

if errorlevel 1 (
    echo.
    echo BUILD FAILED
    pause
    exit /b 1
)

echo.
echo BUILD SUCCESSFUL
echo Distribution created in dist/ folder
pause
```

---

## Testing Strategy

### Unit Tests
- `tests/test_build_distribution.py` - Build script functions
- `tests/test_main_paths.py` - Path resolution in _internal structure

### Integration Tests
- `tests/test_distribution_integration.py` - Full build process
  - Downloads Python
  - Installs dependencies
  - Compiles code
  - Creates shortcut
  - Packages ZIP

### Manual Testing (on clean Windows VM)
```powershell
# Test 1: Extract and run
Expand-Archive STT-Stenographer-v1.0.zip
cd STT-Stenographer
# Double-click "STT - Stenographer.lnk"
# Expected: No console, loading window appears, models download, STT works

# Test 2: Verify no Python required
# Uninstall Python from system or test on VM without Python
# Run application - should work fine

# Test 3: Verify signed executables
# Right-click _internal\runtime\python.exe → Properties → Digital Signatures
# Expected: Valid signature from Python Software Foundation

# Test 4: Verify bytecode-only operation
# Delete all .py files (should already be gone)
# Application still works (imports from .pyc)

# Test 5: Second run (models cached)
# Run again - should start immediately without model download
```

---

## Technical Decisions

### Why Embedded Python?
- ✅ Signed executables (no SmartScreen warnings)
- ✅ No installation required
- ✅ Full control over Python version
- ✅ Smaller than PyInstaller (~180MB vs ~200MB+)
- ✅ Easier to debug and maintain

### Why Bytecode Compilation?
- ✅ Hides source code (mild obfuscation)
- ✅ Slightly faster startup (no compilation step)
- ✅ Professional appearance

### Why `_internal` Structure?
- ✅ Mimics PyInstaller convention
- ✅ Clean root directory (only user files)
- ✅ Clear separation of concerns
- ✅ Easy to understand for users

### Why `pythonw.exe`?
- ✅ Designed for GUI applications
- ✅ No console window flash
- ✅ Professional user experience

### Why Pre-install Dependencies?
- ✅ No internet required (except models)
- ✅ Controlled dependency versions
- ✅ Faster startup
- ✅ More reliable (no pip failures)

---

## Dependencies Required for Build

**Runtime dependencies** (installed in distribution):
- numpy
- onnxruntime
- sounddevice
- onnx-asr
- Pillow
- huggingface_hub

**Build-time dependencies** (for build script):
- requests (download embedded Python)
- pywin32 / win32com.client (create .lnk shortcuts)
- System Python with tkinter installed (to copy tkinter module)

---

## Files to Create

### New Files
1. **`build_distribution.py`** - Main build automation script
2. **`build.bat`** - Windows build launcher
3. **`tests/test_build_distribution.py`** - Build script tests
4. **`tests/test_main_paths.py`** - Path resolution tests
5. **`README.txt`** - User documentation (generated)
6. **`LICENSE.txt`** - Software license (generated)
7. **`DISTRIBUTION.md`** - Developer documentation

### Modified Files
1. **`src/main.py`** - Add path resolution for _internal structure
2. **`src/ModelManager.py`** - Update model paths if needed
3. **`requirements.txt`** - Add build dependencies (pywin32)

### Existing Files (already present)
- ✅ `icon.ico` - Application icon (44KB, 6 resolutions)
- ✅ `stenographer.jpg` - Loading screen (96KB)
- ✅ All source code in `src/`

---

## Build Process Summary

```bash
# Developer runs on Windows:
python build_distribution.py

# Build script performs:
1. Download Python 3.13.x embeddable package (~10MB)
2. Extract to _internal/runtime/
3. Copy tkinter module from system Python (~2MB)
4. Configure python313._pth (including Lib/ for tkinter)
5. Install dependencies to _internal/Lib/site-packages/ (~150MB)
6. Copy application code to _internal/app/
7. Compile all .py → .pyc (legacy format)
8. Delete all .py source files
9. Copy config and assets
10. Create launcher shortcut with icon
11. Create README.txt and LICENSE.txt
12. Package everything → dist/STT-Stenographer-v1.0.zip (~122MB)

# Output:
dist/STT-Stenographer-v1.0.zip (ready to distribute)
```

---

## Success Criteria

### Build Success
- [ ] Build script completes without errors
- [ ] All tests pass
- [ ] Distribution ZIP created (~120MB)
- [ ] Correct directory structure
- [ ] No .py files in distribution (only .pyc)
- [ ] Launcher shortcut created with icon
- [ ] Documentation files present

### Runtime Success (tested on clean Windows VM)
- [ ] Extracts cleanly
- [ ] Double-click launcher works
- [ ] No console window appears
- [ ] No Python installation required
- [ ] No "Unknown Publisher" warnings (signed python.exe)
- [ ] Loading window shows stenographer.jpg
- [ ] Model download dialog appears (first run)
- [ ] Models download successfully (~2GB)
- [ ] STT window opens
- [ ] Microphone capture works (sounddevice)
- [ ] Speech recognition works (ONNX runtime)
- [ ] Text displays correctly (tkinter)
- [ ] Second run starts immediately (models cached)
- [ ] Config loads from correct path
- [ ] All paths resolve correctly

---

## Known Limitations

1. **Windows only** - Embedded package is Windows-specific
2. **Size** - ~180MB extracted + 2GB models (acceptable for ML app)
3. **Python version locked** - Users get Python 3.13.x (good for stability)
4. **Updates** - Need to rebuild entire distribution for code changes
5. **Decompilation** - .pyc files can be decompiled (use obfuscator if needed)

---

## Future Enhancements

1. **Auto-updater** - Check for new versions and download updates
2. **Installer** - Create MSI/NSIS installer instead of ZIP
3. **Code signing** - Sign launcher executable with Authenticode certificate
4. **Obfuscation** - Use PyArmor or similar for better code protection
5. **Multi-language** - Support for non-English languages
6. **Custom branding** - Version info, company name in properties
