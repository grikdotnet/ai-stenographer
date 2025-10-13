# Build Automation Guide

## Overview

The STT project includes automated build scripts to streamline distribution creation and development workflow.

## Build Scripts

### 1. `build_distribution.py` - Full Build
**Use when:** Creating a complete distribution from scratch

**What it does:**
- Downloads Python 3.13.5 embeddable package (~10MB)
- Configures embedded Python with pip and tkinter
- **Collects third-party licenses automatically** (new step 10)
- Installs all dependencies (~150MB)
- Copies and compiles application code to bytecode
- Creates launcher shortcut with icon
- Packages everything into distributable ZIP

**Time:** ~5-10 minutes (first run), ~3-5 minutes (cached Python)

**Usage:**
```bash
python build_distribution.py
```

**When to use:**
- First time building distribution
- After adding/removing dependencies (requirements.txt changed)
- After Python version update
- Before releasing new version

---

### 2. `rebuild_quick.py` - Quick Rebuild
**Use when:** Updating code during development

**What it does:**
- Removes old compiled code
- Copies latest source files
- Recompiles to bytecode
- Updates config files
- Updates assets (stenographer.jpg)
- Copies legal documents (existing)
- **Optional:** Refreshes licenses with `--refresh-licenses`

**Skips:** Python download, pip installation, dependencies

**Time:** ~5-10 seconds (or ~15-30 seconds with --refresh-licenses)

**Usage:**
```bash
# Standard quick rebuild (uses existing licenses)
python rebuild_quick.py

# Refresh licenses too (e.g., after dependency changes)
python rebuild_quick.py --refresh-licenses
```

**When to use:**
- After changing any .py file in src/
- After modifying main.py
- After updating config/stt_config.json
- After changing stenographer.jpg
- Use `--refresh-licenses` after updating requirements.txt (without full rebuild)

---

### 3. `watch_and_rebuild.py` - Auto Rebuild
**Use when:** Actively developing and testing

**What it does:**
- Watches for file changes (src/, main.py, config/, assets)
- Automatically runs `rebuild_quick.py` when changes detected
- Shows rebuild status in console
- Continues watching until Ctrl+C

**Time:** Continuous (rebuilds in ~5-10 seconds per change)

**Usage:**
```bash
python watch_and_rebuild.py
```

**Workflow:**
1. Run watcher in one terminal
2. Edit code in your IDE
3. Save file
4. Watcher detects change and rebuilds automatically
5. Test the built application
6. Repeat steps 2-5

---

### 4. `build.bat` - Interactive Menu (Windows)
**Use when:** Prefer GUI menu over command line

**Features:**
```
1. Full Build
2. Quick Rebuild
3. Watch and Auto-Rebuild
4. Run Application (test built version)
5. Clean Build Directory
6. Exit
```

**Usage:**
```bash
build.bat
```

Select option by number and press Enter.

---

### 5. `src/LicenseCollector.py` - License Collection Module

**Use when:** Automatically during build (or manual verification)

**What it does:**

- Collects license files from all runtime Python packages
- Creates LICENSES/ directory with individual package licenses
- Generates THIRD_PARTY_NOTICES.txt with attribution
- Includes Python runtime license
- Validates license compliance

**Note:** Automatically integrated into build scripts. No manual execution needed.

**Integrated into:**

- `build_distribution.py` - Step 10 (always runs)
- `rebuild_quick.py` - Optional with `--refresh-licenses`

**Output:**

- `LICENSES/` directory with per-package subdirectories
- `THIRD_PARTY_NOTICES.txt` in project root

---

## Development Workflow

### Initial Setup
```bash
# 1. First-time full build
python build_distribution.py

# 2. Verify it works
cd dist/STT-Stenographer
_internal\runtime\pythonw.exe _internal\app\main.pyc
```

### Daily Development
```bash
# Option A: Manual rebuild after changes
# 1. Edit code in src/
# 2. Save file
# 3. Run quick rebuild
python rebuild_quick.py
# 4. Test
cd dist/STT-Stenographer && _internal\runtime\pythonw.exe _internal\app\main.pyc

# Option B: Automatic rebuild (recommended)
# 1. Start watcher in terminal
python watch_and_rebuild.py
# 2. Edit code, save, test immediately
# 3. Repeat edit-save-test cycle
```

### Before Release
```bash
# 1. Clean build
python -c "import shutil; shutil.rmtree('dist/STT-Stenographer', ignore_errors=True)"

# 2. Full build
python build_distribution.py

# 3. Test thoroughly
cd dist/STT-Stenographer
_internal\runtime\pythonw.exe _internal\app\main.pyc

# 4. Package (when Step 9 is implemented)
# Creates STT-Stenographer-v1.0.zip
```

---

## File Watching Details

### Watched Files
The file watcher monitors:
- `src/*.py` - All source modules
- `main.py` - Application entry point
- `config/stt_config.json` - Configuration
- `stenographer.jpg` - Loading screen image

### Not Watched (requires manual action)
- `requirements.txt` - Dependencies list
  - After changes: Run `python rebuild_quick.py --refresh-licenses` or full rebuild
- `icon.ico` - Application icon (requires full rebuild)
- Build scripts themselves (requires restart of watcher)

---

## Troubleshooting

### "Build directory doesn't exist"
**Solution:** Run full build first:
```bash
python build_distribution.py
```

### "Error compiling bytecode"
**Check:**
1. Python syntax errors in source files
2. Python.exe exists in dist/STT-Stenographer/_internal/runtime/

**Solution:**
```bash
# Fix syntax errors, then:
python rebuild_quick.py
```

### "Module not found" when running app
**Cause:** Dependency missing or _pth file misconfigured

**Solution:** Full rebuild:
```bash
python build_distribution.py
```

### Watcher not detecting changes
**Check:**
1. File is in watched directories (src/, main.py, config/, stenographer.jpg)
2. File was actually saved (check IDE save status)
3. No permission errors in console

**Solution:** Restart watcher:
```bash
# Ctrl+C to stop
python watch_and_rebuild.py  # Start again
```

---

## Build Times Reference

| Operation | Time | Notes |
|-----------|------|-------|
| Full Build (first time) | 5-10 min | Downloads Python + dependencies |
| Full Build (cached) | 3-5 min | Uses cached Python, re-installs deps |
| Quick Rebuild | 5-10 sec | Only updates app code |
| Auto Rebuild (per change) | 5-10 sec | Same as quick rebuild |

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Build Distribution

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.13'
      - name: Install build dependencies
        run: pip install requests pywin32
      - name: Build distribution
        run: python build_distribution.py
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: STT-Stenographer
          path: dist/STT-Stenographer-v*.zip
```

---

## Performance Tips

### Speed Up Full Builds
1. **Keep Python cached** - Don't delete `.cache/` folder
2. **Use SSD** - Build on SSD, not HDD
3. **Exclude from antivirus** - Add project folder to exclusions

### Speed Up Quick Rebuilds
1. **Use file watcher** - Saves time vs manual rebuilds
2. **Minimize file changes** - Watcher rebuilds on ANY change
3. **Test without rebuilding** - If only testing logic, run from source:
   ```bash
   python main.py  # Test without rebuild
   ```

---

## Advanced Usage

### Rebuild Only Specific Files
Modify `rebuild_quick.py` to copy only specific modules:
```python
# Example: Only rebuild Recognizer module
src_files = [project_root / "src" / "Recognizer.py"]
```

### Custom Build Configuration
Edit `build_distribution.py` constants:
```python
PYTHON_VERSION = "3.13.0"  # Change Python version
PYTHON_ARCH = "amd64"      # Or "win32" for 32-bit
```

### Parallel Dependency Installation
Speed up full builds by using pip's parallel download:
```python
# In install_dependencies() function:
["--use-deprecated=legacy-resolver", "--parallel=4"]
```

---

## Build Output Structure

After successful build:
```
dist/
└── STT-Stenographer/
    ├── STT - Stenographer.lnk    # Launcher (Step 19)
    ├── README.txt                 # User documentation (Step 18)
    ├── LICENSE.txt                # Main application license (Step 11)
    ├── EULA.txt                   # End user agreement (Step 11)
    ├── THIRD_PARTY_NOTICES.txt    # Attribution info (Step 10 + 11)
    ├── LICENSES/                  # Third-party licenses (Step 10 + 11)
    │   ├── numpy/LICENSE.txt
    │   ├── onnxruntime/LICENSE.txt
    │   ├── sounddevice/LICENSE.txt
    │   ├── python/LICENSE.txt
    │   └── ... (all dependencies)
    ├── icon.ico                   # Application icon
    ├── stenographer.jpg           # Loading screen image
    └── _internal/
        ├── runtime/               # Python 3.13.5 embeddable
        ├── Lib/site-packages/     # Python dependencies
        ├── app/
        │   ├── src/               # Compiled .pyc files
        │   ├── config/            # Configuration files
        │   └── main.pyc           # Entry point (bytecode)
        └── models/                # AI models (downloaded at runtime)
```

---

## Legal Compliance

### Automated License Collection (Step 10)

The build process automatically collects and packages all required legal documents:

**Collected Automatically:**
- Third-party package licenses (from pip packages)
- Python runtime license
- THIRD_PARTY_NOTICES.txt with full attribution

**Manual Files (must exist in project root):**
- LICENSE.txt - Main application license
- EULA.txt - End user license agreement

**Compliance Features:**
- Validates all runtime dependencies have licenses
- Warns if licenses are missing
- Includes license README for users
- Documents LGPL components (libsndfile)
- Includes ML model licenses (Parakeet, Silero)

### License Verification

To manually verify licenses (optional):

```python
# Python script to collect licenses manually
from src.LicenseCollector import LicenseCollector

collector = LicenseCollector(output_dir="LICENSES")
collector.collect_all_licenses()
collector.create_python_license_entry()
collector.generate_notices_file()
collector.generate_readme()
```

Or simply run a full build and inspect the generated files:

```bash
python build_distribution.py

# Check generated files
cat THIRD_PARTY_NOTICES.txt
ls LICENSES/
```

---

## Next Steps

After completing all 19 build steps, you'll have:
- ✓ Automated full builds with license collection
- ✓ Quick rebuilds for development
- ✓ File watching for instant updates
- ✓ Complete distributable package
- ✓ Professional Windows application
- ✓ Legal compliance documentation

**Current Status:** All 19 steps implemented
**Next:** Package into distributable ZIP (future enhancement)
