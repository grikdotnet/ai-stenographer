# AI Stenographer Native Launcher

## Overview

This directory contains the native Windows executable launcher for the AI Stenographer MSIX package. The launcher is required because MSIX packages cannot directly execute `.lnk` shortcuts or Python bytecode files.

## Files

- **AIStenographer.cpp** - C++ source code for the native launcher
- **AIStenographer.exe** - Compiled native launcher executable (139 KB)
- **build_launcher.py** - Python build script to compile the launcher
- **build_launcher.bat** - Windows batch build script (alternative)

## Purpose

The native launcher serves as the entry point for the MSIX package:

1. **MSIX Requirement**: AppxManifest.xml requires a native `.exe` as the application entry point
2. **Launch Logic**: The launcher executable:
   - Locates the Python runtime: `_internal\runtime\pythonw.exe`
   - Finds the application bytecode: `_internal\app\main.pyc`
   - Launches Python with the application
   - Exits immediately (background operation)

## Architecture

```
AIStenographer.exe (native launcher)
    │
    └─> pythonw.exe (Python 3.13.5 embedded runtime)
            │
            └─> main.pyc (AI Stenographer application)
```

## Building

### Requirements

- Visual Studio Build Tools 2022
- MSVC v14.44.35207 (or compatible version)
- Windows SDK 10.0.26100.0 (or compatible version)

### Build Command

```bash
python build_launcher.py
```

Or alternatively:

```cmd
build_launcher.bat
```

### Build Output

- **Executable**: `AIStenographer.exe` (~139 KB)
- **Intermediate files**: Automatically cleaned up after build

## Implementation Details

### Path Resolution

The launcher uses the following algorithm to locate Python and the application:

```cpp
// Get executable directory
GetModuleFileNameW(NULL, exePath, MAX_PATH);
PathRemoveFileSpecW(exePath);

// Build paths
pythonPath = exePath + "\\_internal\\runtime\\pythonw.exe"
appPath = exePath + "\\_internal\\app\\main.pyc"
```

### Error Handling

The launcher includes validation for:
- Missing Python runtime
- Missing application bytecode
- Process creation failures

All errors are displayed via `MessageBoxW` with descriptive error messages.

### Process Management

The launcher uses `CreateProcessW` with the following behavior:
- **Working directory**: Set to executable directory
- **Handle inheritance**: Disabled
- **Window display**: Inherited from parent
- **Process wait**: None (launcher exits immediately)

## Compilation Options

### Compiler Flags

- `/EHsc` - C++ exception handling
- `/O2` - Maximum optimization
- `/W3` - Warning level 3
- `/UNICODE` - Unicode build
- `/SUBSYSTEM:WINDOWS` - GUI application (no console)

### Linked Libraries

- `shlwapi.lib` - Shell path manipulation (`PathRemoveFileSpecW`)
- `user32.lib` - User interface functions (`MessageBoxW`)

## Integration with MSIX

### AppxManifest.xml Reference

```xml
<Application Id="App"
             Executable="AIStenographer.exe"
             EntryPoint="Windows.FullTrustApplication">
```

### Package Structure

```
AIStenographer_1.0.0.0_x64/
├── AIStenographer.exe          <-- This launcher
├── AppxManifest.xml
├── Assets/
└── _internal/
    ├── runtime/
    │   └── pythonw.exe
    └── app/
        └── main.pyc
```

## Maintenance

### Updating the Launcher

1. Modify `AIStenographer.cpp`
2. Run `python build_launcher.py`
3. Test the new executable
4. Rebuild MSIX package

### Troubleshooting

**Issue**: Compiler not found
- **Solution**: Verify Visual Studio Build Tools 2022 installation path in `build_launcher.py`

**Issue**: SDK headers not found
- **Solution**: Update SDK version in `build_launcher.py` to match installed version

**Issue**: Runtime errors
- **Solution**: Add debug output to launcher or test with `cmd.exe` wrapper

## References

- [MSIX entry point limitations](https://www.advancedinstaller.com/how-to-fix-msix-limitations-with-psf.html)
- [Entry points discussion](https://github.com/microsoft/msix-packaging/discussions/618)
- [Microsoft Store Preparation Guide](../../docs/MICROSOFT_STORE_PREPARATION.md)
