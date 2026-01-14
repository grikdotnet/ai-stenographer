# Microsoft Store Preparation Guide

**Status:** Planning Phase

## Overview

This document outlines the steps to transform the current portable distribution (created by `build_distribution.py`) into a Microsoft Store-ready MSIX package.

## Current Distribution vs. Store Requirements

### Current State (Portable)
- **Format:** ZIP archive with `.lnk` shortcut launcher
- **Entry Point:** `cmd.exe /C start /B _internal\runtime\pythonw.exe _internal\app\main.pyc`
- **File Paths:** Relative to executable (`_internal/models/`, `_internal/app/config/`)
- **Signing:** Python executables only (system Python binaries)
- **Assets:** `stenographer.jpg` + Windows system icons
- **Distribution Size:** ~150MB (models downloaded on first run, ~2GB)

### Store Requirements
- **Format:** MSIX signed package
- **Entry Point:** Native `.exe` wrapper (no `.lnk` shortcuts allowed)
- **File Paths:** AppData sandbox (`%LOCALAPPDATA%/AI-Stenographer/`)
- **Signing:** MSIX package certificate + timestamp
- **Assets:** Multiple PNG icons (44x44, 150x150, 310x310, etc.)
- **Legal:** Privacy policy required (microphone/internet access)
- **Metadata:** AppxManifest.xml with capabilities declaration

---

## Implementation Tasks

### Task 1: AppxManifest.xml Creation

Create `msix/AppxManifest.xml` with required Store metadata.

**Key Requirements:**
- **Identity:** Publisher must match certificate Subject field
- **Version:** Format `X.Y.Z.0` (last digit reserved for Store)
- **Capabilities:** `<DeviceCapability Name="microphone" />` + `<Capability Name="internetClient" />`
- **Entry Point:** Reference to launcher executable (see Task 2)

**Template Structure:**

```xml
<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities">

  <Identity
    Name="AI.Stenographer"
    Publisher="CN=Grigori Kochanov"
    Version="1.0.0.0" />

  <Properties>
    <DisplayName>AI Stenographer</DisplayName>
    <PublisherDisplayName>Grigori Kochanov</PublisherDisplayName>
    <Logo>Assets\StoreLogo.png</Logo>
    <Description>Real-time offline speech-to-text dictation</Description>
  </Properties>

  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.17763.0" MaxVersionTested="10.0.22621.0" />
  </Dependencies>

  <Resources>
    <Resource Language="en-us" />
  </Resources>

  <Applications>
    <Application Id="App" Executable="AIStenographer.exe" EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements
        DisplayName="AI Stenographer"
        Description="Real-time offline speech-to-text dictation with local processing"
        BackgroundColor="transparent"
        Square150x150Logo="Assets\Square150x150Logo.png"
        Square44x44Logo="Assets\Square44x44Logo.png">
        <uap:DefaultTile Wide310x150Logo="Assets\Wide310x150Logo.png" />
        <uap:SplashScreen Image="Assets\SplashScreen.png" />
      </uap:VisualElements>
    </Application>
  </Applications>

  <Capabilities>
    <Capability Name="internetClient" />
    <DeviceCapability Name="microphone" />
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
</Package>
```

**Sources:**
- [App capability declarations](https://learn.microsoft.com/en-us/windows/uwp/packaging/app-capability-declarations)
- [App package manifest](https://learn.microsoft.com/en-us/uwp/schemas/appxpackage/appx-package-manifest)
- [How to specify device capabilities](https://learn.microsoft.com/en-us/uwp/schemas/appxpackage/how-to-specify-device-capabilities-in-a-package-manifest)

---

### Task 2: Native Launcher Executable

Create `msix/launcher/AIStenographer.exe` as MSIX entry point.

**Why Needed:**
- MSIX cannot execute `.lnk` shortcuts or bytecode directly ([MSIX limitations](https://www.advancedinstaller.com/how-to-fix-msix-limitations-with-psf.html))
- Store requires native executable as entry point ([Entry points discussion](https://github.com/microsoft/msix-packaging/discussions/618))
- Current `cmd.exe` chain won't work in Store environment

**Implementation Options:**

#### Option A: Minimal C++ Stub (Recommended)
**Pros:** Small size (~50KB), fast startup, no runtime dependencies
**Cons:** Requires C++ toolchain (Visual Studio)

```cpp
// AIStenographer.cpp
#include <windows.h>
#include <shlwapi.h>
#include <string>

#pragma comment(lib, "shlwapi.lib")

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Get executable directory
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    PathRemoveFileSpecW(exePath);

    // Build path to Python runtime
    std::wstring pythonPath = std::wstring(exePath) + L"\\_internal\\runtime\\pythonw.exe";
    std::wstring appPath = std::wstring(exePath) + L"\\_internal\\app\\main.pyc";

    // Launch Python with app
    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi;

    std::wstring cmdLine = L"\"" + pythonPath + L"\" \"" + appPath + L"\"";

    if (!CreateProcessW(NULL, &cmdLine[0], NULL, NULL, FALSE, 0, NULL, exePath, &si, &pi)) {
        MessageBoxW(NULL, L"Failed to launch AI Stenographer", L"Error", MB_ICONERROR);
        return 1;
    }

    // Don't wait for process to exit (background operation)
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return 0;
}
```

**Build Command:**
```bash
cl.exe /EHsc /O2 /Fe:AIStenographer.exe AIStenographer.cpp /link /SUBSYSTEM:WINDOWS
```

#### Option B: Python Wrapper with PyInstaller
**Pros:** Pure Python, easier to maintain
**Cons:** Larger size (~5MB), slower startup

```python
# launcher.py
import sys
import subprocess
from pathlib import Path

def main():
    exe_dir = Path(sys.executable).parent
    python_exe = exe_dir / "_internal" / "runtime" / "pythonw.exe"
    app_main = exe_dir / "_internal" / "app" / "main.pyc"

    subprocess.Popen([str(python_exe), str(app_main)],
                     cwd=str(exe_dir),
                     creationflags=subprocess.CREATE_NO_WINDOW)

if __name__ == "__main__":
    main()
```

**Build Command:**
```bash
pyinstaller --onefile --windowed --name AIStenographer launcher.py
```

**Decision:** Use **Option A (C++ stub)** for Windows Store build only (keep `.lnk` for portable distribution).

**Sources:**
- [MSIX entry point limitations](https://www.advancedinstaller.com/how-to-fix-msix-limitations-with-psf.html)
- [BAT file launcher workaround](https://www.advancedinstaller.com/launch-bat-file-from-msix-shortcut.html)

---

### Task 3: Path Resolution Updates

Modify `main.py` to support AppData locations for Store environment.

**Current Logic (lines 38-97):**
```python
if getattr(sys, 'frozen', False):
    # Distribution mode
    internal_dir = exe_dir / "_internal"
    MODELS_DIR = internal_dir / "models"
    CONFIG_DIR = app_dir / "config"
else:
    # Development mode
    MODELS_DIR = project_dir / "models"
    CONFIG_DIR = project_dir / "config"
```

**Store Requirement:**
- Models: `%LOCALAPPDATA%\AI-Stenographer\models\` (user-writable, ~2GB)
- Config: `%LOCALAPPDATA%\AI-Stenographer\config\` (user-writable)
- App code: `C:\Program Files\WindowsApps\...\` (read-only, signed)

**Updated Logic:**

```python
import os
from pathlib import Path

def resolve_paths() -> Dict[str, Path]:
    """
    Resolves application paths for different environments.

    Returns:
        Dictionary with resolved paths for models, config, etc.

    Environments:
    - Development: Uses project directory (./models, ./config)
    - Portable: Uses _internal relative paths
    - Store: Uses AppData for writable data
    """
    # Detect environment
    is_store = os.environ.get('MSIX_PACKAGE_IDENTITY') is not None
    is_frozen = getattr(sys, 'frozen', False)

    if is_store:
        # Microsoft Store environment (sandboxed)
        app_data = Path(os.environ['LOCALAPPDATA']) / "AI-Stenographer"
        app_data.mkdir(parents=True, exist_ok=True)

        return {
            "MODELS_DIR": app_data / "models",
            "CONFIG_DIR": app_data / "config",
            "APP_DIR": Path(sys.executable).parent,
            "ENVIRONMENT": "store"
        }
    elif is_frozen:
        # Portable distribution
        exe_dir = Path(sys.executable).parent
        internal_dir = exe_dir / "_internal"

        return {
            "MODELS_DIR": internal_dir / "models",
            "CONFIG_DIR": internal_dir / "app" / "config",
            "APP_DIR": internal_dir / "app",
            "ENVIRONMENT": "portable"
        }
    else:
        # Development mode
        project_dir = Path(__file__).parent

        return {
            "MODELS_DIR": project_dir / "models",
            "CONFIG_DIR": project_dir / "config",
            "APP_DIR": project_dir,
            "ENVIRONMENT": "development"
        }
```

**Migration Strategy:**
- Store version checks for existing `_internal/models/` on first run
- Offers to copy models to AppData (avoid re-downloading 2GB)
- Config files copied automatically with defaults preserved

**Configuration Variable:**
Add to build script:
```python
# build_msix_distribution.py
BUILD_TYPE = "store"  # or "portable" or "macos"
```

---

### Task 4: Visual Assets Creation

Generate required icon suite from `stenographer.jpg`.

**Required Assets (PNG format):**

| Asset | Size | Purpose | Transparency |
|-------|------|---------|--------------|
| `Square44x44Logo.png` | 44x44 | Taskbar icon | Optional |
| `Square150x150Logo.png` | 150x150 | Medium tile | Optional |
| `Wide310x150Logo.png` | 310x150 | Wide tile | Optional |
| `StoreLogo.png` | 50x50 | Store listing | Required |
| `SplashScreen.png` | 620x300 | App launch | Optional |

**Note:** User confirmed that Windows built-in icons are acceptable. However, Store requires custom assets for listing.

**Solution:**
1. Use `stenographer.jpg` (1020x768) as source
2. Create simple icon suite with cropped/resized versions
3. Store logo can be simple text-based design if preferred
4. No designer needed - can use ImageMagick or Pillow for generation

**Automated Generation Script (to be created):**
```python
# generate_store_assets.py
from PIL import Image

source = Image.open("stenographer.jpg")
# Center crop and resize to required dimensions
# Save to msix/Assets/ directory
```

---

### Task 5: Privacy Policy

Create `PRIVACY_POLICY.txt` based on existing EULA/LICENSE structure.

**Required Disclosures:**
- Microphone access (local speech recognition only)
- Internet access (model downloads from HuggingFace Hub only)
- No data collection or transmission
- Local processing guarantee
- Model download transparency

**Draft Template:**

```
Privacy Policy

Last updated: December 16, 2025
Publisher: Grigori Kochanov
Application: AI Stenographer

1. Data Collection

AI Stenographer does NOT collect, store, or transmit any personal data,
audio recordings, or transcribed text to external servers.

2. Microphone Access

The application requires microphone access to perform real-time speech
recognition. All audio processing occurs locally on your device. Audio
data is never saved to disk or transmitted over the network.

3. Internet Access

Internet access is used solely to download AI models (~2GB) from
HuggingFace Hub on first launch. After initial download, the application
can operate fully offline.

Downloaded models:
- NVIDIA Parakeet TDT 0.6B (speech-to-text)
- Silero VAD (voice activity detection)

4. Local Processing

All speech recognition, voice activity detection, and text processing
occurs entirely on your local device. No cloud services are used.

5. Data Storage

The application stores:
- AI models: %LOCALAPPDATA%\AI-Stenographer\models\ (~2GB)
- Configuration: %LOCALAPPDATA%\AI-Stenographer\config\ (<1MB)

No user data, transcripts, or audio is stored.

6. Third-Party Services

Model downloads use HuggingFace Hub CDN. No account or authentication
is required. Downloads are anonymous and use HTTPS.

7. Children's Privacy

The application does not knowingly collect data from children under 13.

8. Changes to This Policy

Updates will be posted with version releases. Continued use implies
acceptance of policy changes.

9. Contact

For privacy concerns, contact: [your email or GitHub issues URL]
```

**Sources:**
- [EULA.txt](file://c:/workspaces/stt/EULA.txt) Section 7 (existing privacy statement)
- [LICENSE.txt](file://c:/workspaces/stt/LICENSE.txt) Section 4 (local processing guarantee)

---

### Task 6: Certificate Signing

Generate self-signed certificate for testing (Store will re-sign for production).

**Important Notes:**
- Self-signed certificate is for **testing only** ([Certificate guide](https://learn.microsoft.com/en-us/windows/msix/package/create-certificate-package-signing))
- Microsoft Store will **re-sign** package with trusted certificate during submission ([Signing overview](https://learn.microsoft.com/en-us/windows/msix/package/signing-package-overview))
- Certificate Subject must match Publisher in `AppxManifest.xml` ([MSIX certificates](https://www.advancedinstaller.com/msix-certificates-developer.html))

**Generation Steps:**

SDK_PATH = "C:\Program Files (x86)\Windows Kits\10\App Certification Kit"

```powershell
# Create self-signed certificate for testing
$cert = New-SelfSignedCertificate `
    -Type Custom `
    -Subject "CN=Grigori Kochanov" `
    -KeyUsage DigitalSignature `
    -FriendlyName "AI Stenographer Test Certificate" `
    -CertStoreLocation "Cert:\CurrentUser\My" `
    -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3", "2.5.29.19={text}")

# Export certificate to file
$password = ConvertTo-SecureString -String "test123" -Force -AsPlainText
Export-PfxCertificate -Cert "Cert:\CurrentUser\My\$($cert.Thumbprint)" `
    -FilePath "msix\AIStenographer_TestCert.pfx" `
    -Password $password

# Export public key for installation
Export-Certificate -Cert "Cert:\CurrentUser\My\$($cert.Thumbprint)" `
    -FilePath "msix\AIStenographer_TestCert.cer"
```

**Install Certificate on Test Machine:**
```powershell
# Install to Trusted Root (required for sideloading)
Import-Certificate -FilePath "msix\AIStenographer_TestCert.cer" `
    -CertStoreLocation "Cert:\LocalMachine\Root"
```

**Sign MSIX Package:**
```powershell
# Using Windows SDK SignTool
signtool.exe sign /fd SHA256 /a /f "AIStenographer_TestCert.pfx" /p "test123" /t http://timestamp.digicert.com "AIStenographer_1.0.0.0_x64.msix"
```

**Timestamping Recommendation:**
- Always use timestamp server ([Signing overview](https://learn.microsoft.com/en-us/windows/msix/package/signing-package-overview))
- Allows package to be installed even after certificate expires
- Recommended server: `http://timestamp.digicert.com`

**Sources:**
- [Create certificate for package signing](https://learn.microsoft.com/en-us/windows/msix/package/create-certificate-package-signing)
- [Self-signed certificate guide](https://www.tmurgent.com/TmBlog/?p=3461)
- [MSIX signing documentation](https://msixhero.net/documentation/how-to-sign-msix-packages/)

---

### Task 7: Build Script - build_msix_distribution.py

Create new build script for MSIX packaging (separate from portable distribution).

**Key Differences from `build_distribution.py`:**
- Uses `AIStenographer.exe` launcher instead of `.lnk` shortcut
- Includes `AppxManifest.xml` in package root
- Adds `Assets/` directory with icons
- Uses `MakeAppx.exe` tool from Windows SDK for packaging
- Skips model pre-packaging (download on first run maintained)
- Generates folder structure compatible with MSIX layout

**Package Structure:**
```
AIStenographer_1.0.0.0_x64/
├── AppxManifest.xml
├── AIStenographer.exe                  # C++ launcher stub
├── Assets/
│   ├── Square44x44Logo.png
│   ├── Square150x150Logo.png
│   ├── Wide310x150Logo.png
│   ├── StoreLogo.png
│   └── SplashScreen.png
├── _internal/
│   ├── runtime/                        # Python 3.13.5 embeddable
│   ├── Lib/                           # Dependencies
│   ├── app/                           # Application bytecode
│   └── models/                        # Empty (download on first run)
├── README.txt
├── LICENSE.txt
├── EULA.txt
├── PRIVACY_POLICY.txt
└── THIRD_PARTY_NOTICES.txt
```

**Build Process:**

```python
# Pseudo-code for build_msix_distribution.py

1. Create MSIX staging directory
2. Copy AppxManifest.xml with version substitution
3. Build C++ launcher (AIStenographer.exe) using cl.exe
4. Download and extract Python embeddable package
5. Install dependencies (pip install -r requirements.txt)
6. Compile .py to .pyc bytecode
7. Package .pyc into .zip archives (zipimport)
8. Copy Assets/ (generated icons)
9. Copy legal documents (LICENSE, EULA, PRIVACY_POLICY)
10. Run MakeAppx.exe to create .msix package
11. Sign package with certificate using signtool.exe
12. Verify signature and manifest validity
```

**Windows SDK Requirements:**
- `MakeAppx.exe` - Package creation tool
- `SignTool.exe` - Code signing tool
- Typically located: `C:\Program Files (x86)\Windows Kits\10\bin\<SDK version>\x64\`

**Command Examples:**

```bash
# Create MSIX package
MakeAppx.exe pack /d "AIStenographer_1.0.0.0_x64" /p "AIStenographer_1.0.0.0_x64.msix"

# Sign package
SignTool.exe sign /fd SHA256 /a /f "AIStenographer_TestCert.pfx" /p "test123" /t http://timestamp.digicert.com "AIStenographer_1.0.0.0_x64.msix"

# Verify signature
SignTool.exe verify /pa "AIStenographer_1.0.0.0_x64.msix"
```

**Sources:**
- [MSIX packaging tutorial](https://www.advancedinstaller.com/application-packaging-training/msix-packaging/ebook/msix-step-by-step-tutorials.html)
- [App package requirements](https://learn.microsoft.com/en-us/windows/apps/publish/publish-your-app/msix/app-package-requirements)

---

### Task 8: Testing & Validation

Test MSIX package installation and functionality before Store submission.

**Sideloading Steps:**

```powershell
# 1. Enable Developer Mode (Windows Settings)
# Settings → Update & Security → For developers → Developer mode

# 2. Install certificate (if not already done)
Import-Certificate -FilePath "AIStenographer_TestCert.cer" -CertStoreLocation "Cert:\LocalMachine\Root"

# 3. Install MSIX package
Add-AppxPackage -Path "AIStenographer_1.0.0.0_x64.msix"

# 4. Launch app
Start-Process "shell:AppsFolder\AI.Stenographer_<publisher_id>!App"
```

**Validation Checklist:**

- [ ] Package installs without errors
- [ ] App launches and shows GUI
- [ ] Microphone permission prompt appears
- [ ] Microphone access works (speech recognition functional)
- [ ] Model download works (internet access functional)
- [ ] Models saved to AppData (`%LOCALAPPDATA%\AI-Stenographer\models\`)
- [ ] Config files created in AppData
- [ ] App appears in Start Menu with correct icon
- [ ] Uninstall works cleanly (no leftover files in AppData)
- [ ] Certificate validation passes
- [ ] Manifest validation passes (no errors in Event Viewer)

**Debugging Tools:**

```powershell
# Check installed packages
Get-AppxPackage -Name "AI.Stenographer"

# View app logs
Get-WinEvent -LogName Microsoft-Windows-AppxPackaging/Operational | Where-Object {$_.Message -like "*AI.Stenographer*"}

# Uninstall for testing
Remove-AppxPackage -Package (Get-AppxPackage -Name "AI.Stenographer").PackageFullName
```

---

## Store Submission Preparation

### Partner Center Requirements

**App Listing Information:**
- **Category:** Productivity → Dictation ([Category selection](https://learn.microsoft.com/en-us/windows/apps/publish/publish-your-app/msix/app-package-requirements))
- **Description:** Real-time offline speech-to-text dictation with local AI processing
- **Keywords:** speech-to-text, dictation, transcription, offline, STT, voice typing
- **Screenshots:** 3-9 images (1366x768 or higher recommended)
- **Privacy Policy URL:** Link to hosted PRIVACY_POLICY.txt (can use GitHub Pages)
- **Age Rating:** IARC questionnaire (likely rated E for Everyone)

**Store Listing Assets:**
- 1366x768 screenshot (minimum 1 required, up to 9 recommended)
- Store logo: 300x300 PNG (higher resolution than manifest's 50x50)
- Optional: Promotional images, video trailer

**Certification Requirements:**
- No malware or deceptive behavior
- Microphone capability justified (disclosed in privacy policy)
- Internet capability justified (model downloads disclosed)
- No crashes during testing
- Functional on Windows 10 version 1809+ and Windows 11

---

## Build Configuration Variables

Add to `build_msix_distribution.py`:

```python
# Build configuration
BUILD_TYPE = "store"  # Options: "portable", "store", "macos"
PYTHON_VERSION = "3.13.5"
APP_VERSION = "1.0.0.0"  # MSIX format (last digit reserved for Store)
PUBLISHER_NAME = "CN=Grigori Kochanov"
PACKAGE_NAME = "AI.Stenographer"

# Paths
MODELS_LOCATION = "appdata"  # Options: "internal", "appdata"
CONFIG_LOCATION = "appdata"  # Options: "internal", "appdata"

# Features
INCLUDE_MODELS_IN_PACKAGE = False  # Keep download-on-first-run
MIGRATE_FROM_PORTABLE = True       # Auto-detect and migrate existing models
```

---

## Next Steps

1. **User Decision:** Approve this plan and confirm timeline
2. **Development:** Implement Tasks 1-8 sequentially
3. **Testing:** Sideload MSIX on clean Windows 10/11 machines
4. **Submission:** Create Partner Center listing and submit package
5. **Certification:** Respond to Microsoft feedback if needed
6. **Launch:** Publish to Store and update project documentation

---

## Related Files

**Existing:**
- [build_distribution.py](file://c:/workspaces/stt/build_distribution.py) - Portable distribution builder (reference)
- [main.py](file://c:/workspaces/stt/main.py) - Entry point (requires path resolution updates)
- [EULA.txt](file://c:/workspaces/stt/EULA.txt) - End-user license agreement
- [LICENSE.txt](file://c:/workspaces/stt/LICENSE.txt) - Software license

**To Be Created:**
- `msix/AppxManifest.xml` - Store package manifest
- `msix/launcher/AIStenographer.cpp` - Native launcher stub
- `msix/Assets/*.png` - Icon suite (5 files)
- `PRIVACY_POLICY.txt` - Privacy policy document
- `build_msix_distribution.py` - MSIX build script
- `generate_store_assets.py` - Icon generation script
- `msix/create_test_certificate.ps1` - Certificate generation script

---

## References

### Microsoft Documentation
- [App capability declarations](https://learn.microsoft.com/en-us/windows/uwp/packaging/app-capability-declarations)
- [App package manifest](https://learn.microsoft.com/en-us/uwp/schemas/appxpackage/appx-package-manifest)
- [Create certificate for package signing](https://learn.microsoft.com/en-us/windows/msix/package/create-certificate-package-signing)
- [Sign package overview](https://learn.microsoft.com/en-us/windows/msix/package/signing-package-overview)
- [App package requirements](https://learn.microsoft.com/en-us/windows/apps/publish/publish-your-app/msix/app-package-requirements)
- [How to specify device capabilities](https://learn.microsoft.com/en-us/uwp/schemas/appxpackage/how-to-specify-device-capabilities-in-a-package-manifest)

### Third-Party Resources
- [MSIX certificates developer guide](https://www.advancedinstaller.com/msix-certificates-developer.html)
- [MSIX limitations and PSF fixes](https://www.advancedinstaller.com/how-to-fix-msix-limitations-with-psf.html)
- [Self-signed certificate tutorial](https://www.tmurgent.com/TmBlog/?p=3461)
- [MSIX signing guide](https://msixhero.net/documentation/how-to-sign-msix-packages/)
- [MSIX packaging tutorials](https://www.advancedinstaller.com/application-packaging-training/msix-packaging/ebook/msix-step-by-step-tutorials.html)
- [BAT file launcher workaround](https://www.advancedinstaller.com/launch-bat-file-from-msix-shortcut.html)

---

**Document Status:** Draft for Review
**Requires User Approval:** Yes
**Implementation Start:** Pending approval
