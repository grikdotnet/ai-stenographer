# Legal Documentation in Distribution

## Overview

The STT distribution includes comprehensive legal documentation to ensure compliance with third-party licenses and provide clear terms to end users.

## Files Included in Distribution

### Root Directory Legal Files

1. **LICENSE.txt**
   - Main application license
   - Copyright © 2025 Grigori Kochanov
   - Terms: Free for personal/commercial use, no warranty, local processing

2. **EULA.txt**
   - End User License Agreement
   - Terms of use for end users

3. **THIRD_PARTY_NOTICES.txt**
   - Attribution for all third-party components
   - Lists all open-source packages used
   - References to individual license files

### LICENSES/ Folder

Contains individual license files for each third-party package:

- `coloredlogs/` - Coloredlogs logging library
- `huggingface_hub/` - HuggingFace model downloads
- `numpy/` - Numerical computing
- `onnx_asr/` - ONNX ASR inference
- `onnxruntime/` - ONNX Runtime
- `Pillow/` - Image processing
- `python/` - Python language license
- `PyYAML/` - YAML parsing
- `requests/` - HTTP library
- `scipy/` - Scientific computing
- `sounddevice/` - Audio capture
- `soundfile/` - Audio file I/O
- `torch/` - PyTorch
- `torchaudio/` - PyTorch audio
- `tqdm/` - Progress bars

Each subdirectory contains the package's LICENSE.txt file.

## Build Integration

### Automatic Copying

Legal documents are automatically copied during:

1. **Full Build** (`build_distribution.py`)
   - Step 8: Copies all legal docs to distribution root

2. **Quick Rebuild** (`rebuild_quick.py`)
   - Step 7-8: Updates legal docs if changed

3. **File Watcher** (`watch_and_rebuild.py`)
   - Watches LICENSE.txt, EULA.txt, THIRD_PARTY_NOTICES.txt
   - Auto-rebuilds if any legal file is modified

### Distribution Structure

```
STT-Stenographer/                    # Root
├── LICENSE.txt                      # Main license
├── EULA.txt                         # End user agreement
├── THIRD_PARTY_NOTICES.txt         # Attribution
├── LICENSES/                        # Third-party licenses
│   ├── README.txt                   # About this folder
│   ├── coloredlogs/
│   │   └── LICENSE.txt
│   ├── numpy/
│   │   └── LICENSE.txt
│   └── ... (15 packages total)
└── _internal/                       # Application files
```

## Compliance Notes

### License Requirements

**MIT License packages:** Require inclusion of license text in distribution ✓

**Apache 2.0 packages:** Require inclusion of license and NOTICE files ✓

**BSD License packages:** Require inclusion of license text ✓

**PSF License (Python):** Requires inclusion of Python license ✓

### Distribution Checklist

When creating a distribution, ensure:

- ✓ LICENSE.txt present in root
- ✓ EULA.txt present in root
- ✓ THIRD_PARTY_NOTICES.txt present in root
- ✓ LICENSES/ folder with all 15+ packages
- ✓ Each package has LICENSE.txt file
- ✓ LICENSES/README.txt explains structure

All these are automatically handled by build scripts.

## Updating Legal Documents

### Adding New Dependencies

When adding a new dependency to `requirements.txt`:

1. **Get the license:**
   ```bash
   pip show <package-name> | grep License
   ```

2. **Create folder:**
   ```bash
   mkdir LICENSES/<package-name>
   ```

3. **Copy license file:**
   - Download from package repository
   - Or extract from installed package
   - Save as `LICENSES/<package-name>/LICENSE.txt`

4. **Update THIRD_PARTY_NOTICES.txt:**
   - Add entry for new package
   - Include package name, version, license type, homepage

5. **Rebuild:**
   ```bash
   python rebuild_quick.py  # Copies new license
   ```

### Modifying Main License

To update LICENSE.txt or EULA.txt:

1. Edit the file in project root
2. Run rebuild (manual or auto via watcher)
3. Updated file automatically copied to distribution

### No Manual Copying Required

The build system automatically:
- Copies legal files to distribution root
- Preserves LICENSES/ folder structure
- Includes README in LICENSES/ folder
- Updates on every rebuild

## Verification

### Check Legal Docs in Build

```bash
# Verify root legal files
ls -lh dist/STT-Stenographer/*.txt

# Verify LICENSES folder
ls -la dist/STT-Stenographer/LICENSES/

# Count packages
ls -d dist/STT-Stenographer/LICENSES/*/ | wc -l
```

Should show:
- 3 text files in root (LICENSE, EULA, THIRD_PARTY_NOTICES)
- 15+ package folders in LICENSES/
- Each package folder has LICENSE.txt

## Legal Document Templates

### LICENSE.txt Format
- Copyright notice
- Grant of rights
- Disclaimer of warranty
- Limitation of liability
- Attribution requirements

### EULA.txt Format
- Acceptance of terms
- Grant of license
- Restrictions
- Termination
- Warranty disclaimer

### THIRD_PARTY_NOTICES.txt Format
```
Third-Party Notices
===================

This application includes the following third-party components:

Package Name
------------
Version: X.Y.Z
License: MIT
Homepage: https://...
Copyright: © Year Author
License File: LICENSES/package-name/LICENSE.txt

[Repeat for each package]
```

## Questions & Answers

**Q: Can I distribute without license files?**
A: No. Third-party licenses require their inclusion in distributions.

**Q: Can I remove packages from LICENSES/ folder?**
A: Only if you're no longer using that package. Otherwise, it's required.

**Q: Do I need to update licenses when updating packages?**
A: Yes, if the license changed. Check package changelog.

**Q: Can I use a different main LICENSE.txt?**
A: Yes, but ensure it's compatible with all third-party licenses.

**Q: Are legal docs included in final ZIP?**
A: Yes, when Step 9 is completed (packaging).

## Support

For questions about licensing or legal compliance:
- Check third-party package documentation
- Review OSI-approved licenses: https://opensource.org/licenses
- Consult legal counsel for commercial use

---

**Note:** This documentation is automatically kept in sync with the actual distribution through the build automation system.
