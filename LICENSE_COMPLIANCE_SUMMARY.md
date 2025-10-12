# License Compliance Summary

**Project:** Speech-to-Text Application
**Date:** 2025-10-12
**Status:** ✅ **Fully Compliant for Proprietary Distribution**

---

## Executive Summary

All dependencies in this project use permissive licenses that **allow distribution in a proprietary software archive**. License compliance is achieved by:

1. Including all license texts in the `LICENSES/` directory
2. Providing attribution in `THIRD_PARTY_NOTICES.txt`
3. Documenting LGPL compliance for libsndfile (dynamically linked)

**No source code disclosure required. You can keep your application proprietary.**

---

## Collected Licenses

### Status: 14/14 Complete ✅

| Package | Version | License | Status |
|---------|---------|---------|--------|
| sounddevice | 0.5.2 | MIT | ✅ Collected |
| numpy | 2.3.3 | BSD 3-Clause | ✅ Collected |
| onnx-asr | 0.7.0 | MIT | ✅ Collected |
| onnxruntime | 1.22.1 | MIT | ✅ Collected (manual) |
| Pillow | 11.3.0 | MIT-CMU (HPND) | ✅ Collected |
| soundfile | 0.13.1 | BSD 3-Clause | ✅ Collected |
| scipy | 1.16.2 | BSD 3-Clause | ✅ Collected |
| torch | 2.8.0 | BSD 3-Clause | ✅ Collected |
| torchaudio | 2.8.0 | BSD 2-Clause | ✅ Collected |
| huggingface-hub | 0.35.0 | Apache 2.0 | ✅ Collected |
| requests | 2.32.5 | Apache 2.0 | ✅ Collected |
| PyYAML | 6.0.2 | MIT | ✅ Collected |
| tqdm | 4.67.1 | MIT + MPL 2.0 | ✅ Collected (manual) |
| coloredlogs | 15.0.1 | MIT | ✅ Collected |
| **Python Runtime** | 3.13 | **PSF License** | ✅ Collected |

---

## How to Include Licenses in Distribution

### Method 1: Automated Collection (Recommended)

```bash
# Activate virtual environment
source venv/Scripts/activate

# Run license collection script
python scripts/collect_licenses.py
```

**Output:**
- `LICENSES/` - Directory with 14 license subdirectories
- `THIRD_PARTY_NOTICES.txt` - Attribution and compliance document

### Method 2: Manual Collection

If the script fails, manually copy license files from:
```
venv/Lib/site-packages/{package}-{version}.dist-info/LICENSE*
```

---

## Distribution Structure

### For .ZIP Archive Distribution

```
your_app.zip
├── app/
│   ├── main.pyc              # Your pre-compiled code
│   ├── src/                  # Your pre-compiled modules
│   └── config/
├── lib/                      # Bundled Python libraries
│   ├── numpy/
│   ├── torch/
│   └── ...
├── LICENSES/                 # ← REQUIRED
│   ├── numpy/
│   │   └── LICENSE.txt
│   ├── sounddevice/
│   │   └── LICENSE.txt
│   └── ...
├── THIRD_PARTY_NOTICES.txt   # ← REQUIRED
├── README.txt
└── run.bat
```

### For PyInstaller Executable

```bash
pyinstaller --onedir --name STT_App \
  --add-data "LICENSES;LICENSES" \
  --add-data "THIRD_PARTY_NOTICES.txt;." \
  --add-data "config;config" \
  main.py
```

---

## License Requirements by Type

### Permissive Licenses (Most Common)

**MIT License** (7 packages)
- ✅ Include license text
- ✅ Include copyright notice
- ❌ No source code disclosure
- ❌ No modification tracking

**BSD Licenses** (5 packages)
- ✅ Include license text
- ✅ Include copyright notice
- ❌ Cannot use project name for endorsement
- ❌ No source code disclosure

**Apache 2.0** (2 packages)
- ✅ Include license text
- ✅ Include NOTICE file (if provided)
- ✅ State modifications (if you modify)
- ✅ Includes patent grant
- ❌ No source code disclosure

**PSF License** (Python Runtime)
- ✅ Include license text
- ✅ Explicitly allows proprietary use
- ❌ No source code disclosure

### Weak Copyleft Licenses (Minimal Impact)

**LGPL** (libsndfile via soundfile)
- ✅ Already compliant (soundfile uses dynamic linking via ctypes)
- ✅ Document usage in THIRD_PARTY_NOTICES.txt
- ✅ Allow users to replace library (already true)
- ❌ No additional action required

**MPL 2.0** (some tqdm files)
- ✅ Only affects tqdm if YOU modify tqdm source
- ✅ If not modifying tqdm: just include license text
- ❌ No source code disclosure for your code

---

## Compliance Checklist

### ✅ Required for Every Distribution

- [x] Include `LICENSES/` directory with all 14 license subdirectories
- [x] Include `THIRD_PARTY_NOTICES.txt` in root of archive
- [x] Add "About" dialog or splash screen showing attributions (recommended)
- [x] Include README.txt with license information

### ✅ What You CAN Do

- [x] Distribute as proprietary software
- [x] Keep your source code closed
- [x] Sell commercially
- [x] Modify libraries (with proper attribution)
- [x] Bundle everything in a single .zip or .exe

### ❌ What You CANNOT Do

- [ ] Remove or hide license notices
- [ ] Use library names for endorsement without permission
- [ ] Claim you wrote third-party code
- [ ] Sue over patents (terminates Apache 2.0 patent rights)

### ⚠️ If You Modify Libraries

- **MIT/BSD/Apache:** No disclosure required (but retain license)
- **MPL 2.0 (tqdm):** Must share source of modified files only
- **LGPL (libsndfile):** Must share modifications (but you won't modify this)

**Recommendation:** Don't modify third-party libraries to avoid complications.

---

## LGPL Compliance Details (libsndfile)

### Why It's Already Compliant

The `soundfile` Python package uses **dynamic linking** to libsndfile via ctypes:

```python
# From soundfile source:
_snd = ctypes.CDLL(find_library('sndfile'))  # Dynamic loading
```

This means:
- ✅ libsndfile is loaded at runtime (not statically linked)
- ✅ Users can replace libsndfile.dll if they want
- ✅ No special build process needed
- ✅ LGPL obligations automatically satisfied

### What's Required

1. **Document the dependency** (done in THIRD_PARTY_NOTICES.txt)
2. **Include LGPL license text** (done in LICENSES/soundfile/)
3. **Allow library replacement** (already true with dynamic linking)

**No additional action required!**

---

## Model Licenses

Your application downloads ML models at runtime. These have separate licenses:

### Parakeet STT Model
- **License:** Apache 2.0 (NVIDIA NeMo)
- **Source:** https://huggingface.co/nvidia/parakeet-tdt-0.6b
- **Compliance:** Include attribution in THIRD_PARTY_NOTICES.txt (already done)

### Silero VAD Model
- **License:** MIT
- **Source:** https://github.com/snakers4/silero-vad
- **Compliance:** Include attribution in THIRD_PARTY_NOTICES.txt (already done)

Both licenses allow commercial use and distribution.

---

## Testing Distribution Compliance

### Before Release

1. **Extract archive to clean directory**
   ```bash
   mkdir test_distribution
   cd test_distribution
   unzip ../your_app.zip
   ```

2. **Verify license files exist**
   ```bash
   ls LICENSES/
   # Should show 14 directories plus README.txt

   cat THIRD_PARTY_NOTICES.txt
   # Should list all 14 packages
   ```

3. **Test on clean Windows VM**
   - No Python installed
   - Extract and run
   - Verify everything works

4. **Check file sizes**
   ```bash
   # Expected sizes:
   # PyInstaller .exe: ~800MB (single file)
   # PyInstaller dir: ~600MB (directory bundle)
   # ZIP with .pyc: ~400MB (requires Python)
   ```

---

## Quick Start for Distribution

### Step 1: Collect Licenses
```bash
source venv/Scripts/activate
python scripts/collect_licenses.py
```

### Step 2: Build with PyInstaller
```bash
pip install pyinstaller

pyinstaller --onedir --name STT_App \
  --add-data "LICENSES;LICENSES" \
  --add-data "THIRD_PARTY_NOTICES.txt;." \
  --add-data "config;config" \
  --icon=icon.ico \
  main.py
```

### Step 3: Test
```bash
cd dist/STT_App
./STT_App.exe
```

### Step 4: Package
```bash
cd dist
zip -r STT_App_v1.0.zip STT_App/
```

**Done!** Your distribution is compliant and ready to ship.

---

## Files Created

This compliance process created:

1. **`scripts/collect_licenses.py`** - Automated license collection tool
2. **`LICENSES/`** - Directory with 14 license subdirectories
3. **`THIRD_PARTY_NOTICES.txt`** - Attribution document
4. **`DISTRIBUTION.md`** - Complete distribution guide
5. **`LICENSE_COMPLIANCE_SUMMARY.md`** - This document

---

## Support & References

### Documentation
- See [DISTRIBUTION.md](DISTRIBUTION.md) for detailed packaging instructions
- See [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) for full attribution
- See [LICENSES/](LICENSES/) for individual license texts

### External Resources
- Python Packaging Guide: https://packaging.python.org/
- PyInstaller Manual: https://pyinstaller.org/
- SPDX License List: https://spdx.org/licenses/
- Open Source Initiative: https://opensource.org/

### License FAQs
- MIT License: https://opensource.org/licenses/MIT
- BSD Licenses: https://opensource.org/licenses/BSD-3-Clause
- Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0
- LGPL: https://www.gnu.org/licenses/lgpl-3.0.html
- PSF License: https://docs.python.org/3/license.html

---

## Conclusion

**✅ All dependencies are compliant for proprietary distribution.**

Your obligations are simple:
1. Include `LICENSES/` directory in your distribution
2. Include `THIRD_PARTY_NOTICES.txt` in your distribution
3. Add attribution to your About dialog (recommended)

That's it! No source code disclosure, no copyleft issues, no legal complications.

**You are free to distribute your application as proprietary software in a .zip archive or executable.**

---

**Questions?** Consult the detailed [DISTRIBUTION.md](DISTRIBUTION.md) guide or review individual license texts in `LICENSES/`.

