
## **Milestone 0: Distribution Optimization**

**Goal:** Reduce distribution filesystem overhead by packaging .pyc files into .zip archives

### Background

Current distribution in `dist/AI-Stenographer/_internal/Lib/site-packages/` contains **2,997 individual .pyc files** across ~40 packages (167MB total). This creates:
- Filesystem overhead (thousands of small files)
- Slow directory traversal
- Inefficient disk space usage

**Solution:** Use Python's built-in `zipimport` to package each library's .pyc files into single .zip archives (e.g., `numpy.zip`, `requests.zip`).

### Tasks

#### 0.1 Design zip packaging strategy (TDD)

- Research Python zipimport capabilities with .pyc files
- Design per-library zip structure (preserve native binaries outside)
- Define exclusion rules (native .pyd/.dll files must stay uncompressed)
- Document import path requirements for python313._pth

#### 0.2 Write tests for zip packaging (TDD - Tests First)

- Test: `test_zip_site_packages_creates_archives()` - Verify .zip files created
- Test: `test_zip_preserves_native_binaries()` - Native .pyd/.dll stay uncompressed
- Test: `test_zip_removes_source_pyc()` - Original .pyc files removed after zipping
- Test: `test_zip_import_compatibility()` - Python can import from .zip archives
- Test: `test_zip_handles_single_file_modules()` - Single .pyc files → .zip
- Test: `test_zip_preserves_data_folders()` - e.g., numpy.libs/, _sounddevice_data/

#### 0.3 Implement zip_site_packages() function

- Implement function in [build_distribution.py](build_distribution.py)
- Scan site-packages for top-level packages and modules
- Create .zip archives with ZIP_DEFLATED compression
- Preserve native binaries in original locations
- Handle single-file modules (e.g., `sounddevice.pyc` → `sounddevice.zip`)
- Clean up original .pyc files after successful archiving

#### 0.4 Integration and testing

- Add `zip_site_packages()` to main build pipeline (after `compile_site_packages()`)
- Run build and verify distribution structure
- Test imports from embedded Python: `python.exe -c "import numpy; import requests"`
- Measure reduction: file count before/after, disk space impact

#### 0.5 Update build tests

- Update `test_build_distribution.py` to expect .zip archives
- Verify python313._pth includes correct paths for zipimport
- Add regression test to prevent future .pyc proliferation

### Details

- **Dependencies:** None (build system improvement)
- **Estimated effort:** 1-2 days
- **Risk:** Low (Python zipimport is stable since 2.3)
- **Success criteria:**
  - ✅ .pyc file count reduced by >90% (2997 → ~300 native binaries)
  - ✅ All critical imports work (`numpy`, `onnxruntime`, `sounddevice`, etc.)
  - ✅ Distribution size unchanged or smaller (better compression)
  - ✅ All existing tests pass

### Expected Results

**Before:**
```
_internal/Lib/site-packages/
├── numpy/__init__.pyc
├── numpy/core/multiarray.pyc
├── numpy/core/_multiarray_umath.cp313-win_amd64.pyd  (native)
├── numpy.libs/libopenblas.dll                         (native)
└── ... (2997 files total)
```

**After:**
```
_internal/Lib/site-packages/
├── numpy.zip                                          (all .pyc files)
├── numpy/core/_multiarray_umath.cp313-win_amd64.pyd  (native, uncompressed)
├── numpy.libs/                                        (native DLLs, uncompressed)
└── ... (~40 .zip files + ~300 native binaries)
```

---

## **Milestone 1: Performance & Hardware Acceleration (Critical - Q1)**

**Goal:** Fix CPU overload on battery-powered devices using NPU/GPU acceleration

### Tasks

#### 1.1 Investigate OpenVINO integration for NPU/GPU inference

- Research OpenVINO compatibility with Parakeet ONNX model
- Profile current CPU usage patterns on Intel Core 135U + NPU
- Analyze buffer overflow logs and identify bottlenecks

#### 1.2 Design execution provider abstraction layer

- Create ExecutionProviderManager class
- Define fallback chain: NPU → GPU → CPU
- Update ModelManager to support multiple providers

#### 1.3 Implement OpenVINO execution provider

- Integrate OpenVINO as alternative to onnxruntime
- Add device detection (Intel NPU, iGPU)
- Handle model conversion if needed (ONNX → OpenVINO IR)

#### 1.4 Test and optimize on target hardware

- Benchmark on Intel Core 135U (NPU + iGPU)
- Measure CPU usage (target: <20% on battery)
- Work around for buffer overflows
- Validate recognition accuracy

### Details

- **Dependencies:** None (critical path blocking other work)
- **Success criteria:** CPU usage < 20% on battery, no audio buffer overflows, accuracy maintained

---

## **Milestone 2: Text Quality Improvements (High Priority - Q1)**

**Goal:** Make recognized text readable with paragraphs and confidence-based selection

### Tasks

#### 2.1 Implement paragraph segmentation

- Use existing silence duration tracking ([AudioSource.py:151](src/AudioSource.py#L151), [165-170](src/AudioSource.py#L165-L170))
- Add configurable paragraph threshold (e.g., 2.0s silence)
- Create ParagraphBreakMarker to insert breaks in TextMatcher
- Update GuiWindow to render paragraph breaks

#### 2.2 Confidence-based text selection in overlaps

- Extract confidence scores from Parakeet model output
- Add `confidence: float` field to RecognitionResult dataclass
- Update TextMatcher.resolve_overlap() to prefer higher-confidence text
- Log confidence scores in verbose mode

#### 2.3 STT hallucination handling (multiple approaches)

- **Approach A:** Post-processing filter in TextMatcher
- **Approach B:** Confidence thresholding (drop low-confidence filler words)
- **Approach C:** Pattern detection in RecognitionResult
- **Approach D:** Configurable word blacklist in config JSON
- Research which approach works best, may combine multiple

### Details

- **Dependencies:** None (parallel to M1)
- **Estimated effort:** 3-4 weeks
- **Risk:** Low
- **Success criteria:** Readable paragraphs, fewer hallucinations, better overlap resolution

---

## **Milestone 3: UI/UX Enhancements (Medium Priority - Q2)**

**Goal:** Polish user interface with better feedback and display modes

### Tasks

#### 3.1 Improve model download progress display

- Capture HuggingFace Hub CLI output from stdout/stderr
- Parse progress info (download speed, ETA, bytes transferred)
- Update ModelDownloadDialog with real-time progress text
- Show both models' progress independently

#### 3.2 Timestamp insertion feature (STANDALONE)

- Add "Insert Timestamp" button in GuiWindow
- Format timestamps with configurable format (default: `[HH:MM:SS]`)
- Insert at current cursor position in text
- **Note:** Hotkey support moved to M4

#### 3.3 Subtitle/overlay display mode

- Create SubtitleWindow class (Tkinter transparent overlay)
- Large font, high contrast, always-on-top
- Toggle button in main GuiWindow
- Position configurable (top/bottom/center)

### Details

- **Dependencies:** None
- **Estimated effort:** 3 weeks
- **Risk:** Low
- **Success criteria:** Better download UX, timestamp insertion works, subtitle mode functional

---

## **Milestone 4: System Integration & Hotkeys (High Priority - Q2)**

**Goal:** Enable background operation with global hotkeys and text insertion into other apps

### Tasks

#### 4.1 Global hotkey framework

- Research cross-platform hotkey libraries (pynput, keyboard, system-specific)
- Implement HotkeyManager class
- Register configurable hotkeys (e.g., Ctrl+Shift+Space)
- Handle conflicts and permissions (accessibility on macOS/Linux)

#### 4.2 Text insertion into other applications

- Create TextInserter class (keyboard simulation)
- Detect active window/cursor position
- Simulate keyboard input to insert recognized text
- Handle different OS clipboard mechanisms

#### 4.3 Hotkey-driven workflow modes

- Push-to-talk mode (hold hotkey, speak, release → insert text)
- Toggle mode (press hotkey to start/stop recognition)
- Clipboard mode (copy instead of insert)

**4.4 Background mode optimization**

- Minimize GUI when running in background
- System tray icon with status indicator
- Low-latency wake-up from hotkey

### Details

- **Dependencies:** M1 (needs stable CPU usage for background operation)
- **Estimated effort:** 4 weeks
- **Risk:** Medium-High (OS permissions, keyboard simulation security restrictions)
- **Success criteria:** Global hotkeys work, text inserts into other apps, background mode stable

---

## **Milestone 5: MacOS Port (High Priority - Q3)**

**Goal:** Full MacOS support for both Intel and Apple Silicon architectures

### Tasks

**5.1 Platform compatibility research**

- Test all Python dependencies on macOS (sounddevice, tkinter, onnxruntime)
- Research universal binary vs. architecture-specific builds
- Investigate ONNX Runtime support for Apple Silicon
- Check OpenVINO support for macOS (may not be available)

**5.2 Build system for macOS**

- Update [build_distribution.py](build_distribution.py) for .app bundle creation
- Create PyInstaller spec for macOS
- Handle Info.plist configuration
- Research notarization requirements and costs

**5.3 MacOS-specific integrations**

- Implement macOS hotkey support (different from Windows/Linux)
- Request Accessibility permissions for text insertion
- Optional: Menu bar app integration
- Test on both Intel and Apple Silicon Macs

**5.4 Architecture decision**

- Determine if single universal binary is feasible
- If not, create separate Intel/ARM64 builds
- Document hardware requirements per architecture

### Details

- **Dependencies:** M1 (hardware acceleration), M4 (hotkey system)
- **Estimated effort:** 5-6 weeks
- **Risk:** High (Apple code signing costs ~$99/year, notarization complexity, architecture differences)
- **Success criteria:** App runs on macOS (Intel + ARM64), passes notarization, hotkeys work

---

## **Milestone 6: LLM Integration (High Priority - Q3-Q4)**

**Goal:** Post-process recognized text with local LLM for cleanup, formatting, summarization

### Tasks

**6.1 LLM integration architecture**

- Design plugin-based LLMProcessor interface
- Support multiple backends (llama.cpp, ollama, OpenAI API)
- Define async processing pipeline (non-blocking STT)
- Create configuration schema for LLM settings

**6.2 Text cleanup processor**

- Implement sentence formation (capitalize, punctuation)
- Grammar correction
- Remove STT artifacts (duplicates, fragments)
- Preserve technical terms and proper nouns

**6.3 Summarization & advanced features**

- Real-time summarization (paragraph-level)
- Meeting notes formatting (bullet points, action items)
- Contextual awareness (maintain conversation context)

**6.4 Model management**

- Download/manage local LLM models (Llama, Mistral, etc.)
- Model selection UI (different models for different tasks)
- Performance tuning (quantization, context length)
- Monitor LLM latency vs. STT latency (balance speed vs. quality)

### Details

- **Dependencies:** M2 (needs clean text input), M1 (CPU headroom for LLM)
- **Estimated effort:** 6-8 weeks
- **Risk:** Medium (LLM model size/distribution, performance on older hardware)
- **Success criteria:** Text cleanup works, summarization functional, <1s LLM latency

---

## **Priority Matrix**

| Priority | Milestone | Blocking Issues | User Impact |
|----------|-----------|-----------------|-------------|
| **0** | M0: Distribution Optimization | Quick win, improves distribution quality | Low |
| **1** | M1: Performance | **CRITICAL** - App unusable on battery | High |
| **2** | M2: Text Quality | Foundation for M6 | High |
| **3** | M4: System Integration | Key differentiator feature | High |
| **4** | M3: UI/UX | Polish, user experience | Medium |
| **5** | M5: MacOS Port | Platform expansion | High (for Mac users) |
| **6** | M6: LLM Integration | Major feature, depends on M1/M2 | Very High |

---

## **Recommended Execution Order**

### **Phase 0 - Quick Wins (Days 1-2)**

- **Days 1-2:** M0 Distribution Optimization (zip packaging) - Quick improvement

### **Phase 1 - Critical Fixes (Weeks 1-7)**

- **Weeks 1-4:** M1 Performance (NPU/GPU acceleration) - BLOCKING
- **Weeks 5-7:** M2 Text Quality (paragraphs, confidence, hallucinations)

### **Phase 2 - Core Features (Weeks 8-15)**

- **Weeks 8-11:** M4 System Integration (hotkeys, text insertion)
- **Weeks 12-15:** M6.1-6.2 LLM Integration (architecture + text cleanup)

### **Phase 3 - Polish & Expansion (Weeks 16-24)**

- **Weeks 16-18:** M3 UI/UX (progress bars, timestamps, subtitles)
- **Weeks 19-21:** M6.3-6.4 LLM Advanced Features (summarization, models)
- **Weeks 22-27:** M5 MacOS Port (platform expansion)

### **Phase 4 - Final Polish (Weeks 28-30)**

- Bug fixes, documentation, performance tuning
- Integration testing across all milestones
- Release preparation

---

## **Risk Assessment & Mitigation**

### High-Risk Items

1. **OpenVINO NPU support**
   - Risk: May not work well on specific Intel NPU models
   - Mitigation: Test early with sample models, maintain CPU fallback

2. **MacOS notarization**
   - Risk: $99/year + Apple bureaucracy complexity
   - Mitigation: Budget for Apple Developer Program, follow notarization guides

3. **Hotkey permissions**
   - Risk: Modern OSes restrict background keyboard access
   - Mitigation: Research existing tools (Talon, Dragon), request proper permissions

4. **LLM model size**
   - Risk: Could bloat distribution to 5-10GB
   - Mitigation: Separate LLM models as optional downloads, use quantized models

5. **Confidence scores from Parakeet**
   - Risk: Model may not expose confidence easily
   - Mitigation: Research onnx_asr API, may need raw logits processing

---

## **Key Architectural Changes**

### Milestone 1 - New Components

- `ExecutionProviderManager` - Manages NPU/GPU/CPU fallback
- `OpenVINOProvider` - OpenVINO integration layer

### Milestone 2 - Updates

- `RecognitionResult` - Add `confidence: float` field
- `TextMatcher` - Add `ParagraphBreakMarker` logic
- `GuiWindow` - Render paragraph breaks

### Milestone 4 - New Components

- `HotkeyManager` - Global hotkey registration
- `TextInserter` - Keyboard simulation for text insertion

### Milestone 6 - New Components

- `LLMProcessor` - Plugin interface for LLM backends
- `TextCleanupProcessor` - Sentence formation, grammar
- `SummarizationProcessor` - Meeting notes, summaries

---

## **Testing Strategy**

Each milestone should maintain the TDD approach:

1. **Write tests first** - Define expected behavior
2. **Implement to pass tests** - Focus on logic, not implementation details
3. **Integration tests** - Test component interactions
4. **Performance tests** - Measure CPU, latency, memory

### Test Coverage Targets

- M1: Performance benchmarks (CPU < 20%, no overflows)
- M2: Text quality metrics (paragraph detection, confidence accuracy)
- M3: UI tests (progress display, timestamp insertion)
- M4: Hotkey tests (registration, text insertion simulation)
- M5: Platform tests (macOS compatibility matrix)
- M6: LLM tests (cleanup quality, latency < 1s)

---

## **Success Metrics**

### Milestone 1

- ✅ CPU usage < 20% on battery
- ✅ Zero audio buffer overflows
- ✅ Recognition accuracy ≥ baseline

### Milestone 2

- ✅ Paragraphs correctly separated on 2s+ silence
- ✅ Confidence-based selection improves overlap quality
- ✅ Hallucinations reduced by >50%

### Milestone 3

- ✅ Download progress shows real-time updates
- ✅ Timestamp insertion works in GUI
- ✅ Subtitle mode renders correctly over other apps

### Milestone 4

- ✅ Hotkeys work in background
- ✅ Text inserts into other apps (notepad, browser, etc.)
- ✅ Push-to-talk latency < 500ms

### Milestone 5

- ✅ App runs on macOS Intel + Apple Silicon
- ✅ Passes Apple notarization
- ✅ Hotkeys work on macOS with proper permissions

### Milestone 6

- ✅ Text cleanup improves readability (user study)
- ✅ Summarization produces useful meeting notes
- ✅ LLM latency < 1s per paragraph

---


## **Future Considerations (Beyond Milestones)**

- Android/iOS mobile app (voice notes)
- Browser extension (dictation in web apps)
- Multi-speaker diarization (who said what)
- Custom vocabulary/domain adaptation
- Cloud sync for transcriptions
- API for third-party integrations

---

**Document Version:** 1.1
**Last Updated:** 2025-10-16
**Status:** Planning Phase (M0 in progress)
