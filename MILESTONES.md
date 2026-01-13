## **Milestone 3: Buikd for Microsoft Store

**Goal:** Build and publish the application in Mocrosoft Store


## **Milestone 4: Add features**

### Tasks

#### 4.1 Timestamp insertion

- Add "Insert Timestamp" button in GuiWindow
- Format timestamps with configurable format (default: `[HH:MM:SS]`)
- Design the main window to show timestamps

#### 4.2 Subtitle/overlay display mode

- Create SubtitleWindow class (Tkinter transparent overlay)
- Large font, high contrast
- Toggle button in main GuiWindow
- Position configurable

#### 4.3 Text insertion into other applications

- Create TextInserter class (keyboard simulation)
- Detect active window/cursor position
- Simulate keyboard input to insert recognized text
- Implement Hotkey
- Push-to-talk mode (hold hotkey, speak, release → insert text)
- Toggle mode (press hotkey to start/stop recognition)
- Clipboard mode (copy instead of insert)

---

## **Milestone 5: Speech recognition improvements
### Tasks

#### 5.1 STT hallucination handling (multiple approaches)

- **Approach A:** Post-processing filter
- **Approach B:** Confidence thresholding
- **Approach C:** Pattern detection in TextMatcher
- **Approach D:** Phrase blacklist

#### 5.2 Evaluate canary-1b-v2
https://huggingface.co/nvidia/canary-1b-v2

---

## **Milestone 6: MacOS Port (High Priority - Q3)**

**Goal:** Full MacOS support for both Intel and Apple Silicon architectures

### Tasks

**6.1 Platform compatibility research**

- Add CoreML execution provider
- Test all Python dependencies on macOS (sounddevice, tkinter)
- Research universal binary vs. architecture-specific builds (Intel/M)

**6.2 Build system for macOS**

- Create a build script for MacOS
- Make build scripts include different execution providers for different platforms
- Handle Info.plist configuration
- Research notarization requirements and costs

**6.3 MacOS-specific integrations**

- Implement macOS hotkey support
- Request Accessibility permissions for text insertion
- Optional: Menu bar app integration

**6.4 Architecture decision**

- Determine if single universal binary is feasible
- If not, create separate Intel/ARM64 builds
- Document hardware requirements per architecture

---

## **Milestone 7: LLM Integration**

**Goal:** Post-process recognized text with local LLM for cleanup, formatting, summarization

### Tasks

**7.1 LLM integration architecture**

- Design plugin-based LLMProcessor interface
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

