## **Milestone 7: MacOS Port and other models**

**Goal:** Full MacOS support for both Intel and Apple Silicon architectures

### Tasks

**7.1 Platform compatibility research**

- Find the CoreML execution provider implementation

**7.2 Evaluate other ASR models**

Canary-1b-v2
Qwen3-asr
Vibevoice-ASR

**7.2 Build system for macOS**

- Create a build script for MacOS
- Make build scripts include different execution providers for different platforms
- Handle Info.plist configuration
- Research notarization requirements and costs

**7.3 MacOS-specific integrations**

- Implement macOS hotkey support
- Request Accessibility permissions for text insertion
- Optional: Menu bar app integration

**7.4 Architecture decision**

- Determine if single universal binary is feasible
- If not, create separate Intel/ARM64 builds
- Document hardware requirements per architecture

---

## **Milestone 8: LLM Integration**

**Goal:** Post-process recognized text with local LLM for cleanup, formatting, summarization

### Tasks


**8.1 LLM integration architecture**

- Design plugin-based LLMProcessor interface
- Define async processing pipeline (non-blocking STT)
- Create configuration schema for LLM settings

**8.2 Text cleanup processor**

- Implement sentence formation (capitalize, punctuation)
- Grammar correction
- Remove STT artifacts (duplicates, fragments)
- Preserve technical terms and proper nouns

**8.3 Summarization & advanced features**

- Real-time summarization (paragraph-level)
- Meeting notes formatting (bullet points, action items)
- Contextual awareness (maintain conversation context)
