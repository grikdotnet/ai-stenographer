# CLAUDE.md

Test driven development: first plan tests, design and create tests, implement code later. SOLID design is mandatory.

The test should focus on:

- Output behavior - Does it produce correct results?
- Logic - Does it do what expected?
- Queue interaction

Create tests to check the application code, never re-implement application logic in tests.
Avoid tests of parameters passed to methods, variables assignments and other checks that make no sense.

Create dockblocks to describe logic and algorithms. Comment logic instead of commenting code.
Avoid comments duplicating code, example:
|  # RMS normalization
|  rms_config = config['audio'].get('rms_normalization', {})

Before doing anything check whether the tofo item is in the task definition, or it is made up. Before doing anything on your own initiative, ask for the user confirmation.

Define types of parameters and return where possible.

In the end of task execution think what can be wrong, and offer what to check.

After creating git commit mesages replace "Generated with [Claude Code] ..." line or "Co-Authored-By" with a joke.
Never commit automatically without user command or confirmation.

In the start of an answer, instead of "You're absolutely right!", "Good/Great/Excellent question!", "Good catch!" and other a generic phrases be creative - use "OK", "Fine", RPG-style like "As you whish, my master", sarcasm like "Are you proud of yoursel, smartypants?", "What options do I have?".

Sarcasm and jokes are welcome everywhere.

## Project Overview

This is a real-time Speech-to-Text (STT) system built in Python that captures audio from microphone and converts it to text using the Parakeet ONNX model. The system uses a pipeline architecture with threaded components processing audio streams through queues.

## Commands

### Running the Application

Always activate venv when running python with `source venv/Scripts/activate`.

```bash
python main.py
```

Starts the real-time STT pipeline. Press Ctrl+C to stop.

### Verbose Mode

```bash
python main.py -v
```

Runs the pipeline with detailed logging for debugging text duplication issues. Shows:

- `windower(): creating window, timestamp={timestamp}, len={length}, hash={hash}, chunk_ids={[1,2,3...]}`
- `recognize(): window_timestamp={timestamp}, audio_len={length}, audio_hash={hash}, chunk_ids={[1,2,3...]}`
- `recognize(): text: '{text}'` - Text recognition results
- `process_text() current_text: received '{text}'` - TextMatcher input
- `process_text() previous_text: '{previous_text}'` - TextMatcher state
- `sending to partial_queue: {partial_data}` - Partial text output
- `sending to final_queue: {final_data}` - Final text output

### Windowing Configuration

```bash
python main.py --window=2.0 --step=1.0
```

Configures windowing parameters to reduce text duplication:

- `--window=X.X` - Window duration in seconds (default: 2.0)
- `--step=X.X` - Step size in seconds (default: 1.0, which gives 50% overlap)
- For less duplication, use larger step sizes: `--step=1.5` (25% overlap) or `--step=2.0` (no overlap)

### Testing

```bash
python -m pytest tests/
```

Runs all tests in the tests directory.

### Model Setup

Models are downloaded automatically when you first run the application. If models are missing, a GUI dialog will appear:

```bash
python main.py
```

**First run (models missing):**
- Dialog appears showing required models (~2GB total)
- Click "Download Models" to download automatically
- Progress bars show download status
- Pipeline starts after successful download

**Subsequent runs:**
- No dialog shown if models already exist
- Pipeline starts immediately

The application downloads:
- Parakeet STT Model (~1.5GB) → `./models/parakeet/`
- Silero VAD Model (~0.2GB) → `./models/silero_vad/`

## Architecture

The system uses a **pipeline architecture** with threaded components communicating via queues.

📖 **See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation** including diagrams, threading model, data flow, and design principles.

### Quick Reference

**Core Pipeline Flow:**

1. **AudioSource + VAD** → preliminary `AudioSegment` → `chunk_queue`
2. **AdaptiveWindower** (sync call) → finalized `AudioSegment` → `chunk_queue`
3. **Recognizer** → `RecognitionResult` → `text_queue`
4. **TextMatcher** → GUI display (via `GuiWindow`)

**Key Components:** `AudioSource`, `VoiceActivityDetector`, `AdaptiveWindower`, `Recognizer`, `TextMatcher`, `GuiWindow`

**Data Types:** `AudioSegment` (preliminary/finalized), `RecognitionResult` (with `is_preliminary` flag)

## Technical Specifications

### Audio Parameters

- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Frame duration: 32ms (512 samples) - matches Silero VAD requirements
- All audio arrays are numpy float32, range -1.0 to 1.0

### VAD Configuration

Located in `config/stt_config.json`:

```json
{
  "audio": {
    "sample_rate": 16000,
    "chunk_duration": 0.032
  },
  "vad": {
    "model_path": "./models/silero_vad/silero_vad.onnx",
    "frame_duration_ms": 32,
    "threshold": 0.5,
    "max_speech_duration_ms": 3000,
    "silence_energy_threshold": 1.5
  },
  "windowing": {
    "window_duration": 3.0,
    "step_size": 1.0
  }
}
```

**VAD Parameters Explained:**

- `frame_duration_ms: 32` - Silero VAD's optimal frame size
- `threshold: 0.5` - Speech probability threshold (0.5 = 50% confidence)
- `silence_energy_threshold: 1.5` - Cumulative silence probability threshold for segment finalization
- `max_speech_duration_ms: 3000` - Match recognition window, force splits on continuous speech

**Windowing Parameters:**

- `window_duration: 3.0` - 3-second recognition windows provide sufficient context
- `step_size: 1.0` - 1-second step creates 33% overlap for continuity

**Design Rationale:**

1. **Silence Energy**: Cumulative probability scoring ensures robust silence detection
2. **Recognition Context**: 3s windows provide enough context for accurate STT
3. **Responsiveness**: 1s step size ensures quick updates while maintaining continuity

### Model Requirements

**Parakeet STT Model:**
- Model: nemo-parakeet-tdt-0.6b-v3 (ONNX format)
- Location: `./models/parakeet/`
- Library: onnx_asr

**Silero VAD Model:**
- Model: silero_vad.onnx (quantized to f16)
- Location: `./models/silero_vad/`
- Library: onnxruntime
- Source: HuggingFace (onnx-community/silero-vad)

### Testing Infrastructure

**Core Pipeline Tests:**
- **`tests/conftest.py`**: Pytest fixtures for test data (downloads Silero's en.wav once)
- **`tests/test_vad.py`**: 3 unit tests for VoiceActivityDetector
- **`tests/test_audiosource_vad.py`**: 3 integration tests for AudioSource + VAD (uses AudioSegment)
- **`tests/test_adaptive_windower.py`**: 6 tests for AdaptiveWindower (uses AudioSegment with chunk_ids)
- **`tests/test_gui_window.py`**: 22 tests for GuiWindow (preliminary/final text display)

**Model Download Tests:**
- **`tests/test_model_manager.py`**: 8 tests for ModelManager (model detection and download)
- **`tests/test_gui_factory.py`**: 4 tests for GuiFactory (shared GUI infrastructure)
- **`tests/test_model_download_dialog.py`**: 6 tests for ModelDownloadDialog (GUI download flow)
- **`tests/test_main_integration.py`**: 3 integration tests for main.py startup flow

**Test Principles:**
- **Real audio**: Tests use downloaded speech samples, not synthetic audio
- **Type safety**: All tests use AudioSegment and RecognitionResult dataclasses
- **TDD approach**: Tests written before implementation

## Dependencies

- sounddevice: Audio capture
- numpy: Audio data processing
- onnx_asr: Parakeet model inference
- onnxruntime: Silero VAD inference
- soundfile: Audio file I/O for tests
- scipy: Audio resampling in tests
- queue, threading: Pipeline coordination
- pytest: Testing framework
- tkinter: GUI (model download dialog, text display)
- huggingface_hub: Model downloads from HuggingFace

## Key Components

**Core Pipeline:**
- **`src/pipeline.py`**: Main STTPipeline orchestrator
- **`src/AudioSource.py`**: Microphone audio capture with VAD integration
- **`src/VoiceActivityDetector.py`**: Silero VAD for speech detection
- **`src/AdaptiveWindower.py`**: Segment finalization logic
- **`src/Recognizer.py`**: Parakeet STT inference
- **`src/TextMatcher.py`**: Duplicate text filtering
- **`src/GuiWindow.py`**: Two-stage text display (preliminary/final)

**Model Management:**
- **`src/ModelManager.py`**: Model detection, download, and validation
- **`src/ModelDownloadDialog.py`**: GUI dialog for downloading missing models
- **`src/GuiFactory.py`**: Shared GUI component factory

**Data Types:**
- **`src/types.py`**: AudioSegment, RecognitionResult dataclasses