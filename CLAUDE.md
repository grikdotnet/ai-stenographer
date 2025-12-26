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

In the start of an answer, be creative, instead of generic phrases like "You're absolutely right!", "Good/Great/Excellent question!", "Good catch!" use "OK", "Fine", RPG-style like "As you whish, my master", sarcasm style like "Are you proud of yoursel, smartypants?", write some fun words.

Sarcasm and jokes are welcome everywhere.

It's December 2025 now.

## Project Overview

This is a real-time Speech-to-Text (STT) system built in Python that captures audio from microphone and converts it to text using the Parakeet ONNX model. The system uses a pipeline architecture with threaded components processing audio streams through queues.

## Commands

### Running the Application

Always activate venv when running python with `source venv/Scripts/activate`.

```bash
python main.py
```

Starts the real-time STT pipeline.

### Verbose Mode

```bash
python main.py -v
```

Runs the pipeline with detailed logging for debugging text duplication issues. Shows:

### File Input (Testing Mode)

```bash
python main.py --input-file=path/to/audio.wav -v
```

Uses a WAV file as input instead of microphone for reproducible testing:

- `--input-file=path` - Path to WAV file (replaces microphone input)
- Audio is automatically resampled to 16kHz if needed
- Stereo files are converted to mono (first channel)
- Chunks are fed to pipeline at real-time rate (32ms per chunk)
- Useful for testing timestamp accuracy and model behavior

**Example with test audio:**
```bash
python main.py --input-file=tests/fixtures/en.wav -v
```

### Testing

```bash
python -m pytest tests/
```

Runs all tests in the tests directory.

## Architecture

The system uses a **pipeline architecture** with threaded components communicating via queues.

📖 **See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation** including diagrams, threading model, data flow, and design principles.

### Quick Reference

**Core Pipeline Flow:**

1. **AudioSource** → captures 32ms frames → raw audio dict → `chunk_queue`
2. **SoundPreProcessor** → reads `chunk_queue` → RMS normalization + VAD + buffering → preliminary `AudioSegment` → `speech_queue`
3. **SoundPreProcessor** → calls **AdaptiveWindower** (sync) → finalized `AudioSegment` → `speech_queue`
4. **Recognizer** → reads `speech_queue` → processes AudioSegments → `RecognitionResult` → `text_queue`
5. **TextMatcher** → filters/processes text → GUI display (via `GuiWindow`)

**Key Components:** `AudioSource`, `SoundPreProcessor`, `VoiceActivityDetector`, `AdaptiveWindower`, `Recognizer`, `TextMatcher`, `GuiWindow`

**Data Types:** `AudioSegment` (preliminary/finalized), `RecognitionResult` (with `status` field)

## Technical Specifications

### Audio Parameters

- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Frame duration: 32ms (512 samples) - matches Silero VAD requirements
- All audio arrays are numpy float32, range -1.0 to 1.0

### VAD Configuration

Located in `config/stt_config.json` in section "vad".

**VAD Parameters Explained:**

- `frame_duration_ms: 32` - Silero VAD's optimal frame size
- `threshold: 0.5` - Speech probability threshold (0.5 = 50% confidence)
- `silence_energy_threshold: 1.5` - Cumulative silence probability threshold for segment finalization
- `max_speech_duration_ms: 3000` - Maximum segment duration; triggers intelligent breakpoint splitting at silence boundaries

**Windowing Parameters:**

- `window_duration` - duration of aggregation windows for recognition (3.0 seconds creates 3-second windows with automatic overlap based on trailing segment duration, MIN_OVERLAP_DURATION = 1.0s)

**Design Rationale:**

1. **Silence Energy**: Cumulative probability scoring ensures robust silence detection
2. **Recognition Context**: left and right context is sound before and after the speech data, required by STT models for better speech recognition
4. **Intelligent Splitting**: Splitting the speech by searching backward for natural breakpoints to avoid mid-word cuts

### Model Requirements

**Parakeet STT Model:**
- Model: nemo-parakeet-tdt-0.6b-v3 (ONNX format)
- Location: `./models/parakeet/`

**Silero VAD Model:**
- Model: silero_vad.onnx (quantized to f16)
- Location: `./models/silero_vad/`

### Testing Infrastructure

**Core Pipeline Tests:**
- **`tests/conftest.py`**: Pytest fixtures for test data (downloads Silero's en.wav once)
- **`tests/test_vad.py`**: Unit tests for VoiceActivityDetector
- **`tests/test_sound_preprocessor.py`**: Comprehensive tests for SoundPreProcessor (VAD integration, RMS normalization, speech buffering, intelligent breakpoint splitting)
- **`tests/test_adaptive_windower.py`**: Tests for AdaptiveWindower (window aggregation with chunk_ids)
- **`tests/test_gui_window.py`**: Tests for GuiWindow (preliminary/final text display)

**Model Download Tests:**
- **`tests/test_model_manager.py`**: 8 tests for ModelManager (model detection and download)
- **`tests/test_gui_factory.py`**: 4 tests for GuiFactory (shared GUI infrastructure)
- **`tests/test_model_download_dialog.py`**: 6 tests for ModelDownloadDialog (GUI download flow)
- **`tests/test_main_integration.py`**: 3 integration tests for main.py startup flow

**Test Principles:**
- **TDD approach**: Tests written before implementation
- **Real audio**: Tests use real speech samples, not synthetic audio
- **Type safety**: All tests use AudioSegment and RecognitionResult dataclasses

## Key Components

**Core Pipeline:**
- **`src/pipeline.py`**: Main STTPipeline orchestrator
- **`src/AudioSource.py`**: Microphone audio capture (simplified, no VAD)
- **`src/SoundPreProcessor.py`**: VAD processing, RMS normalization, speech buffering
- **`src/VoiceActivityDetector.py`**: Silero VAD for speech detection (used by SoundPreProcessor)
- **`src/AdaptiveWindower.py`**: Window aggregation (called synchronously by SoundPreProcessor)
- **`src/Recognizer.py`**: Parakeet STT inference with context-aware recognition
- **`src/TextMatcher.py`**: Text overlap resolution and routing (preliminary/final)
- **`src/GuiWindow.py`**: Two-stage text display (preliminary/final)

**Data Types:**
- **`src/types.py`**: AudioSegment, RecognitionResult dataclasses