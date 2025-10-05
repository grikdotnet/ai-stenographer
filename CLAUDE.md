# CLAUDE.md

Test driven development only: plan, design and create tests first, implement code after tests are ready. SOLID design is mandatory.
Work only on a task you were given. Before adding features on your own initiative, offer and ask.

The test should focus on:

- Output behavior - Does it produce correct text results?
- Logic - Does it do what expected?
- Queue interaction - Are results properly queued?

Write tests to check the application code, never re-implement application logic in tests.
Avoid tests of parameters passed to methods, variables assignments and other checks that make no sense.

Create dockblocks to describe logic and algorithms. Avoid commenting separate operations, skip comments which  duplicate function names and variables.

Define types of parameters and return if possible.

Instead of writing "You're absolutely correct" or similar, write "OK".

Instead of claiming that everything works evaluate what may be wrong, and offer what to check.

After creating git commit mesages replace "Generated with [Claude Code] ..." line or "Co-Authored-By" with a joke.

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

```bash
python download_model.py
```

Downloads the Parakeet STT model and Silero VAD model:
- Parakeet: `./models/parakeet/`
- Silero VAD: `./models/silero_vad/`

## Architecture

The system uses a **pipeline architecture** with threaded components communicating via queues:

### Core Pipeline Flow

1. **AudioSource + VAD** → captures 32ms frames, detects speech → preliminary `AudioSegment` → `chunk_queue`
2. **AudioSource** → calls **AdaptiveWindower** directly (no thread) → aggregates into finalized `AudioSegment` → `chunk_queue`
3. **Recognizer** → processes both preliminary and finalized AudioSegments → `RecognitionResult` → `text_queue`
4. **TextMatcher** → filters/processes text with word-level timing → `final_queue` + `partial_queue`
5. **TwoStageDisplayHandler** → displays final and partial results

**Key Architecture:** Single `chunk_queue` receives both preliminary (instant) and finalized (high-quality) `AudioSegment` instances. AdaptiveWindower is called synchronously by AudioSource rather than running in a separate thread.

### Voice Activity Detection (VAD) Flow

AudioSource now uses Silero VAD for word-level speech segmentation:

1. **32ms Frame Capture**: Microphone delivers 32ms audio frames (512 samples @ 16kHz)
2. **VAD Processing**: Each frame processed through Silero VAD ONNX model
3. **Speech Detection**:
   - Frames with speech probability > 0.5 are accumulated
   - Minimum speech duration: 64ms (2 frames) ensures temporal consistency
   - Silence timeout: 32ms (1 frame) creates word-level cuts
   - Maximum speech duration: 3000ms (3s) matches recognition window
4. **Segment Emission**: Complete speech segments emitted as preliminary `AudioSegment` instances
5. **Window Aggregation**: AdaptiveWindower combines words into 3s overlapping finalized `AudioSegment` instances

### Data Types

**`src/types.py`** defines the core data structures:

- **`AudioSegment`**: Unified dataclass for both preliminary and finalized segments
  - `type: Literal['preliminary', 'finalized']` - Discriminator field
  - `data: np.ndarray` - Audio samples (float32, -1.0 to 1.0)
  - `start_time: float` - Segment start in seconds
  - `end_time: float` - Segment end in seconds
  - `chunk_ids: list[int]` - Tracking IDs (preliminary=[single_id], finalized=[id1, id2, ...])

- **`RecognitionResult`**: Speech recognition output
  - `text: str` - Recognized text
  - `start_time: float` - Recognition start time
  - `end_time: float` - Recognition end time
  - `is_preliminary: bool` - True for instant results, False for high-quality results

### Key Components

- **`src/types.py`**: Core dataclass definitions for AudioSegment and RecognitionResult
- **`src/pipeline.py`**: Main `STTPipeline` class that orchestrates all components
- **`src/AudioSource.py`**: Microphone capture + VAD, creates preliminary AudioSegments with unique chunk IDs
- **`src/VoiceActivityDetector.py`**: Silero VAD ONNX for word-level speech segmentation
- **`src/AdaptiveWindower.py`**: Aggregates preliminary AudioSegments into finalized windows with merged chunk_ids
- **`src/Recognizer.py`**: Parakeet ONNX model, processes both preliminary and finalized AudioSegments
- **`src/TextMatcher.py`**: Text processing with word-level timing preservation
- **`src/TwoStageDisplayHandler.py`**: Console output for final and partial results

### Threading Model

Components run in separate threads (except AdaptiveWindower which is called directly):
- **AudioSource**: Thread with microphone callback, creates preliminary AudioSegments
- **Recognizer**: Thread processing chunk_queue (both preliminary and finalized AudioSegments)
- **TextMatcher**: Thread processing text_queue RecognitionResults
- **TwoStageDisplayHandler**: Thread displaying final/partial text
- **AdaptiveWindower**: No thread - called synchronously by AudioSource for each preliminary segment

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
    "min_speech_duration_ms": 64,
    "max_speech_duration_ms": 3000,
    "silence_timeout_ms": 32
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
- `min_speech_duration_ms: 64` - Minimum 2 frames for temporal consistency, filters transient noise
- `silence_timeout_ms: 32` - Cut at word boundaries (single frame of silence)
- `max_speech_duration_ms: 3000` - Match recognition window, force splits on continuous speech

**Windowing Parameters:**

- `window_duration: 3.0` - 3-second recognition windows provide sufficient context
- `step_size: 1.0` - 1-second step creates 33% overlap for continuity

**Design Rationale:**

1. **Word-Level Granularity**: 32ms silence timeout segments speech into individual words
2. **Temporal Consistency**: 64ms minimum (2 frames) distinguishes real speech from clicks
3. **Recognition Context**: 3s windows provide enough context for accurate STT
4. **Responsiveness**: 1s step size ensures quick updates while maintaining continuity

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

- **`tests/conftest.py`**: Pytest fixtures for test data (downloads Silero's en.wav once)
- **`tests/test_vad.py`**: 3 unit tests for VoiceActivityDetector
- **`tests/test_audiosource_vad.py`**: 3 integration tests for AudioSource + VAD (uses AudioSegment)
- **`tests/test_adaptive_windower.py`**: 6 tests for AdaptiveWindower (uses AudioSegment with chunk_ids)
- **Real audio**: Tests use downloaded speech samples, not synthetic audio
- **Type safety**: All tests use AudioSegment and RecognitionResult dataclasses

## Dependencies

- sounddevice: Audio capture
- numpy: Audio data processing
- onnx_asr: Parakeet model inference
- onnxruntime: Silero VAD inference
- soundfile: Audio file I/O for tests
- scipy: Audio resampling in tests
- queue, threading: Pipeline coordination
- pytest: Testing framework