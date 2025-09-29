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

Never praise, avoid writing "you are right", and that everything works.
When the task is complete, evaluate what may be wrong with the solution, offer what to check.

## Project Overview

This is a real-time Speech-to-Text (STT) system built in Python that captures audio from microphone and converts it to text using the Parakeet ONNX model. The system uses a pipeline architecture with threaded components processing audio streams through queues.

## Commands

### Running the Application

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

Downloads the Parakeet model to `./models/parakeet/` directory.

## Architecture

The system uses a **pipeline architecture** with threaded components communicating via queues:

### Core Pipeline Flow

1. **AudioSource** → captures microphone audio chunks → `chunk_queue`
2. **Windower** → processes chunks into recognition windows → `window_queue`
3. **Recognizer** → runs STT on windows using Parakeet model → `text_queue`
4. **TextMatcher** → filters/processes text → `final_queue` + `partial_queue`
5. **TwoStageDisplayHandler** → displays final and partial results

### Key Components

- **`src/pipeline.py`**: Main `STTPipeline` class that orchestrates all components
- **`src/AudioSource.py`**: Microphone audio capture using sounddevice
- **`src/Windower.py`**: Buffers audio chunks into recognition windows with VAD
- **`src/Recognizer.py`**: Parakeet ONNX model wrapper for speech recognition
- **`src/TextMatcher.py`**: Text processing and filtering logic
- **`src/TwoStageDisplayHandler.py`**: Console output for final and partial results

### Threading Model

Each pipeline component runs in its own thread, processing data from input queues and pushing to output queues. This allows real-time processing with minimal latency.

## Technical Specifications

### Audio Parameters

- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Chunk duration: 100ms
- All audio arrays are numpy float32, range -1.0 to 1.0

### Model Requirements

- Model: nemo-parakeet-tdt-0.6b-v3 (ONNX format)
- Location: `./models/parakeet/`
- Library: onnx_asr

### Testing Infrastructure

- **`tests/audio_fixture.py`**: `MockAudioStream` class for simulating microphone input
- Test files should use the mock audio stream to avoid requiring actual microphone hardware

## Dependencies

- sounddevice: Audio capture
- numpy: Audio data processing
- onnx_asr: Parakeet model inference
- queue, threading: Pipeline coordination