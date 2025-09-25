# CLAUDE.md

Avoid commenting every line and comments that duplicate the function names, in comments describe logic and algorithms rather than functions.
Test driven development only - plan, design and create tests first, implement code when tests are ready.

Avoid tests of parameters passed to methods, variables assignments and other checks that make no sense. The test should focus on:

- Output behavior - Does it produce correct text results?
- Logic - Does it do what expected?
- Queue interaction - Are results properly queued?

## Project Overview

This is a real-time Speech-to-Text (STT) system built in Python that captures audio from microphone and converts it to text using the Parakeet ONNX model. The system uses a pipeline architecture with threaded components processing audio streams through queues.

## Commands

### Running the Application

```bash
python main.py
```

Starts the real-time STT pipeline. Press Ctrl+C to stop.

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
5. **OutputHandlers** → displays final and partial results

### Key Components

- **`src/pipeline.py`**: Main `STTPipeline` class that orchestrates all components
- **`src/AudioSource.py`**: Microphone audio capture using sounddevice
- **`src/Windower.py`**: Buffers audio chunks into recognition windows with VAD
- **`src/Recognizer.py`**: Parakeet ONNX model wrapper for speech recognition
- **`src/TextMatcher.py`**: Text processing and filtering logic
- **`src/OutputHandlers.py`**: Console output for final and partial results

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