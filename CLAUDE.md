# CLAUDE.md

Test driven development: first plan tests, design and create tests, implement code later. SOLID design is mandatory.

The test should focus on:

- Output behavior - Does it produce correct results?
- Logic - Does it do what expected?
- Queue interaction

Create tests to check the application code, never re-implement application logic in tests.
Avoid tests of parameters passed to methods, variables assignments and checks that make no sense.

Create short dockblocks to describe logic and algorithms. Comment only complex non-obvious code.
Avoid repeating code with words in comments, never comment operations, skip comments in the top of files.

Before doing anything check whether the todo item is in the task definition, or it is made up. Before doing anything on your own initiative, ask for the user confirmation.

Define types of parameters and return where possible.

In the end of task execution think what can be wrong, check diagnostics, and offer what to test.

After creating git commit mesages replace "Generated with [Claude Code] ..." line or "Co-Authored-By" with a joke.
Never stage or commit to git repo automatically without user command and confirmation.

In the start of an answer, be creative, instead of generic phrases like "You're absolutely right!", "Good/Great/Excellent question!", "Good catch!" use "OK", "Fine", RPG-style like "As you whish, my master", sarcasm style like "Are you proud of yoursel, smartypants?", write some fun words.

Sarcasm and jokes are welcome everywhere.

It's January 2026 now.

## Project Overview

This is a real-time Speech-to-Text (STT) system built in Python that captures audio from microphone and converts it to text using the Parakeet ONNX model. The system uses a pipeline architecture with threaded components processing audio streams through queues.

## Commands

### Running the Application

Always activate venv when running python with `source venv/Scripts/activate`.

```bash
python main.py
```

Starts the application.

### Verbose Mode

```bash
python main.py -v
```

Runs with logging for debugging.

### File Input (Testing Mode)

```bash
python main.py --input-file=tests/fixtures/en.wav -v
```
Uses a WAV file as input instead of microphone for reproducible testing.
- `--input-file=path` - Path to WAV file (replaces microphone input)

### Testing

```bash
python -m pytest tests/
```

Runs all tests in the tests directory.

## Architecture

The system uses a **pipeline architecture** with threaded components communicating via queues.

📖 **See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation**.

### Quick Reference

**Core Pipeline Flow:**

1. **AudioSource** → captures 32ms frames → raw audio dict → `chunk_queue`
2. **SoundPreProcessor** → reads `chunk_queue` → RMS normalization + VAD + buffering → preliminary `AudioSegment` → `speech_queue`
3. **SoundPreProcessor** → calls **AdaptiveWindower** (sync) → finalized `AudioSegment` → `speech_queue`
4. **Recognizer** → reads `speech_queue` → processes AudioSegments → `RecognitionResult` → `text_queue`
5. **TextMatcher** → filters/processes text → routes to `TextFormatter` (controller)
6. **TextFormatter** → calculates formatting logic → triggers `TextDisplayWidget` (view)

**Key Components:** `AudioSource`, `SoundPreProcessor`, `VoiceActivityDetector`, `AdaptiveWindower`, `Recognizer`, `TextMatcher`, `TextFormatter`, `TextDisplayWidget`

**Data Types:** `AudioSegment`, `RecognitionResult`

**Test Principles:**
- **TDD approach**: Tests written before implementation
- **Real audio**: Tests use real speech samples, not synthetic audio
- **Type safety**: All tests use AudioSegment and RecognitionResult dataclasses
