# CLAUDE.md

Test driven development: first plan tests, design and create tests, implement code later. SOLID design is mandatory.

The test should focus on:

- Output behavior - Does it produce correct results?
- Logic - Does it do what expected?
- Queue interaction

Create tests to check the application code, never re-implement application logic in tests.
Avoid tests of parameters passed to methods, variables assignments and checks that make no sense.

## Comment and Docstring Guidelines

### Docstrings (Required)
- **Classes**: Purpose (1-2 sentences), responsibilities, observer pattern note if applicable
- **Methods**: Purpose, Args, Returns
- Use "Algorithm:" sections in method docstrings for explanations of complex multi-step logic

### Inline Comments (STRICTLY MINIMAL)
**Default rule: NO inline comments.** Code should be self-documenting through:
- Clear variable and function names
- Type hints
- Small, focused functions
- Docstrings with "Algorithm:" sections

**Only add inline comments for**:
1. **Critical decisions**: Magic numbers, tolerances, thresholds with justification
2. **External API quirks**: Behavior that contradicts expectations or documentation
3. **Performance/correctness trade-offs**: Why a specific approach was chosen over alternatives

**Strictly NEVER comment**:
- What the code does (let code speak for itself)
- Variable assignments, loops, conditionals, function calls
- State machine transitions (document in docstring instead)
- Type information (use type hints)
- Obvious operations like concatenation, indexing, arithmetic
- File headers or copyright notices
- Step-by-step algorithm flow
- In a file top when the comment is similar to the class docstring

### Preferred Pattern: Docstring Algorithm Section + Minimal Inline

For complex multi-step logic:
1. **Put algorithm overview in docstring** with numbered steps
2. **Keep inline comments RARE** - only for cryptic decisions

### Examples

✅ **Good** (critical threshold with justification):
```python
# 0.37s tolerance accounts for VAD RMS normalization delay
boundary_tolerance = 0.37
```

✅ **Good** (external API quirk):
```python
# Parakeet uses space-prefix tokenization: [" O", "ne"] = "One"
if not filtered_tokens[0].startswith(' '):
    word_start_idx = self._find_word_boundary(tokens, first_filtered_idx)
```

❌ **Bad** (repeating code with words):
```python
# Set word_start_idx to first_filtered_idx
word_start_idx = first_filtered_idx

# Loop through indices from first_filtered_idx - 1 to 0
for idx in range(first_filtered_idx - 1, -1, -1):
    # Check if token starts with space
    if tokens[idx].startswith(' '):
        # Found word boundary - this is the start of the word
        word_start_idx = idx
        break
```

❌ **Bad** (obvious operations):
```python
# Concatenate audio parts
full_audio = np.concatenate(audio_parts)

# Calculate duration
duration_with_context = len(full_audio) / self.sample_rate
```

### Refactoring Over Commenting
If you find yourself writing multiple inline comments to explain complex code:
1. **Move algorithm explanation to docstring**
2. **Use intermediate variables** with clear names
3. **Simplify conditional logic**

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
