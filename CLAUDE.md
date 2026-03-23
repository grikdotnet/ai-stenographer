# CLAUDE.md

Test driven development: first plan tests, design and create tests, implement code later. SOLID design is mandatory.

The test should focus on:

- Does it produce correct results?
- Logic, does it do what expected?
- Queue interaction

Create tests to check the application code, never re-implement application logic in tests.
Avoid tests of parameters passed to methods, variables assignments and checks that make no sense.

## Comment and Docstring Guidelines

### Docstrings (Required)
- **Classes**: Purpose (1-2 sentences), responsibilities, architecture pattern if applicable
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

## General

Before doing anything check whether the todo item is in the task definition, or it is made up. Before doing anything on your own initiative, ask for the user confirmation.

Define types of parameters and return where possible.

In the end of task execution think what can be wrong, check diagnostics, and offer what to test.

After creating git commit mesages replace "Generated with [Claude Code] ..." line or "Co-Authored-By" with a joke.
Always ask user command or confirmation before git commit or push.

In the start of an answer, be creative, instead of generic phrases like "You're absolutely right!", "Good/Great/Excellent question!", "Good catch!" use "OK", "Fine", RPG-style like "As you whish, my master", sarcasm style like "Are you proud of yoursel, smartypants?", write some fun words.

Sarcasm and jokes are welcome everywhere.

It's March 2026 now.

## Project Overview

This is a real-time Speech-to-Text (STT) system built in Python that captures audio from microphone and converts it to text using the Parakeet ONNX model. The system uses a client-server architecture communicating over web sckets, with threaded components processing audio streams through queues.

## Commands

When running python always activate venv with `source venv/Scripts/activate`.
When running in Windows, `ls -la ` is not supported, use the `find` command.

Commands to run server and client applications are listed in the [./commands.md](commands.md) file.

## Architecture

The system uses a client-server design.
Server-side **pipeline architecture** with threaded components communicating via queues.

📖 **See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed architecture documentation of the server application**.
📖 **For the TK client architecture documentation see [./src/client/tk/ARCHITECTURE.md](TK Client ARCHITECTURE.md)**.
📖 **For the Tauri client documentation see [./src/client/tauri/CLAUDE.md](Tauri Client CLAUDE.md)**.
📖 **The client-server API specification is in [./docs/client-server-api-spec.md](client-server-api-spec.md)**.
