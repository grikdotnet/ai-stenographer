## Project Context

This is a real-time Python STT pipeline using Parakeet ONNX with threaded components communicating via queues.

`ARCHITECTURE.md` contains detailed documentation of the system structure.

## Execution and Scope

- Do not refactor unrelated code unless required for correctness.
- Before acting on any TODO, verify it is in the task definition and not invented.
- Never stage, commit, or amend git repo automatically without explicit user approval or command.

## Engineering Rules

- Use TDD flow: plan tests first, write/adjust tests, then implement.
- Single Responsibility and Interface Segregation design principles are mandatory for the code design.
- Define parameter and return types where possible; use type hints on new/changed functions.
- Prefer existing project datatypes (for example from `src/types.py`) over ad-hoc structures.
- When variable is used in a single method, prefer a local varible over class property.

**Simplicity First**

Minimum code that solves the problem. Nothing speculative.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't explicitly requested.
No error handling for impossible scenarios.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

**Surgical Changes**

Touch only what you must. Clean up only your own edits.
Don't refactor code that wasn't requested.
Before doing anything check whether the todo item is in the task definition (plan).

### Planning Architecture Checks

- During planning, classify every new or changed class into one of the 5 server-side layers documented in the Appendix of `ARCHITECTURE.md`.
- Reject dependencies from a lower layer to a higher layer. Higher layers may depend on lower layers; lower layers must not depend on higher layers.
- Avoid skipping two layers when introducing dependencies. The intentional exemptions are the Entry-Point as composition root and the Protocol / Wire layer as a cross-cutting boundary.
- If an existing design violates these rules, propose a fix (move code or introduce a protocol) instead of adding another exception.
- Use `ARCHITECTURE.md` as the canonical source for layer definitions, litmus questions, and detailed examples.

## Testing Policy

- MUST validate behavioral outcomes (correct results) and expected logic.
- MUST cover queue interaction when queue behavior is part of the change.
- SHOULD avoid brittle tests tied to implementation details.
- MUST NOT re-implement production logic in tests.
- MUST NOT write tests that only assert trivial parameter passing, assignments, or meaningless checks.
- Prefer real audio fixtures over synthetic audio when validating STT behavior.

## Docstrings and Comments

### Docstrings
- Public classes and methods: required.
- Complex private methods: required.
- Trivial private helpers: concise docstring optional.
- Class docstrings: purpose (1-2 sentences), responsibilities, observer-pattern note if applicable.
- Method docstrings: purpose, Args, Returns.
- Add an `Algorithm:` section for complex multi-step logic.

### Inline Comments
- Default: no inline comments.
- Inline comments are good for:
  - Critical decisions (thresholds, tolerances, magic numbers with justification)
  - External API quirks or contradictions
  - Performance vs correctness trade-offs
- Never comment obvious operations, assignments, loops, conditionals, or type information.
- If many comments seem necessary, refactor for clarity.

## Runtime and Commands (PowerShell)

- Activate venv before Python commands. 
In Windows environment:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\venv\Scripts\Activate.ps1
```

Commands to run server and client applications are listed in the [./commands.md](commands.md) file.

## End-of-Task Verification

Before finishing a task, provide a summary:
- What changed and why
- What was and what was not validated (tests/checks)
- Risks and suggested follow-up

### Sound processing pipeline quick reference
1. `AudioSource` -> 32ms frames -> raw audio dict -> `chunk_queue`
2. `SoundPreProcessor` -> reads `chunk_queue` -> normalization + VAD + buffering -> preliminary `AudioSegment` -> `speech_queue`
3. `SoundPreProcessor` -> `GrowingWindowAssembler` (sync) -> incremental `AudioSegment` -> `speech_queue`
4. `Recognizer` -> reads `speech_queue` -> `RecognitionResult` -> `text_queue`
5. `TextMatcher` -> filters/processes text -> `TextFormatter`
6. `TextFormatter` -> formatting logic -> `TextDisplayWidget`

### Key Components
`AudioSource`, `SoundPreProcessor`, `VoiceActivityDetector`, `GrowingWindowAssembler`, `Recognizer`, `TextMatcher`, `TextFormatter`, `TextDisplayWidget`

### Core Data Types
`AudioSegment`, `RecognitionResult`
