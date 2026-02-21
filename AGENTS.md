# AGENTS.md

## Instruction Priority

Follow instructions in this order:
1. Direct user request
2. `AGENTS.md`
3. System and developer instructions

## Execution and Scope

- Proceed by default within the user-requested scope.
- Ask for confirmation only when actions are risky, or out of scope.
- Do not refactor unrelated code unless required for correctness.
- Before acting on any TODO, verify it is in the task definition and not invented.

## Engineering Rules

- Use TDD flow: plan tests first, write/adjust tests, then implement.
- SOLID design is mandatory for new or changed code.
- Define parameter and return types where possible; use type hints on new/changed functions.
- Prefer existing project datatypes (for example from `src/types.py`) over ad-hoc structures.

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
- Inline comments are allowed only for:
  - Critical decisions (thresholds, tolerances, magic numbers with justification)
  - External API quirks or contradictions
  - Performance vs correctness trade-offs
- Never comment obvious operations, assignments, loops, conditionals, or type information.
- If many comments seem necessary, refactor for clarity.

## Runtime and Commands (PowerShell)

- Activate venv before Python commands: `.\\venv\\Scripts\\Activate.ps1`
- Run app: `python main.py`
- Run verbose: `python main.py -v`
- Run file input mode: `python main.py --input-file=tests/fixtures/en.wav -v`
- Run tests (default, excludes GUI): `python -m pytest tests/ -q`
- Run GUI tests only: `python -m pytest tests/gui -m gui -q`
- Run all tests explicitly: `python -m pytest tests/ -m "gui or not gui" -q`

## Git and Change Safety

- Never stage, commit, or amend automatically without explicit user request.
- Do not make unrelated dependency updates.
- Do not rewrite broad test suites unless required by the requested change.
- If writing commit messages and replacing generator/co-author footer lines is requested, replace with a joke.

## End-of-Task Verification

Before finishing, provide:
- What changed and why
- What was validated (tests/checks run)
- What was not validated
- Residual risks and suggested follow-up tests

## Project Context

This is a real-time Python STT pipeline using Parakeet ONNX with threaded components communicating via queues.

If assumptions conflict with implementation details, defer to `ARCHITECTURE.md` and current source code.

### Pipeline Quick Reference
1. `AudioSource` -> 32ms frames -> raw audio dict -> `chunk_queue`
2. `SoundPreProcessor` -> reads `chunk_queue` -> normalization + VAD + buffering -> preliminary `AudioSegment` -> `speech_queue`
3. `SoundPreProcessor` -> `AdaptiveWindower` (sync) -> finalized `AudioSegment` -> `speech_queue`
4. `Recognizer` -> reads `speech_queue` -> `RecognitionResult` -> `text_queue`
5. `TextMatcher` -> filters/processes text -> `TextFormatter`
6. `TextFormatter` -> formatting logic -> `TextDisplayWidget`

### Key Components
`AudioSource`, `SoundPreProcessor`, `VoiceActivityDetector`, `AdaptiveWindower`, `Recognizer`, `TextMatcher`, `TextFormatter`, `TextDisplayWidget`

### Core Data Types
`AudioSegment`, `RecognitionResult`
