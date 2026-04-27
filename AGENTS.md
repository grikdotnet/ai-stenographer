## Project Context

This is a real-time Python STT pipeline using Parakeet ONNX with threaded components communicating via queues.

`ARCHITECTURE.md` contains detailed documentation of the system structure.
Client-specific architecture is documented in `client/tauri/ARCHITECTURE.md`.

## Execution and Scope

- Before acting on any TODO, verify it is explicitly in the task definition or approved plan.
- Never stage, commit, or amend git repo automatically without explicit user approval or command.
- Create a plan before making changes. The plan must include intended behavior, likely files/components, tests first, applicable architecture classification for new/changed classes, and explicit non-goals.

## Communication and Uncertainty

- State assumptions explicitly before relying on them.
- If the task is unclear, stop and name what is confusing. Ask before changing code.
- If there is confusion or a contradiction in the repo, docs, tests, or task wording, surface it clearly.
- Present meaningful options and tradeoffs instead of silently choosing between materially different approaches.
- If a simpler approach exists, say so. Push back when the requested path adds unnecessary complexity or risk.

## Planning Rules

- Define success criteria before implementation. Phrase work as verifiable goals:
  - "Add validation" -> "Write tests for invalid inputs, then make them pass."
  - "Fix the bug" -> "Write a test that reproduces it, then make it pass."
  - "Refactor X" -> "Ensure relevant tests pass before and after."
- For multi-step tasks, write each step with its verification check:
  - `[Step] -> verify: [check]`
- Structure plans for implementation by sub-agents with task ownership, dependencies, and implementation order.
- Include verification instructions for an agent to check the implementation against the success criteria and the approved plan.

## Engineering Rules

- Use TDD flow: plan tests first, write/adjust tests, then implement.
- Single Responsibility and Interface Segregation design principles are mandatory for the code design.
- Define parameter and return types where possible; use type hints on new/changed functions.
- Prefer existing project datatypes (for example from `src/types.py`) over ad-hoc structures.
- When a variable is used in a single method, prefer a local variable over a class property.

**Simplicity First**

Minimum code that solves the problem. Nothing speculative.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't explicitly requested.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

**Surgical Changes**

Touch only what you must. Clean up only your own edits.
Don't refactor code that wasn't requested.

### Planning Architecture Checks

- During planning, classify every new or changed class into one of the 5 server-side layers documented in the Appendix of `ARCHITECTURE.md`.
- Reject dependencies from a lower layer to a higher layer. Higher layers may depend on lower layers; lower layers must not depend on higher layers.
- Avoid skipping two layers when introducing dependencies. The intentional exemptions are the Entry-Point as composition root and the Protocol / Wire layer as a cross-cutting boundary.
- If an existing design violates these rules, propose a fix instead of adding another exception.
- Use `ARCHITECTURE.md` as the canonical source for layer definitions, litmus questions, and detailed examples.
- For Tauri client changes, use `client/tauri/ARCHITECTURE.md` as the canonical architecture source instead of the server layer catalogue.

## Testing Policy

- MUST validate behavioral outcomes (correct results) and expected logic.
- MUST cover queue interaction when queue behavior is part of the change.
- MUST validate ordering, backpressure, shutdown/drain behavior, and sentinel handling when touching threaded pipeline, session, websocket, or recognizer-service code.
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

- Run Python commands from the repository root through the checked-in virtual environment:
```powershell
.\venv\Scripts\python.exe -m pytest tests/ -q
```

- Default Python validation: `.\venv\Scripts\python.exe -m pytest tests/ -q`
- Default Tauri frontend validation: from `client/tauri`, run `npm test`
- Default Tauri Rust validation: from `client/tauri/src-tauri`, run `cargo test`

Commands to run server and client applications are listed in the [./commands.md](commands.md) file.

## End-of-Task Verification

Before finishing a task, provide a summary:
- What changed and why
- What was and what was not validated (tests/checks)
- Risks and suggested follow-up
