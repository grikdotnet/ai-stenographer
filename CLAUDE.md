## Project Overview

This is a real-time Speech-to-Text (STT) system built in Python that captures audio from microphone and converts it to text using the Parakeet ONNX model. The system uses a client-server architecture communicating over web sckets, with threaded components processing audio streams through queues.

## General

State your assumptions explicitly. If uncertain, ask.
If there is a confusion express it clearly. Surface tradeoffs.
Present options - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.

**Define success criteria.**

Compose tasks as verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

**Arrange tasks for sub-agents.**

Decompose the plan in multiple independent tasks.
Define an execution plan for sub-agents.
Define instructions for sub-agents to veryfy implementation agains the success criteria and the plan.

**Simplicity First**

Minimum code that solves the problem. Nothing speculative.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't explicitly requested.
No error handling for impossible scenarios.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

**Surgical Changes**

Touch only what you must. Clean up only your own edits.
Don't "improve" adjacent code, comments, or formatting.
Don't refactor code that wasn't requested.
Before doing or planning anything check whether the todo item is in the task definition (plan), or it is made up. 

After creating git commit mesages replace "Generated with [Claude Code] ..." line or "Co-Authored-By" with a joke.
Always ask user command or confirmation before git commit or push.

In the start of an answer, be creative, instead of generic phrases like "You're absolutely right!", "Good/Great/Excellent question!", "Good catch!" use "OK", "Fine", RPG-style like "As you whish, my master", sarcasm style like "Are you proud of yoursel, smartypants?", write some fun words.

Sarcasm and jokes are welcome everywhere.

It's April 2026 now.

## Test driven development
First plan tests, design and create tests, implement code later. SOLID design is mandatory.

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


## Commands

When running python always activate venv with `source venv/Scripts/activate`.
When running in Windows, `ls -la ` is not supported, use the `find` command.

Activate venv before running any Python commands:

```bash
source venv/Scripts/activate
```
Commands to run server and client applications are listed in the [./commands.md](commands.md) file.


## Architecture

The system uses a client-server design.
Server-side **pipeline architecture** with threaded components communicating via queues.

📖 **See [ARCHITECTURE.md](ARCHITECTURE.md) for the architecture documentation of the server application**.
📖 **For the Tauri client documentation see [./src/client/tauri/CLAUDE.md](Tauri Client CLAUDE.md)**.
📖 **The client-server API specification is in [./docs/client-server-api-spec.md](client-server-api-spec.md)**.

## Architectural Dependency Rules

Server-side code is organised into 5 layers — see the Appendix in [ARCHITECTURE.md](ARCHITECTURE.md). Ordered from foundational (lowest) to composition root (highest):

1. **System / Infrastructure** — OS, filesystem, sockets, ONNX, logging, CLI
2. **Protocol / Wire** — message types, codecs
3. **Model / Domain** — pure logic, DTOs, algorithms, state machines
4. **Service / Orchestration** — lifecycle, queues, threads, pipeline wiring
5. **Entry-Point / Application** — composition root (`main.py`, `ServerApp`)

### Rule 1 — Dependency direction (strict)

A class in a **higher** layer MAY import, construct, and call a class in a **lower** layer.

A class in a **lower** layer MUST NOT import, construct, or call a class in a **higher** layer.

Examples:
- `RecognizerService` (Service) uses `Recognizer` (Domain) and `ModelLoader` (System) — allowed.
- `Recognizer` (Domain) MUST NOT import `RecognizerService` or `SessionManager` (Service).
- `ModelLoader` (System) MUST NOT import any Domain, Service, or Entry-Point class.
- No code outside `main.py` / `ServerApp` MAY import from the Entry-Point layer.

If a lower-layer class needs to notify a higher-layer class, define a protocol/callback in the lower layer and let the higher layer implement it (observer pattern). `RecognitionResultPublisher` and `TextRecognitionSubscriber` already follow this shape.

### Rule 2 — Layer adjacency (prefer; do not skip two layers)

A class SHOULD depend only on the **adjacent lower layer**.

Reaching two layers down (e.g. a Service class directly calling into Infrastructure when a Domain abstraction already exists) SHOULD be avoided. If such a dependency is unavoidable, route it through an abstraction in the intermediate layer.

**Exemption — Entry-Point.** The composition root wires layers together and may reference all layers.

**Exemption — Protocol is a cross-cutting boundary.** Protocol types (`src/network/types.py`) and codecs (`src/network/codec.py`) are DTOs designed to cross boundaries. Importing Protocol is OK.

### When adding a new class

1. Classify it using questions in "Classifying new entities" section of ARCHITECTURE.md
2. Check imports against Rule 1 — a red `from src.server.X` inside a Domain file is a structural error, not a choice.
3. Check its imports against Rule 2 — if you find yourself reaching into Infrastructure from Service, ask whether a Domain abstraction is missing.

### When refactoring

If an existing file violates these rules, report to the user, propose fixing the violation over adding an exception. Document exceptions in the class docstring with the reason.