# CLAUDE.md — Tauri Client

Sub-project of the STT monorepo. A Windows desktop client built with **Tauri 2 + React 19 + TypeScript**.

Connects to the Python STT server (`main.py --server-only`) via WebSocket and provides real-time transcription display, text insertion into the focused window, and a QuickEntry popup.

## Stack

| Layer | Technology |
|-------|-----------|
| Shell | Tauri 2.10.3 (`src-tauri/`) — Rust backend |
| UI | React 19 + TypeScript (strict) |
| Build | Vite 8 |
| Tests | Vitest + Testing Library (frontend), `cargo test` (Rust) |

## Structure

```
src-ts/                           — React frontend (pure view layer)
  App.tsx                         — root component, Tauri event subscriptions + command invocations
  main.tsx                        — React DOM entry
  styles.css                      — design system (Segoe UI, warm palette, frosted glass)
  types/viewState.ts              — AppViewState, HeaderStatus
  components/
    AppShell.tsx                  — layout container
    HeaderBar.tsx                 — title bar, status pill, custom window controls
    ButtonPanel.tsx               — Pause/Resume, Clear, Insert toggle
    TranscriptPanel.tsx           — scrollable finalized + preliminary text
  quickentry/
    QuickEntryApp.tsx             — separate React root for the popup window
    quickentry.css                — dark semi-transparent popup styling

src-tauri/                        — Rust backend (all I/O lives here)
  src/
    main.rs                       — CLI parsing, tracing init, Tauri builder, auto-connect on startup
    lib.rs                        — module declarations
    cli.rs                        — --server-url (required), --input-file (optional)
    error.rs                      — AppError enum (thiserror)
    state.rs                      — AppState (Starting/Running/Paused/Shutdown), AppStateManager
    protocol.rs                   — ServerMessage enum, AudioFrameEncoder, ServerMessageDecoder
    transport.rs                  — WsClientTransport (tokio-tungstenite, mpsc backpressure)
    audio.rs                      — AudioSource trait, CpalAudioSource, FileAudioSource
    recognition.rs                — RecognitionResult, RecognitionSubscriber trait, FanOut
    formatting.rs                 — TextFormatter (paragraph breaks, dedup, observer callback)
    keyboard.rs                   — KeyboardSimulator trait, Win32 SendInput impl
    insertion.rs                  — TextInserter (subscriber) + InsertionController (toggle)
    focus.rs                      — FocusTracker (SetForegroundWindow + AttachThreadInput fallback)
    hotkey.rs                     — GlobalHotkeyListener (Ctrl+Space, Enter, Escape)
    quickentry.rs                 — QuickEntryController + QuickEntrySubscriber
    orchestrator.rs               — ClientOrchestrator (wires everything, auto-connect, event emission)
    commands.rs                   — Tauri IPC commands (get_state, pause, resume, clear, toggle_insertion, quickentry_*)
  tauri.conf.json                 — window config (980x620, custom decorations)
  capabilities/
    default.json                  — main window permissions (window mgmt + events)
    quickentry.json               — popup window (event listen only)
  tests/                          — Rust integration tests (one file per module)
```

## Commands

```bash
npm test                           # Vitest suite (frontend)
npm run test:watch                 # watch mode
npm run dev                        # Vite dev server (http://localhost:1420)
npm run tauri:dev                  # full Tauri dev (hot-reload)
npm run tauri:build                # production build + bundle
cd src-tauri && cargo test         # Rust tests (132 tests)
```

### Running with GUI
```bash
# Launch the Tauri client — pass server URL via STT_SERVER_URL env var
STT_SERVER_URL=ws://127.0.0.1:<port> npm run tauri:dev
```

```bash
# With file input instead of microphone
STT_SERVER_URL=ws://127.0.0.1:<port> STT_INPUT_FILE=tests/fixtures/en.wav npm run tauri:dev
```

The Rust binary reads `STT_SERVER_URL` and `STT_INPUT_FILE` via clap's `env` attribute.
Note: npm `--key=value` config flags do not work reliably for underscore-named keys.

### Running headless rust client

```bash
# Build
cd client/tauri/src-tauri
cargo build

# Run headless — no Node, no Vite, no window
.\target\debug\stt-tauri-client.exe --headless --server-url=ws://127.0.0.1:64314

# With file input
.\target\debug\stt-tauri-client.exe --headless --server-url=ws://127.0.0.1:64314 --input-file=..\..\..\tests\fixtures\en.wav
```

## Architecture

**Rust-heavy**: all I/O (WebSocket, audio capture, keyboard injection, hotkeys) lives in Rust. React is a pure view layer.

- **Rust -> JS**: Tauri events (`app.emit(...)`) — `stt://transcript-update`, `stt://state-changed`, `stt://connection-status`, `stt://insertion-state-changed`, `stt://quickentry-text`, `stt://quickentry-visibility`
- **JS -> Rust**: Tauri commands (`invoke(...)`) — `pause`, `resume`, `clear`, `toggle_insertion`, `quickentry_submit`, `quickentry_cancel`
- **Connection**: automatic on startup via `orchestrator.connect()` in the Tauri `setup()` hook

## Development Guidelines

- **TDD**: write tests before implementation. Tests live next to source (`*.test.tsx`) or in `src-tauri/tests/`.
- **SOLID**: small, single-responsibility modules; traits for testability (KeyboardSimulator, WindowFocusApi, AudioSource, EventEmitter).
- **No inline logic in JSX**: extract handlers as named functions.
- **TypeScript strict**: no `any`, explicit return types on all exported functions.
- **Win32 code**: behind `#[cfg(windows)]` with no-op fallbacks for cross-platform compilation.
- Follow the comment/docstring rules in the root `CLAUDE.md`.

## AppViewState

Central read-only state passed top-down from App.tsx:

```ts
type HeaderStatus = "connecting" | "listening" | "paused" | "error";

interface AppViewState {
  title: string;
  subtitle: string;
  status: HeaderStatus;
  transcriptText: string;
  preliminaryText: string;
  isPaused: boolean;
  insertionEnabled: boolean;
  connectionError?: string;
}
```
