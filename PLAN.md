# Tauri STT Client — Full Functionality Implementation Plan

## Context

The Tauri client at `src/client/tauri/` is currently a UI-only shell — a polished React layout (HeaderBar, ButtonPanel, TranscriptPanel) with no networking, audio capture, or protocol code. The goal is to turn it into a fully functional STT client on par with the Tk and WinUI clients: WebSocket transport, audio capture, recognition display, **text insertion**, QuickEntry popup, session lifecycle, and error handling.

This plan supersedes `snug-munching-minsky.md`, fixing 11 issues from review.

---

## Architecture: Rust-Heavy

All I/O lives in Rust. React is a pure view layer.

- **Rust → JS**: Tauri events (`app.emit(...)`)
- **JS → Rust**: Tauri commands (`invoke(...)`)

### Rust Module Layout (`src-tauri/src/`)

```
main.rs           — builder, CLI, RunEvent::CloseRequested handler, auto-connect on startup
cli.rs            — --server-url (required), --input-file (optional)
error.rs          — AppError enum (thiserror)
state.rs          — AppState enum + AppStateManager (Result, not panic)
protocol.rs       — Wire types, AudioFrameEncoder, ServerMessageDecoder
transport.rs      — WsClientTransport (mpsc channel, drop-new backpressure)
audio.rs          — AudioSource trait, CpalAudioSource, FileAudioSource
recognition.rs    — RecognitionResult, RecognitionSubscriber trait, FanOut
formatting.rs     — TextFormatter (subscriber, paragraph breaks, dedup)
keyboard.rs       — KeyboardSimulator trait + Win32 SendInput impl
insertion.rs      — TextInserter (subscriber) + InsertionController (toggle)
focus.rs          — FocusTracker (SetForegroundWindow + AttachThreadInput fallback)
hotkey.rs         — GlobalHotkeyListener (Ctrl+Space + popup Enter/Escape)
quickentry.rs     — QuickEntryController + QuickEntrySubscriber
orchestrator.rs   — ClientOrchestrator (wires everything, exposes connect())
commands.rs       — All #[tauri::command] functions
```

---

## Waves & Phases

```
Wave 1 (parallel):  Phase 0 (State+CLI+Error)  |  Phase 3 (TextFormatter — pure Rust, no I/O deps)
                            ↓
Wave 2 (parallel):  Phase 1 (Transport+Protocol)  |  Phase 2 (Audio Capture) 
                            ↓
Wave 3 (parallel):  Phase 4 (Insertion+Keyboard+Lifecycle)  |  Phase 5 (Orchestrator+Errors)
                            ↓
Wave 4 (sequential): Phase 6 (QuickEntry: dedicated Tauri window, hotkey, focus)
                            ↓
Wave 5 (sequential): Integration wiring + smoke test
```

---

## Phase 0: State Machine + CLI + Logging

**Create:**
- `state.rs` — `AppState` enum (`Starting`, `Running`, `Paused`, `Shutdown`), `AppStateManager` with valid-transition table, `Mutex` guard, observer pattern
- `cli.rs` — Parse `--server-url=ws://host:port` (required) + `--input-file=path.wav` (optional)
- `error.rs` — `AppError` enum: `InvalidTransition`, `ConnectionFailed`, `ProtocolError`, `AudioError`

**Modify:**
- `main.rs` — parse CLI, init `tracing` (file + stdout), wire state into Tauri managed state
- `Cargo.toml` — add `clap`, `tracing`, `tracing-subscriber`, `tracing-appender`, `thiserror`

**Design decisions:**
- `set_state()` returns `Result<(), AppError>` — **never panics** (fix #6)
- `Shutdown → Shutdown` is a silent no-op returning `Ok(())`
- Observers called outside lock (copy-on-notify)

**Tests:** valid transitions succeed, invalid return `Err`, shutdown→shutdown no-op, observer notification, CLI parsing with/without `--input-file`

---

## Phase 1: WebSocket Transport + Protocol

**Create:**
- `protocol.rs` — `ServerMessage` tagged enum (serde), `AudioFrameEncoder` (4-byte LE header_len + JSON + f32 PCM), `ServerMessageDecoder`
- `transport.rs` — `WsClientTransport`: `tokio::sync::mpsc::channel(20)`, drain loop (mpsc→WS), receive loop (WS→callback)

**Modify:** `Cargo.toml` — add `tokio` (full), `tokio-tungstenite`, `futures-util`

**Design decisions:**
- Backpressure: `try_send()` — on `TrySendError::Full`, **drop the new frame** and log warning (fix #4, matches Tk/WinUI behavior)
- `stop(server_initiated)` sends `control_command:shutdown` if client-initiated

**Tests:** binary encode/decode roundtrip, all message type decoding, unknown type → Err, queue full → drop new frame, connect timeout

---

## Phase 2: Audio Capture

**Create:**
- `audio.rs` — `AudioSource` trait (`start`/`stop`), `CpalAudioSource` (default input, 16kHz mono f32, 512-sample chunks ≈ 32ms), `FileAudioSource` (reads WAV via `hound`, streams at real-time pace via `tokio::time::interval`)

**Modify:** `Cargo.toml` — add `cpal`, `hound`

**Design decisions (fix #7, #8):**
- `FileAudioSource` implements same `AudioSource` trait — CLI `--input-file` swaps source at startup
- After file ends, simply stops — no sentinel empty chunk (fix #8: matches Tk `FileAudioSource` which just sets `is_running = False` at EOF; the server treats every binary frame as audio with no special EOF semantics)
- Pads last chunk to 512 samples if shorter (matching Tk behavior)

**Tests:** file source produces 512-sample chunks, stops after exhaustion (no trailing empty chunk), last short chunk is zero-padded, cpal format check (`#[ignore]` — needs hardware)

---

## Phase 3: Text Formatting

**Create (Rust):**
- `formatting.rs` — `TextFormatter` implementing `RecognitionSubscriber`: accumulates finals (space-separated), 2.0s paragraph gap, duplicate suppression, partial tracking
- `recognition.rs` — `RecognitionResult` struct, `RecognitionSubscriber` trait, `RecognitionFanOut`

**Modify (TypeScript):**
- `types/viewState.ts` — extend `HeaderStatus` to `"connecting" | "listening" | "paused" | "error"`, add `insertionEnabled`, `connectionError?`
- `App.tsx` — subscribe to Tauri events (`stt://transcript-update`, `stt://state-changed`, `stt://connection-status`, `stt://insertion-state-changed`)
- `TranscriptPanel.tsx` — render preliminary segments in gray italic
- `styles.css` — `.transcript-panel__preliminary`

**Tests (Rust):** partial → rerender, finalization appends, duplicate suppressed, paragraph break on 2s gap, clear resets. FanOut dispatches to all subscribers, subscriber exception doesn't break others.
**Tests (Vitest):** ButtonPanel insert button, TranscriptPanel preliminary rendering

---

## Phase 4: Text Insertion + Keyboard + Lifecycle

**Create:**
- `keyboard.rs` — `KeyboardSimulator` trait + `Win32KeyboardSimulator` (`SendInput`, `KEYBDINPUT` Unicode). Trait enables test fakes. (fix #1)
- `insertion.rs` — `TextInserter` (implements `RecognitionSubscriber`, only acts on finalization when enabled), `InsertionController` (toggle, emits `stt://insertion-state-changed`) (fix #1)
- `commands.rs` — all `#[tauri::command]` functions: `pause`, `resume`, `clear`, `toggle_insertion`, `get_state`

**Modify:**
- `main.rs` — register commands, handle `RunEvent::CloseRequested` for graceful shutdown (fix #5 — no `core:window:allow-on-close-requested` permission needed)
- `ButtonPanel.tsx` — add Insert toggle button, wire to `toggle_insertion` command
- `capabilities/default.json` — add `core:event:allow-listen`, `core:event:allow-emit` (do NOT add invalid `allow-on-close-requested`)
- `Cargo.toml` — add `windows` crate with `Win32_UI_Input_KeyboardAndMouse`, `Win32_UI_WindowsAndMessaging`, `Win32_System_Threading`, `Win32_Foundation`

**Design decisions:**
- `RecognitionFanOut` fans out to both `TextFormatter` (display) and `TextInserter` (keyboard injection)
- `TextInserter.on_partial()` is a no-op; `on_finalization()` checks `enabled`, calls `keyboard.type_text()`
- `toggle_insertion` command → `InsertionController.toggle()` → flips state, propagates to inserter, emits event, returns new state

**Tests:** inserter ignores partial, types finalization when enabled, skips when disabled, controller toggle flips and propagates, fake keyboard records text

---

## Phase 5: Orchestrator + Error Handling

**Create:**
- `orchestrator.rs` — `ClientOrchestrator`: `connect()` opens WS with 10s timeout, validates `session_created` + `protocol_version == "v1"`, stores session_id, starts transport + audio loops. Wires `RecognitionFanOut` with `TextFormatter` + `TextInserter` subscribers.

**Modify:**
- `main.rs` — **call `orchestrator.connect()` automatically on Tauri `setup()` hook** (fix #9: the orchestrator has a `connect()` method but needs an explicit caller). The `connect()` call runs as a spawned `tokio::spawn` task so it doesn't block window creation. The `connect` command is NOT exposed to JS — connection is automatic on startup, matching the Tk client's `ClientApp.from_session_created()` pattern.
- `App.tsx` — handle `stt://connection-status` event, show error in header
- `HeaderBar.tsx` — status pill color variants (green=listening, yellow=paused, red=error, gray=connecting)

**Design decisions (fix #7):**
- Connection failure → emit `stt://connection-status { connected: false, error: "..." }`, display in UI (no crash)
- Protocol mismatch → close WS with code 1002, emit error
- `--input-file` → `FileAudioSource` wired instead of `CpalAudioSource`
- On `ping` → send pong. On `session_closed` → Shutdown. On `error` → emit server error event.

**Tests:** connect success → Running, timeout → error event, protocol mismatch → close 1002, pause/resume transitions, stop sends shutdown, file source wired when `--input-file` set

---

## Phase 6: QuickEntry — Dedicated Floating Window

QuickEntry requires a **separate always-on-top Tauri window** (not an overlay in the main window). When another app is foreground and the user presses Ctrl+Space, the popup must appear on top of everything. A React overlay inside the main window cannot do this. This matches the WinUI implementation which uses a dedicated `QuickEntryWindow` (see [QuickEntryWindow.xaml.cs](src/client/winui/SttClient/Views/QuickEntryWindow.xaml.cs)).

**Create (Rust):**
- `focus.rs` — `FocusTracker` with full restore algorithm (fix #3):
  1. `save()` → `GetForegroundWindow()`, store handle
  2. `restore()` → try `SetForegroundWindow`; on fail: `GetWindowThreadProcessId` + `GetCurrentThreadId` + `AttachThreadInput(attach)` → retry → `AttachThreadInput(detach)`
  3. All Win32 calls via injectable trait for testability
  4. Best-effort, log warnings, **never panic**

- `hotkey.rs` — `GlobalHotkeyListener` (fix #2):
  - Dedicated thread with Win32 `GetMessage` loop
  - Three hotkey IDs: Ctrl+Space (9001), Enter (9002), Escape (9003)
  - `register_popup_hotkeys(on_submit, on_cancel)` — posts `WM_APP+1` to msg loop → registers Enter+Escape globally
  - `unregister_popup_hotkeys()` — posts `WM_APP+2` → unregisters Enter+Escape
  - Uses `PostThreadMessage` to bridge caller thread → message loop thread

- `quickentry.rs` — `QuickEntrySubscriber` (accumulates finalized text), `QuickEntryController`:
  - `show_popup()`: `focus.save()` → activate subscriber → create/show popup window via `WebviewWindowBuilder` → `register_popup_hotkeys(submit, cancel)`
  - `submit()`: get text → hide popup → `unregister_popup_hotkeys()` → `keyboard.type_text(text)` → `focus.restore()`
  - `cancel()`: hide popup → `unregister_popup_hotkeys()` → `focus.restore()`

**Tauri window configuration (created dynamically via `WebviewWindowBuilder`):**
- Label: `"quickentry"`
- URL: `/quickentry.html` (separate React entry point)
- Always-on-top: `true`
- Decorations: `false` (borderless, matching WinUI)
- Size: 720×120 initial, grows with content up to 60% screen height (matching WinUI `QuickEntryWindow`)
- Transparent: `true` (for acrylic-like effect via CSS)
- Show without activating: use `SetWindowPos(HWND_TOPMOST, SWP_NOACTIVATE)` after creation (matching WinUI pattern)
- On close requested: cancel close, invoke cancel callback instead (matching WinUI `OnClosing`)

**Modify:**
- `commands.rs` — add `quickentry_submit`, `quickentry_cancel`
- `capabilities/default.json` — add `core:window:allow-create`, `core:window:allow-set-always-on-top`, `core:window:allow-set-focus`, `core:window:allow-hide`, `core:window:allow-show` (main window manages popup lifecycle)

**Create:**
- `capabilities/quickentry.json` — capability scoped to `"windows": ["quickentry"]` with `core:event:allow-listen` (popup webview only listens for transcript/visibility events; submit/cancel driven by Rust-side hotkeys, not JS invoke)

**Create (TypeScript):**
- `quickentry/QuickEntryApp.tsx` — separate React root for the popup window, listens to `stt://quickentry-text`, renders live text with keyboard hints ("Enter to send · Esc to exit")
- `quickentry/quickentry.html` — HTML entry point for the popup webview (loads `QuickEntryApp.tsx`)
- `quickentry/quickentry.css` — dark semi-transparent background, white text, rounded corners

**Tests (focus.rs):** save captures hwnd, restore calls SetForegroundWindow, fallback uses AttachThreadInput, detaches after, no-op when no saved hwnd
**Tests (hotkey.rs):** callback fires, popup hotkeys register/unregister Enter+Escape
**Tests (quickentry.rs):** toggle visibility, submit types+restores, cancel restores without typing, popup hotkeys registered on show, unregistered on submit/cancel, close-requested invokes cancel (not destroy)

---


## Event & Command Contracts

### Events (Rust → JS)

| Event | Payload |
|-------|---------|
| `stt://state-changed` | `{ old, new }` |
| `stt://transcript-update` | `{ action, finalizedText, preliminarySegments }` |
| `stt://connection-status` | `{ connected, error? }` |
| `stt://insertion-state-changed` | `{ enabled }` |
| `stt://quickentry-text` | `{ text }` |
| `stt://quickentry-visibility` | `{ visible }` |

### Commands (JS → Rust)

| Command | Returns | Notes |
|---------|---------|-------|
| `get_state()` | `String` | |
| `pause()` | `Result<(), String>` | |
| `resume()` | `Result<(), String>` | |
| `clear()` | `Result<(), String>` | |
| `toggle_insertion()` | `Result<bool, String>` | returns new enabled state |
| `quickentry_submit()` | `Result<(), String>` | |
| `quickentry_cancel()` | `Result<(), String>` | |

**Note:** There is no `connect()` command. Connection is automatic — `main.rs` calls `orchestrator.connect()` in the Tauri `setup()` hook via `tokio::spawn` (fix #9).

---

## Capabilities (Window-Scoped)

Tauri 2 capabilities are scoped to window labels — a webview not matching any capability has **zero IPC access**. Two capability files are needed: one for the main window, one for the dynamically-created QuickEntry popup.

### `capabilities/default.json` — Main Window

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "Main window capabilities for the STT client.",
  "windows": ["main"],
  "permissions": [
    "core:window:allow-close",
    "core:window:allow-minimize",
    "core:window:allow-maximize",
    "core:window:allow-unmaximize",
    "core:window:allow-is-maximized",
    "core:window:allow-start-dragging",
    "core:window:allow-create",
    "core:window:allow-set-always-on-top",
    "core:window:allow-set-focus",
    "core:window:allow-hide",
    "core:window:allow-show",
    "core:event:allow-listen",
    "core:event:allow-emit"
  ]
}
```

### `capabilities/quickentry.json` — QuickEntry Popup Window

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "quickentry",
  "description": "QuickEntry popup window — display-only surface for live transcript.",
  "windows": ["quickentry"],
  "permissions": [
    "core:event:allow-listen"
  ]
}
```

The quickentry webview is a pure display surface: it receives `stt://quickentry-text` and `stt://quickentry-visibility` events from Rust but never invokes commands. Submit/cancel are handled by global hotkeys on the Rust side, so the popup only needs `core:event:allow-listen`.

---

## Cargo Dependencies (Final)

`clap`, `tokio` (full), `tokio-tungstenite`, `futures-util`, `cpal`, `hound`, `reqwest` (stream + native-tls), `tracing`, `tracing-subscriber`, `tracing-appender`, `thiserror`, `windows` (Win32 UI/Input/Threading/Foundation)

---

## Verification

1. `cd src/client/tauri/src-tauri && cargo test` — all Rust tests pass
2. `cd src/client/tauri && npm test` — Vitest passes
3. `npm run tauri:build` — compiles
4. Start server (`python main.py --server-only`), launch Tauri with `--server-url=ws://127.0.0.1:<port>` — auto-connects on startup, transcript appears (fix #9)
5. `--input-file=tests/fixtures/en.wav` — transcript from file, no sentinel empty chunk sent (fix #7, #8)
6. Insert toggle ON → speak → text typed into active window (fix #1)
7. Ctrl+Space → dedicated floating popup appears on top of all windows, speak → Enter → text in previous window (fix #2, #10)
8. QuickEntry submit → previous window regains focus reliably (fix #3)
9. Flood audio under slow server → logs "dropping frame", no OOM (fix #4)
10. Wrong port → error shown in UI, no crash (fix #7)
11. Click X → graceful shutdown with `control_command:shutdown` sent (fix #5)

