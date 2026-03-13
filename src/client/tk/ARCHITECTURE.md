# Tk Client Architecture

The Tk client is an independent subprocess: it captures microphone audio, streams it to the server over WebSocket, receives recognition results as JSON, and renders them in a Tkinter GUI. It performs no ASR inference or VAD — all audio processing is server-side.

See the root [ARCHITECTURE.md](../../../ARCHITECTURE.md) for the server pipeline, network protocol (v1), and the two-process startup model.

---

## Entry Point and Startup Sequence

**File:** `src/client/tk/client.py`

Not intended for direct user invocation — spawned by `main.py` with `--server-url=ws://127.0.0.1:<port>`.

1. Resolves paths at module load time using `ClientPathResolver` (detects dev / portable / MSIX modes).
2. Parses `--server-url=ws://host:port` (required) and optional `--input-file=path`, `-v`.
3. Shows `LoadingWindow` while connecting.
4. Connects WebSocket; reads `session_created` JSON frame; validates `protocol_version == "v1"`.
5. Loads `client_config.json` via `ClientPathResolver`.
6. Calls `ClientApp.from_session_created()` factory.
7. Calls `client_app.set_loop(loop)` and sets `transport._loop` after the asyncio event loop thread starts.
8. Awaits `transport.start(websocket)` on the asyncio loop.
9. Closes `LoadingWindow`, calls `client_app.start()`.
10. Registers a GUI observer for `shutdown` → `root.quit()`; registers `WM_DELETE_WINDOW` → `client_app.stop()`.
11. Runs `root.mainloop()`.
12. On `ConnectionClosed`: `WsClientTransport` transitions `ClientApplicationState` to `shutdown`, which calls `root.quit()` via the GUI observer. Exits with code 1.

---

## Module Organization

| Path | Responsibility |
|---|---|
| `client.py` | Subprocess entry point |
| `ClientApp.py` | Top-level orchestrator; `from_session_created()` factory |
| `ClientApplicationState.py` | 4-state machine; component + GUI observers |
| `ClientPathResolver.py` | Path resolution for dev / portable / MSIX modes |
| `WsClientTransport.py` | Async WebSocket; daemon asyncio thread |
| `RemoteRecognitionPublisher.py` | WS JSON → `RecognitionResultFanOut` dispatch |
| `RecognitionResultFanOut.py` | Thread-safe fan-out publisher |
| `setup_logging.py` | Logging configuration |
| `download_models.py` | CLI model download helper |
| `asr/ModelManager.py` | Client copy of model manager |
| `asr/DownloadProgressReporter.py` | Canonical location of download progress reporter |
| `controllers/PauseController.py` | MVC controller; updates `ClientApplicationState` only |
| `controllers/InsertionController.py` | Insertion mode controller |
| `gui/ApplicationWindow.py` | Main Tk window |
| `gui/ControlPanel.py` | Pause/resume button; observes `ClientApplicationState` |
| `gui/HeaderPanel.py` | Header widget |
| `gui/InsertionModePanel.py` | Insertion mode UI |
| `gui/KeyboardSimulator.py` | Keyboard simulation |
| `gui/LoadingWindow.py` | Splash screen shown during WS connect |
| `gui/ModelDownloadDialog.py` | Model download progress dialog |
| `gui/TextDisplayWidget.py` | Thread-safe text rendering via `root.after()` |
| `gui/TextFormatter.py` | Formatting logic (MVC controller; no GUI knowledge) |
| `gui/TextInserter.py` | Text insertion helper |
| `gui/TextInsertionService.py` | Keyboard insertion subscriber |
| `network/codec.py` | `encode_audio_frame()` — binary `WsAudioFrame` wire format |
| `quickentry/QuickEntryService.py` | Quick-entry popup orchestration |
| `quickentry/QuickEntryController.py` | Quick-entry MVC controller |
| `quickentry/QuickEntryPopup.py` | Quick-entry popup window |
| `quickentry/QuickEntrySubscriber.py` | Recognition result subscriber for quick entry |
| `quickentry/GlobalHotkeyListener.py` | System-wide hotkey detection |
| `quickentry/FocusTracker.py` | Tracks focused window for insertion target |
| `quickentry/windows_effects.py` | Windows-specific visual effects |
| `sound/AudioSource.py` | Microphone capture; 32ms / 16 kHz frames |
| `sound/FileAudioSource.py` | WAV file input; drop-in replacement for testing |

---

## Client-Side Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CLIENT PROCESS (src/client/tk/client.py subprocess)                     │
│                                                                           │
│  AudioSource (sounddevice callback, daemon)                               │
│    → chunk_queue.put_nowait({audio, timestamp, chunk_id})                 │
│                                                                           │
│  AudioBridge daemon thread (ClientApp-AudioBridge)                        │
│    ← chunk_queue.get(timeout=0.05)                                        │
│    → WsClientTransport.send_audio_chunk(chunk)                            │
│      → WsAudioFrame encode → loop.call_soon_threadsafe → asyncio.Queue   │
│                                                                           │
│  WsClientTransport asyncio loop (daemon thread)                          │
│    _drain_loop: asyncio.Queue → websocket.send(bytes) → server            │
│    _receive_loop: websocket text frames → RemoteRecognitionPublisher      │
│                                                                           │
│  RemoteRecognitionPublisher (no thread)                                   │
│    ← JSON WsRecognitionResult frame                                       │
│    → RecognitionResultFanOut.publish_partial_update / publish_finalization│
│                                                                           │
│  RecognitionResultFanOut (no thread)                                      │
│    → TextFormatter.on_partial_update / on_finalization                    │
│    → TextInsertionService                                                 │
│    → QuickEntrySubscriber                                                 │
│                                                                           │
│  TextFormatter (no thread, MVC controller)                                │
│    → DisplayInstructions → TextDisplayWidget.apply_instructions()         │
│                                                                           │
│  TextDisplayWidget (no thread, schedules on main thread via root.after()) │
│  Main thread: Tk mainloop()                                               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Threading Model

| Component | Thread Type | Purpose | Blocking Operations |
|---|---|---|---|
| **Main thread** | Main | Tkinter event loop | `root.mainloop()` |
| **AudioSource** | Daemon (sounddevice callback) | Capture 32ms mic frames; put to chunk_queue | `chunk_queue.put_nowait()` (non-blocking, drops if full) |
| **AudioBridge** (`ClientApp-AudioBridge`) | Daemon | Drain AudioSource.chunk_queue → WsClientTransport | `chunk_queue.get(timeout=0.05)` |
| **WsClientTransport** | Daemon (asyncio event loop) | Send binary audio frames; receive JSON result frames | WS I/O |
| **WsClientTransport._drain_loop** | Async task (transport loop) | asyncio.Queue → `websocket.send(bytes)` | `asyncio.Queue.get()`, `websocket.send()` |
| **WsClientTransport._receive_loop** | Async task (transport loop) | `websocket.__aiter__()` → `RemoteRecognitionPublisher.dispatch()` | WS receive |
| **RemoteRecognitionPublisher** | **No thread** | Decode JSON frames → RecognitionResultFanOut | — |
| **RecognitionResultFanOut** | **No thread** | Fan-out to subscribers; thread-safe copy-on-notify | — |
| **TextFormatter** | **No thread** | Formatting logic (MVC controller + subscriber) | — |
| **TextDisplayWidget** | **No thread** | GUI rendering; schedules on main thread via `root.after()` | — |
| **ClientApplicationState** | **No thread** | Full 4-state machine; GUI + component observers | `threading.Lock` |
| **PauseController** | **No thread** | Updates ClientApplicationState only | — |

---

## Application State

### ClientApplicationState

**File:** `src/client/tk/ClientApplicationState.py`

**Responsibility:** Full client lifecycle including pause/resume.

**State machine:** `starting → running ⟷ paused → shutdown`

**Observer types:**
- **Component observers**: called directly with `(old_state, new_state)`.
- **GUI observers**: scheduled via `root.after(0, callback, old, new)` when called from a background thread, preventing Tk thread violations.

**Observed by:** `AudioSource` (closes stream on pause, reopens on resume), `WsClientTransport` (transitions to shutdown on connection loss), `ControlPanel` (updates button state).

---

## Data Flow

```
AudioSource callback
  → dict {audio: ndarray, timestamp: float, chunk_id: int} → chunk_queue

AudioBridge drain thread
  → WsClientTransport.send_audio_chunk(chunk)
      → WsAudioFrame encode → asyncio.Queue(20) → websocket.send(bytes)

WsClientTransport receive loop
  ← JSON text frame (WsRecognitionResult)
  → RemoteRecognitionPublisher.dispatch(json_text)

RemoteRecognitionPublisher
  → RecognitionResult decode → RecognitionResultFanOut.publish_partial_update / publish_finalization

RecognitionResultFanOut → TextFormatter → DisplayInstructions → TextDisplayWidget → tkinter
```

---

## Component Details

### ClientApp

**File:** `src/client/tk/ClientApp.py`

**Responsibility:** Wires all client-side components; factory via `from_session_created()`.

**Construction order:** `ClientApplicationState` → `RecognitionResultFanOut` → `TextInsertionService` → `PauseController` → `ApplicationWindow` → subscribe `TextFormatter` to fan-out → `RemoteRecognitionPublisher` → `WsClientTransport` → `AudioSource`.

**`start()`:** Transitions state to `running`, starts `AudioBridge` daemon thread, starts `AudioSource`.

**`stop()`:** Idempotent — stops `AudioSource`, stops bridge thread, awaits `transport.stop()` on the asyncio loop, transitions state to `shutdown`.

**AudioBridge (`ClientApp-AudioBridge`):** Daemon thread that drains `AudioSource.chunk_queue` and calls `transport.send_audio_chunk(chunk)`. Fully decouples the sync sounddevice callback from the async transport internals.

---

### WsClientTransport

**File:** `src/client/tk/WsClientTransport.py`

**Responsibility:** Async WebSocket transport on a daemon asyncio event loop thread.

**Outbound:** Sync `send_audio_chunk(chunk)` encodes a `WsAudioFrame` and calls `loop.call_soon_threadsafe(queue.put_nowait, encoded)`. Async `_drain_loop` task drains to `websocket.send(bytes)`. Drops and logs if queue (maxsize=20) is full.

**Inbound:** Async `_receive_loop` iterates websocket text frames and calls `publisher.dispatch(text)`. On `ConnectionClosed`: transitions `ClientApplicationState` to `shutdown`, which stops `AudioSource` and exits the Tk main loop.

---

### RemoteRecognitionPublisher

**File:** `src/client/tk/RemoteRecognitionPublisher.py`

**Responsibility:** Decodes incoming JSON text frames from the server and dispatches to the local publisher.

**Behavior:** `dispatch(json_text)` parses the frame type; routes `recognition_result` frames (partial + final) to `RecognitionResultFanOut`. Unknown types are logged and skipped without raising.

---

### RecognitionResultFanOut

**File:** `src/client/tk/RecognitionResultFanOut.py`

**Responsibility:** Concrete client-side implementation of the `RecognitionResultPublisher` protocol (defined in `src/RecognitionResultPublisher.py`). Manages subscriber list and fan-out.

**Thread safety:** Copy-on-notify pattern (lock-protected list, callbacks outside lock). Subscriber exceptions are logged and isolated — one failing subscriber does not affect others.

**Subscribers:** `TextFormatter` (GUI display), `TextInsertionService` (keyboard insertion), `QuickEntrySubscriber` (quick-entry popup).

---

### AudioSource

**File:** `src/client/tk/sound/AudioSource.py`

**Responsibility:** Capture 32ms audio frames (16 kHz, 512 samples) from the microphone. **Client-side only.**

**Output:** Raw audio dicts `{audio, timestamp, chunk_id}` → `chunk_queue`.

**Observer behavior:** On pause: close stream (release microphone). On resume: create fresh stream.

---

### FileAudioSource

**File:** `src/client/tk/sound/FileAudioSource.py`

**Responsibility:** Drop-in replacement for `AudioSource` that reads from a WAV file at the same frame rate (32ms / 16 kHz). Used with `--input-file=path` for reproducible testing. Implements the same `chunk_queue` interface as `AudioSource`.

---

### PauseController / ControlPanel

**Files:**
- `src/client/tk/controllers/PauseController.py` — MVC controller; only updates `ClientApplicationState`
- `src/client/tk/gui/ControlPanel.py` — GUI widget; observes `ClientApplicationState`; delegates to `PauseController.toggle()`

**PauseController methods:** `pause()`, `resume()`, `toggle()` with guarded state transitions.

**ControlPanel behavior:** Updates button text/state: `"Loading..." → "Pause" → "Resume"`. Delegates clicks to `PauseController`.

**Pause is client-only:** The server never learns about pause. `AudioSource` stops sending audio; `SoundPreProcessor` goes idle naturally.

---

### TextFormatter

**File:** `src/client/tk/gui/TextFormatter.py`

**Responsibility:** Text formatting logic (MVC controller + subscriber, NO GUI knowledge). **Client-side only.** Implements `TextRecognitionSubscriber` protocol; subscribed to `RecognitionResultFanOut`.

**Logic:** Calculates `DisplayInstructions` from `RecognitionResult`; space insertion, paragraph breaks (2.0s threshold), chunk-ID filtering, duplicate detection.

**Output:** Calls `display.apply_instructions()` to trigger GUI updates.

---

### TextDisplayWidget

**File:** `src/client/tk/gui/TextDisplayWidget.py`

**Responsibility:** Render `DisplayInstructions` to tkinter widget with thread-safety. **Client-side only.**

**Thread-Safety:** Detects calling thread; schedules on main thread via `root.after()` if not already on it.

**Styles:** Final text (black, normal), partial text (gray, italic).

---

### QuickEntryService

**File:** `src/client/tk/quickentry/QuickEntryService.py`

**Responsibility:** Orchestrates the quick-entry popup feature. Wires together `GlobalHotkeyListener`, `FocusTracker`, `QuickEntrySubscriber`, `QuickEntryController`, and `KeyboardSimulator`. Subscribes to `RecognitionResultFanOut` to receive recognition results.

**Behavior:** On the configured hotkey (default `ctrl+space`): captures the currently focused window, opens `QuickEntryPopup`, streams recognition results into the popup, and inserts the final text into the previously focused window via `KeyboardSimulator`. Can be disabled via config.

---

### ClientPathResolver

**File:** `src/client/tk/ClientPathResolver.py`

**Responsibility:** Self-contained path resolution for the client subprocess; detects three distribution modes.

**Modes:**
- **development** — relative paths from repo root; models in `<repo>/models`, config in `<repo>/config`.
- **portable** — ZIP distribution with `_internal/app` structure; config beside executable.
- **MSIX** — Windows package; config and models in `%AppData%` (local app data); bundled configs are copied on first run.

---

## Design Patterns

### Observer Pattern — Application State

`ClientApplicationState` supports two observer types:
- **Component observers** (e.g., `AudioSource`): called directly in the mutating thread.
- **GUI observers** (e.g., `ControlPanel`): scheduled via `root.after(0, callback, old, new)` when called from a background thread, preventing Tk thread violations.

Lock-protected mutations; notifications outside lock to prevent deadlocks.

### MVC — Pause/Resume

**Model**: `ClientApplicationState` | **View**: `ControlPanel` | **Controller**: `PauseController` (state-only, no component manipulation)

Flow: `User → ControlPanel → PauseController → ClientApplicationState → Observers → AudioSource`

### MVC — Text Display

**Model**: `RecognitionResult` | **Controller**: `TextFormatter` (no GUI knowledge) | **View**: `TextDisplayWidget` (thread-safe rendering)

Flow: `WsClientTransport → RemoteRecognitionPublisher → RecognitionResultFanOut → TextFormatter → TextDisplayWidget → tkinter`

### Fan-Out Publisher

`RecognitionResultFanOut` implements the `RecognitionResultPublisher` protocol with copy-on-notify semantics: grabs a list snapshot under lock, releases lock before calling subscribers. Subscriber exceptions are logged and isolated — one failing subscriber does not affect others. Subscribers can be added or removed without changing any caller code.

### Async-Sync Bridge

`WsClientTransport` uses the same pattern for both send directions:
- **Outbound**: sync `send_audio_chunk()` → `loop.call_soon_threadsafe(queue.put_nowait, encoded)` → async `_drain_loop` → `websocket.send(bytes)`.
- Bounded queue (maxsize=20) provides natural backpressure with drop-and-log semantics.

### AudioBridge Decoupling

The `ClientApp-AudioBridge` daemon thread sits between the sync sounddevice callback (`AudioSource`) and the async transport (`WsClientTransport`). This decoupling prevents callback starvation: the sounddevice callback is never blocked waiting for WS I/O.
