# Plan: WinUI3 Native Windows Client for STT Server

## Context

The STT system has a Python server (`src/server/`) and an existing Python/Tk client (`src/client/tk/`). The goal is to implement a native Windows WinUI3 C# client that speaks the same WebSocket protocol (v1), replacing the Tk client for Windows users. The server is unchanged; this is a new client process.

The Python client (`src/client/tk/`) is the functional and architectural reference. The wire protocol is defined in `docs/client-server-api-spec.md`.

Target: .NET 10 + Windows App SDK 1.7, project at src/client/winui/. (.NET 10 GA'd November 2025, LTS through 2028; WinAppSDK 1.7 released March 2026.)

---

## Project Structure

```
src/client/winui/
├── SttClient.sln
├── SttClient/                          # WinUI3 app project (.NET 10)
│   ├── SttClient.csproj
│   ├── App.xaml / App.xaml.cs          # Parses --server-url CLI arg, launches orchestrator
│   ├── Protocol/
│   │   ├── AudioFrameEncoder.cs        # Binary frame: uint32-LE header_len + JSON + float32 PCM
│   │   ├── ServerMessageDecoder.cs     # JSON text → sealed ServerMessage hierarchy
│   │   └── WireTypes.cs               # Records: WsAudioFrame, WsRecognitionResult, WsSessionCreated, WsSessionClosed, WsError, WsControlCommand
│   ├── State/
│   │   ├── AppState.cs                 # enum: Starting, Running, Paused, Shutdown
│   │   └── AppStateManager.cs          # Thread-safe state machine + observer notifications
│   ├── Audio/
│   │   ├── IAudioSource.cs             # interface: Start/Stop, event ChunkReady(AudioChunk)
│   │   └── WasapiAudioSource.cs        # NAudio WasapiCapture, 16kHz mono float32, 512-sample chunks
│   ├── Transport/
│   │   └── WsClientTransport.cs        # ClientWebSocket + Channel<byte[]>(capacity:20,DropWrite)
│   ├── Recognition/
│   │   ├── IRecognitionSubscriber.cs   # OnPartialUpdate / OnFinalization
│   │   ├── RecognitionResult.cs        # record: text, start_time, end_time, utterance_id, chunk_ids, token_confidences
│   │   ├── RecognitionResultFanOut.cs  # Thread-safe copy-on-notify multi-subscriber fan-out
│   │   └── RemoteRecognitionPublisher.cs  # Decodes JSON → routes to fan-out or AppStateManager
│   ├── Formatting/
│   │   ├── DisplayInstructions.cs      # record: FinalizedText, PartialText (DTO between formatter and view)
│   │   └── TextFormatter.cs            # IRecognitionSubscriber; paragraph break logic (2.0s threshold)
│   ├── Insertion/
│   │   ├── KeyboardSimulator.cs        # SendInput P/Invoke (user32.dll); InputInjector NOT used (requires uiAccess manifest)
│   │   ├── TextInserter.cs             # IRecognitionSubscriber; types text when enabled
│   │   └── InsertionController.cs      # Toggle enable/disable
│   ├── QuickEntry/
│   │   ├── FocusTracker.cs             # GetForegroundWindow/SetForegroundWindow P/Invoke
│   │   ├── GlobalHotkeyListener.cs     # RegisterHotKey + message-only HWND Win32 message loop
│   │   ├── QuickEntrySubscriber.cs     # Accumulates partial+final text for popup
│   │   └── QuickEntryController.cs     # Orchestrates popup show/hide/submit/cancel
│   ├── ViewModels/
│   │   ├── MainWindowViewModel.cs      # INotifyPropertyChanged; Apply(DisplayInstructions) dispatches to UI thread
│   │   └── QuickEntryViewModel.cs
│   ├── Views/
│   │   ├── MainWindow.xaml/.cs         # Title + Pause button + Insertion toggle + text display
│   │   ├── LoadingPage.xaml/.cs        # Splash with two states: Connecting (spinner) and Error (message + Exit button)
│   │   └── QuickEntryWindow.xaml/.cs   # Borderless window with Microsoft.UI.Xaml.Media.DesktopAcrylicBackdrop; Enter=submit, Esc=cancel
│   └── App/
│       └── ClientOrchestrator.cs       # Wires all components; ConnectAsync → session_created → Running → audio
└── SttClient.Tests/
    ├── SttClient.Tests.csproj          # xUnit, no WinUI/COM dependencies
    └── (test files per phase)
```

### NuGet Packages

| Package | Purpose |
|---|---|
| `Microsoft.WindowsAppSDK` 1.7.x | WinUI3, DispatcherQueue, InputInjector |
| `NAudio` 2.2.x | WASAPI audio capture + MediaFoundationResampler |
| `System.Text.Json` (built-in) | JSON serialization |
| `System.Threading.Channels` (built-in) | Bounded async send queue |
| `Microsoft.Extensions.Logging` | Structured logging abstraction |
| `Serilog.Extensions.Logging` + `Serilog.Sinks.File` | File sink for production diagnostics |
| `xunit`, `Moq`, `coverlet.collector` | Tests |

---

## Component Design

### Threading Model

| Component | Thread | Notes |
|---|---|---|
| WinUI3 UI | Main/UI thread | All property changes via `DispatcherQueue.TryEnqueue` |
| `WasapiAudioSource` callback | NAudio WASAPI thread | <1ms; only `Channel.Writer.TryWrite` |
| `WsClientTransport.DrainLoop` | ThreadPool Task | Channel → `WebSocket.SendAsync` |
| `WsClientTransport.ReceiveLoop` | ThreadPool Task | Reads text frames → `RemoteRecognitionPublisher.Dispatch` |
| `TextFormatter`, `RecognitionResultFanOut` | ReceiveLoop Task | Synchronous; enqueue to DispatcherQueue at view boundary |
| `GlobalHotkeyListener` | Dedicated Thread | Win32 message loop on message-only HWND |
| `AppStateManager` | Any thread | `lock` guards mutation; observers notified outside lock |

### AppState Transition Table

| From | → To | Valid |
|---|---|---|
| `Starting` | `Running`, `Shutdown` | ✓ |
| `Running` | `Paused`, `Shutdown` | ✓ |
| `Paused` | `Running`, `Shutdown` | ✓ |
| `Shutdown` | (any) | ✗ terminal |

Invalid transition → `InvalidOperationException`. `Shutdown` is idempotent: setting `Shutdown` from `Shutdown` is a no-op (not an error).

### Key Design Decisions

**`Channel<byte[]>` for send queue**: `BoundedChannelOptions(capacity: 20, FullMode: DropWrite)`. `TryWrite` is wait-free — safe from WASAPI callback thread. On overflow, `TryWrite` returns false and the **current (newest) frame is discarded** — already-queued frames are preserved. This matches the Python `put_nowait` / `QueueFull` drop behavior. Log the drop at `Debug` level.

**NAudio `WasapiCapture` with resampling fallback**: Target `WaveFormat.CreateIeeeFloatWaveFormat(16000, 1)`. If WASAPI shared-mode device rejects this format, fall back to device-native format and wrap with `MediaFoundationResampler` to produce 16 kHz float32. Test with fixed fake provider; add a smoke test note for real hardware.

**`RegisterHotKey` over `SetWindowsHookEx`**: Correct API for global hotkeys; fires only the registered combination; no system-wide performance impact. Message-only HWND (`HWND_MESSAGE`) is invisible and standard.

**`SendInput` P/Invoke as sole keyboard injection mechanism**: `Windows.UI.Input.Preview.Injection.InputInjector` requires `uiAccess="true"` in the app manifest AND installation in a trusted location (Program Files), making it unsuitable for sideloaded dev builds where it silently fails. Use `SendInput` (user32.dll) exclusively — no manifest changes required.

**`IDispatcherQueueAdapter` interface**: Injected into `MainWindowViewModel` for testability (synchronous fake in tests). `TryEnqueue` returns `bool`; the real adapter no-ops silently if the queue is unavailable during shutdown (returns false, no exception thrown).

**`DisplayInstructions` DTO**: Decouples `TextFormatter` (logic) from `MainWindowViewModel` (UI). Faithfully ports the Python pattern with three fields: `Action` (`RerenderAll` | `Finalize`), `FinalizedText` (full buffer), `PreliminarySegments` (list of partial text segments). `RerenderAll` triggers full text replacement (used after paragraph breaks); `Finalize` is incremental append. Both enum variants are always present; the view decides rendering based on `Action`.

**Sealed `ServerMessage` hierarchy**: `SessionCreated`, `RecognitionResultMessage`, `SessionClosed`, `ErrorMessage`, `PingMessage`. `ServerMessageDecoder` peeks at `"type"` JSON field before deserializing. `RemoteRecognitionPublisher` responds to `ping` by sending a `pong` JSON text frame: `{"type":"pong","session_id":"...","timestamp":...}`.

**session_id validation in `RemoteRecognitionPublisher`**: Stores the `session_id` received in `session_created`. On every incoming `recognition_result`, discards and logs `[Warning]` if `session_id` mismatches.

**Canonical JSON key order in audio frames**: `AudioFrameEncoder` must produce JSON header keys in fixed order: `type`, `session_id`, `chunk_id`, `timestamp`. Use `Utf8JsonWriter` with explicit property writes (not `JsonSerializer` which may reorder). Tests compare against hardcoded golden byte sequences, not against re-running Python.

**`ArrayPool<byte>` lifecycle in `AudioFrameEncoder`**: `Rent()` returns arrays larger than requested. `Encode` returns a `PooledFrame : IDisposable` wrapper containing a rented `byte[]` and an exact-length `Memory<byte>` slice. The caller (`DrainLoop`) uses the `Memory<byte>` for `WebSocket.SendAsync`, then disposes `PooledFrame` to return the array to the pool. This prevents GC pressure at 31 frames/sec.

**Shutdown initiator tracking**: `WsClientTransport` tracks whether shutdown was client- or server-initiated. `StopAsync(serverInitiated: bool)` — skips sending `control_command shutdown` if server already sent `session_closed` (server initiated). Prevents sending to a half-closed socket.

**QuickEntry focus restore**: `FocusTracker.RestoreFocus()` first calls `SetForegroundWindow`; if it returns false (common due to Windows focus rules), falls back to `AttachThreadInput` + `SetForegroundWindow`. Best-effort: logs failure, never throws.

**QuickEntry backdrop**: Use `Microsoft.UI.Xaml.Media.DesktopAcrylicBackdrop` on both Windows 10 (19041+) and Windows 11 — it works on both. `MicaBackdrop` is Windows 11 only and provides no benefit worth the conditional logic.

**Logging**: All components accept `ILogger<T>` via constructor injection. Log levels: `Debug` for per-frame events (dropped frames, dispatch), `Information` for state transitions and session lifecycle, `Warning` for recoverable errors (malformed JSON, session_id mismatch), `Error` for fatal conditions (protocol_version mismatch, unexpected WebSocket close). Output: `Microsoft.Extensions.Logging.Debug` provider during development (visible in VS Output window); Serilog file sink (`logs/sttclient-.log`) for production diagnostics.

**Non-goals**: No reconnection or retry logic. If the connection drops, the app transitions to `Shutdown`. Server URL must use `ws://` scheme; `wss://` and other schemes show an error: "TLS (wss://) is not yet supported in protocol v1."

---

## Wire Protocol Implementation

### Audio Frame Encoding (`AudioFrameEncoder.Encode`)

Reference: `src/client/network/codec.py::encode_audio_frame`

```
[0..3]        uint32 LE (header_len)
[4..4+hlen)   UTF-8 JSON: {"type":"audio_chunk","session_id":"uuid","chunk_id":42,"timestamp":1735...}
[4+hlen..]    float32 LE PCM bytes (MemoryMarshal.Cast<float,byte>)
```

Use `ArrayPool<byte>` to avoid per-frame GC allocations.

### Session Startup

1. `App.xaml.cs` reads `--server-url=ws://host:port` from `Environment.GetCommandLineArgs()`
2. Validate URL scheme is `ws://`; on `wss://` or other scheme show error "TLS (wss://) is not yet supported in protocol v1" and exit
3. Show `LoadingPage`
4. `ClientOrchestrator.ConnectAsync()` — all steps wrapped in `CancellationTokenSource(TimeSpan.FromSeconds(10))`:
   - `ClientWebSocket.ConnectAsync(uri, cts.Token)` — times out at 10s; show "Connection timed out" on `OperationCanceledException`
   - `WebSocket.ReceiveAsync(buffer, cts.Token)` — times out if server doesn't send `session_created` within remaining budget
   - `ServerMessageDecoder.Decode` → assert `SessionCreated`
   - If `protocol_version != "v1"`: log `[Error]`, close WebSocket with code 1002 (protocol error), show error in `LoadingPage` error state, exit
5. Start `DrainLoop` and `ReceiveLoop` Tasks
6. `AppStateManager.SetState(Running)` → `WasapiAudioSource.Start()`
7. Navigate to `MainWindow`

### Shutdown

Two paths depending on initiator:

- **Server-initiated** (`session_closed` received): `WsClientTransport.StopAsync(serverInitiated: true)` → skip `control_command`, just close WebSocket → `AppStateManager.SetState(Shutdown)` → `WasapiAudioSource.Stop()` → `Application.Current.Exit()`
- **Client-initiated** (window close or error): `WsClientTransport.StopAsync(serverInitiated: false)` → send `control_command shutdown` → await `session_closed` or timeout(2s) → close WebSocket → `AppStateManager.SetState(Shutdown)` → `WasapiAudioSource.Stop()` → `Application.Current.Exit()`

Shutdown is idempotent — `AppStateManager` ignores repeated `SetState(Shutdown)` calls.

---

## Phase Dependency Graph

```
Phase 1 (Protocol) ──┐
Phase 2 (State) ─────┼─→ Phase 5 (Transport) ─┐
Phase 3 (Fan-out) ───┘                          ├─→ Phase 9 (Orchestrator)
Phase 4 (Formatter) ──→ Phase 7 (ViewModel) ───┤
Phase 6 (Audio) ──────────────────────────────┤
Phase 8 (QuickEntry) ─────────────────────────┘
```

Phases 1–4 are pure logic with no platform or UI dependencies — they can be developed in parallel. Phase 5 requires 1+2+3. Phase 6 requires 2. Phase 7 requires 2+4. Phase 9 requires all.

---

## Implementation Phases (TDD — tests before implementation)

### Phase 1 — Protocol Layer
**Deliverable**: `AudioFrameEncoder`, `ServerMessageDecoder`, all wire type records.

Tests:
- `AudioFrameEncoderTests`: encode produces correct 4-byte LE prefix, then JSON keys in fixed order (`type`, `session_id`, `chunk_id`, `timestamp`), then float32 bytes — compare against hardcoded golden byte array (derived once from Python reference output); verify key order does not vary across runs
- `ServerMessageDecoderTests`: decode all 5 server message types from known JSON strings; unknown `type` → null; malformed JSON → null (no exception); incoming `recognition_result` with mismatched `session_id` → log Warning, return null

### Phase 2 — State Machine
**Deliverable**: `AppState`, `AppStateManager`.

Tests:
- All valid transitions succeed; all invalid transitions throw `InvalidOperationException`
- Concurrent `SetState` from multiple threads is safe
- Observer called exactly once per transition, outside the lock

### Phase 3 — Recognition Fan-Out and Publisher
**Deliverable**: `IRecognitionSubscriber`, `RecognitionResult`, `RecognitionResultFanOut`, `RemoteRecognitionPublisher`.

Tests:
- Fan-out calls all subscribers; subscriber exception does not affect others
- Thread-safe concurrent subscribe + publish
- `RemoteRecognitionPublisher.Dispatch`: partial → `OnPartialUpdate`; final → `OnFinalization`; `session_closed` → `AppStateManager.SetState(Shutdown)`; unknown type ignored; malformed JSON swallowed

### Phase 4 — Text Formatter
**Deliverable**: `DisplayInstructions`, `TextFormatter`.

Tests (injected delegate captures `DisplayInstructions`):
- Partial update → `PartialText` set, `FinalizedText` unchanged
- Finalization → `FinalizedText` appended, `PartialText` cleared
- Duplicate finalization suppressed
- Gap > 2.0s → paragraph break prepended
- Gap < 2.0s → no paragraph break

### Phase 5 — WebSocket Transport
**Deliverable**: `WsClientTransport`.

Tests (fake `IWebSocket` interface):
- `SendAudioChunk` from background thread enqueues encoded bytes
- Send queue drops (logs) on overflow (capacity 20); does not block caller
- `ReceiveLoop` dispatches text frames to publisher
- `ReceiveLoop` triggers `Shutdown` on WebSocket close
- `StopAsync` cancels both tasks without deadlock

### Phase 6 — Audio Capture
**Deliverable**: `IAudioSource`, `WasapiAudioSource`.

Tests (fake NAudio provider):
- `Start` emits `ChunkReady` with 512-sample float32 arrays
- `Running→Paused` stops emission; `Paused→Running` restarts
- `→Shutdown` stops permanently
- Resampling path: when device native format ≠ 16kHz float32, `MediaFoundationResampler` is inserted; output chunks are still 512 float32 samples at 16kHz

### Phase 7 — ViewModel and Main Window
**Deliverable**: `MainWindowViewModel`, `MainWindow.xaml/.cs`, `LoadingPage.xaml/.cs`.

Tests (synchronous `IDispatcherQueueAdapter`):
- `Apply(DisplayInstructions)` updates `FinalizedText` and `PartialText`
- `AppState.Paused` → `IsPaused = true`; buttons reflect state

### Phase 8 — Text Insertion and QuickEntry
**Deliverable**: `KeyboardSimulator`, `TextInserter`, `InsertionController`, `FocusTracker`, `GlobalHotkeyListener`, `QuickEntrySubscriber`, `QuickEntryController`, `QuickEntryWindow.xaml/.cs`.

Tests:
- `TextInserter`: `OnFinalization` calls `keyboard.TypeText` when enabled; skips when disabled
- `FocusTracker`: save/restore calls P/Invoke via injectable delegate
- `GlobalHotkeyListener`: simulated `WM_HOTKEY` fires callback exactly once
- `QuickEntryController`: hotkey → shows popup; Enter → types text and restores focus; Esc → cancels

### Phase 9 — Orchestrator and Full Integration
**Deliverable**: `ClientOrchestrator.cs` (complete), `App.xaml.cs` startup.

Integration tests (in-process fake WS server using `System.Net.WebSockets.HttpListenerWebSocketContext`):
- Connect → receive `session_created` → state transitions to `Running`
- Inject audio → verify binary frames arrive at fake server with correct format (golden byte check)
- Fake server sends `recognition_result` → verify text appears in ViewModel
- Fake server sends `session_closed` → verify state transitions to `Shutdown` without client sending `control_command`
- Client-initiated shutdown → verify `control_command shutdown` sent before socket closes

Real audio fixture test (using `tests/fixtures/en.wav` from the Python test suite):
- Feed WAV samples through the audio pipeline (bypass WASAPI, inject via `IAudioSource` fake backed by WAV data)
- Route through fake transport → fake server that echoes back a `recognition_result` with `text="one"` → verify `TextFormatter` produces correct `DisplayInstructions`
- Validates end-to-end formatting and transport against known speech content, consistent with project TDD policy

---

## Critical Reference Files

- [src/client/network/codec.py](src/client/network/codec.py) — exact byte layout for `AudioFrameEncoder`
- [src/client/tk/WsClientTransport.py](src/client/tk/WsClientTransport.py) — async/sync bridge design, drop semantics
- [src/client/tk/gui/TextFormatter.py](src/client/tk/gui/TextFormatter.py) — paragraph-break state machine, 2.0s threshold, duplicate suppression
- [src/client/tk/ClientApplicationState.py](src/client/tk/ClientApplicationState.py) — state transition table, outside-lock observer notification
- [docs/client-server-api-spec.md](docs/client-server-api-spec.md) — authoritative wire protocol

---

## Verification

End-to-end test procedure:
1. Start Python server: `source venv/Scripts/activate && python main.py --server-only`
2. Note printed port (e.g. `ws://127.0.0.1:54321`)
3. Run WinUI client: `SttClient.exe --server-url=ws://127.0.0.1:54321`
4. Loading screen appears → transitions to main window
5. Speak → partial text (gray) updates in real time → final text (black) committed on silence
6. Press Ctrl+Space → QuickEntry popup appears with live transcription
7. Switch to Notepad, press Enter in popup → text typed into Notepad
8. Close window → graceful shutdown, server session closes
