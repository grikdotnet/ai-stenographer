# Tauri Client Architecture

## Overview

The Tauri client is the active desktop client for this project.

It is responsible for:

- connecting to the Python STT server over WebSocket
- starting and supervising a development-layout Python server when no external server URL is configured
- capturing live microphone audio or streaming an input file
- managing client lifecycle state
- rendering transcript and model-download UI
- exposing pause/resume/clear/model commands through Tauri IPC
- driving quick entry and text insertion on supported platforms

The Python server remains responsible for VAD, windowing, ASR inference, overlap resolution, and server-side session lifecycle.

---

## Startup Flow

### Owned-server startup

In normal desktop mode without `--server-url` or `STT_SERVER_URL`:

1. `src-tauri/src/main.rs` creates a `ServerSupervisor`
2. `ServerSupervisor` starts `venv/Scripts/python.exe main.py --host=127.0.0.1`
3. the supervisor reads stdout until `Server listening on ws://...`
4. `ClientOrchestrator` connects to the parsed URL
5. audio capture is armed after connect but does not bind the microphone or replay file yet
6. capture starts only after the server sends `server_state=running`
7. normal app shutdown terminates the owned Python child

If the owned child exits unexpectedly, the supervisor allows one unstable
restart. The crash counter resets only after the orchestrator marks the
supervisor connected and the child remains stable for the reset window. The
connection marker is fired after the `session_created` handshake stores the
transport `session_id`; process spawn by itself is not treated as stability.

For each owned-server process generation, `ServerSupervisor` retains a bounded
in-memory tail of the last 50 combined stdout/stderr lines with stream labels.
Startup failures before the `Server listening on ws://...` line and terminal
runtime crashes include the process phase, exit status when available, and that
recent output in the existing connection-error surface. The diagnostics are not
persisted separately and are not uploaded automatically.

### External-server startup

With `--server-url` or `STT_SERVER_URL`, Tauri does not spawn Python. It keeps
the direct external-server workflow and connects to the supplied URL.

In direct Tauri execution:

- `src-tauri/src/main.rs` parses CLI args
- constructs `AppStateManager`
- optionally starts `ServerSupervisor` when no external URL is configured
- constructs `ClientOrchestrator`
- wires Tauri event emission, frontend state updates, and optional quick-entry support
- calls `connect()`
- on successful connect, calls `start_audio()` to arm capture until `server_state=running`

Headless mode is also supported by the Tauri binary through `--headless`.

---

## Main Components

### `ClientOrchestrator`

`src-tauri/src/orchestrator.rs`

The orchestrator is the core client coordinator. It owns:

- transport lifecycle
- session handshake handling
- protocol version validation
- mapping server messages to client state
- audio start/stop behavior
- command building for `close_session`, `list_models`, and `download_model`
- event emission to the frontend

It is the main boundary between the Rust backend and both the WebSocket protocol and the Tauri UI event model.

### `ServerSupervisor`

`src-tauri/src/supervisor.rs`

The supervisor owns the development Python child process when Tauri is the
visible parent. It resolves the repository-local `venv/Scripts/python.exe`,
starts `main.py --host=127.0.0.1`, parses the server URL from stdout, drains
stdout/stderr, terminates the child on normal shutdown, and applies the
restart-once policy for unstable crashes. It also keeps a bounded per-process
stdout/stderr diagnostics buffer so owned-server startup and terminal runtime
failures can be surfaced through the existing connection-error UI.

### `WsClientTransport`

`src-tauri/src/transport.rs`

The transport owns the WebSocket connection and async send/receive loops.

On connect it creates:

- a binary drain loop for outgoing audio frames
- a text drain loop for outgoing control commands
- a receive loop for incoming server text messages

It stores the current `session_id` after the handshake and sends `close_session` on local shutdown unless the stop was server-initiated.

### `AppStateManager`

`src-tauri/src/state.rs`

The client lifecycle states are:

- `Starting`
- `WaitingForServer`
- `Running`
- `Paused`
- `Shutdown`

Important semantics:

- `session_created` moves the client to `WaitingForServer`
- `server_state=running` moves the client to `Running`
- `server_state=waiting_for_model` moves the client back to `WaitingForServer`
- `server_state=shutdown` moves the client to `Shutdown`
- if the server protocol version does not match, the client emits a connection error and disconnects

### Tauri commands

`src-tauri/src/commands.rs`

The Tauri backend exposes commands such as:

- `get_state`
- `get_connection_status`
- `pause`
- `resume`
- `clear`
- `list_models`
- `download_model`
- quick-entry commands

The React UI calls these through `invoke(...)`.

### React UI

`src-ts/App.tsx` and `src-ts/components/*`

The React layer subscribes to backend-emitted events and keeps view state in sync with:

- transcript updates
- connection status
- client lifecycle state changes
- server state changes
- model list updates
- model status responses
- download progress

The UI treats `waiting_for_model` as a first-class state and exposes model download controls.

---

## Protocol Handling

The Tauri client consumes the server protocol defined in `src/network/types.py` and mirrored in Rust protocol types.

### Incoming messages used by the client

- `session_created`
- `server_state`
- `recognition_result`
- `session_closed`
- `error`
- `model_list`
- `model_status`
- `download_progress`

### Outgoing messages used by the client

- binary `audio_chunk` frames
- `control_command(command="close_session")`
- `control_command(command="list_models")`
- `control_command(command="download_model")`

### Handshake

The connect-time handshake is:

1. client opens WebSocket
2. server sends `session_created`
3. server sends `server_state`
4. client stores `session_id`
5. client waits for `server_state=running` before sending audio

This means `session_created` is not treated as proof that the model is ready.

### Protocol version validation

The client expects protocol version `v1`.

If the server sends a different version:

- the client emits `stt://connection-status` with an error
- the client transitions toward shutdown
- the transport disconnects instead of hanging forever in a fake connecting state

---

## Audio Flow

### Capture path

The client can use:

- live microphone capture through `CpalAudioSource`
- file-driven capture through `FileAudioSource`

`ClientOrchestrator.start_audio()` arms capture for the current connection. It installs the callback and starts the selected audio source only when the latest server state is `running`.

The installed callback:

1. reads the current `session_id`
2. checks that the client is in `Running`
3. encodes samples as an audio frame
4. pushes the frame into the transport's bounded binary sender channel

### Backpressure behavior

`WsClientTransport` uses bounded channels.

- closed channel is treated as a transport error
- full channel drops the newest frame by design

This keeps capture from blocking on network I/O.

### Readiness and running-state gating

Audio capture startup is explicitly gated by server readiness and client state.

The client does not bind the microphone or start file replay while the latest server state is `starting`, `waiting_for_model`, or `shutdown`. If the server moves away from `running`, the orchestrator stops active capture and keeps capture armed unless the user manually paused.

Even after capture has started, the callback returns early unless the client is in `Running`.

That means the client stops sending audio whenever state moves away from `Running`, including:

- `Paused`
- `WaitingForServer`
- `Shutdown`

---

## Transcript Flow

`recognition_result` messages are decoded in the orchestrator and dispatched through `RecognitionFanOut`.

The default subscriber bridge updates `TextFormatter`, which then drives:

- finalized utterance history
- preliminary text
- transcript events emitted to the frontend

The backend emits `stt://transcript-update`, and the React app updates view state accordingly.

---

## Model Download Flow

The model-management flow is part of the active Tauri client architecture.

### Request side

The UI can:

- refresh model list with `list_models`
- request a download with `download_model(modelName)`

The orchestrator builds the corresponding control commands using the active `session_id`.

### Response side

The client reacts to:

- `model_list`
  - updates model inventory in the UI
- `model_status`
  - tells the UI whether a request was accepted or the model is already ready
- `download_progress`
  - drives progress/error/completion feedback
- `server_state`
  - controls the transition from waiting to running once the server is actually ready

When the UI sees `waiting_for_model`, it refreshes the model list automatically.

---

## Frontend Event Model

The Rust backend emits these events for the React frontend:

- `stt://connection-status`
- `stt://state-changed`
- `stt://server-state`
- `stt://transcript-update`
- `stt://model-list`
- `stt://model-status`
- `stt://download-progress`

The frontend uses them to keep a single view-state object synchronized with backend behavior.

This event model avoids pushing protocol parsing into TypeScript.

---

## Quick Entry and Insertion

On Windows, the Tauri client also wires quick-entry behavior:

- global hotkey registration
- popup quick-entry window
- insertion controller integration
- focus save/restore around text insertion

These features are client-side conveniences layered on top of the core transcription client. They do not affect the WebSocket protocol.

---

## Concurrency Model

The Tauri client mixes several concurrency domains:

- Tauri runtime and window event loop
- Tokio tasks for connect/receive/send orchestration
- CPAL callback thread for live audio capture
- optional OS hotkey thread on Windows

Important bridges:

- audio callback thread -> bounded Tokio sender for binary audio frames
- backend Rust state changes -> emitted frontend events
- transport receive loop -> synchronous protocol dispatch in the orchestrator

The design keeps the hot audio path and the UI path decoupled.

---

## Relationship to the Server

The Tauri client is intentionally thin with respect to speech recognition.

It does not do:

- VAD
- chunk window assembly
- ASR inference
- overlap resolution

Those remain server responsibilities.

The client is authoritative only for:

- capture
- transport
- local UI state
- quick entry / insertion behavior

Server state remains the source of truth for readiness to stream audio.

