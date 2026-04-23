# STT Server and System Architecture

Client-specific architecture is documented separately in `src/client/tauri/ARCHITECTURE.md`.

## Overview

The system is a two-process application:

- the **Python server** owns model loading, audio preprocessing, VAD, window assembly, recognition, overlap resolution, and WebSocket session lifecycle
- the **Tauri desktop client** owns capture, UI, quick entry, insertion, and client-side transport behavior

This document describes the **server and system architecture**. For Tauri internals, see `src/client/tauri/ARCHITECTURE.md`.

---

## Startup and Deployment

`main.py` is a thin entry point. It resolves paths, builds `ModelRegistry` and `ModelManager`, parses CLI args, constructs `StartupController`, and maps typed errors to exit codes.

`StartupController.run()` owns startup decisions:

1. validates the environment (default mode verifies the Tauri binary exists; `--server-only` skips)
2. sets up logging
3. checks model availability and resolves `StartupDecisions` from CLI args
4. in `--server-only` with a missing model, auto-downloads or prompts on the CLI
5. starts web socket server manager `ServerApp`
6. creates and attaches the recognizer if the model is present, transitions to `running`
7. in default mode, launches the Tauri client with `--server-url=...` and watches for its exit
8. in `--server-only` mode, prints the server URL and QR code and stays headless
9. blocks on the web socker server lifecycle; stops the server


### Model availability at startup

The startup path has two distinct behaviors:

- if the model already exists, startup creates and attaches the recognizer immediately and the server can move to `running`
- if the model is missing:
  - in default mode, the server starts in `waiting_for_model` and the connected Tauri client drives model download over the protocol
  - in server-only mode, startup can auto-download or prompt on the CLI, depending on flags and TTY availability

---

## Server Runtime Topology

The Python server is composed of:

- `ApplicationState`
  - server lifecycle state machine
  - valid states: `starting`, `waiting_for_model`, `running`, `shutdown`

- `ServerApp`
  - web socket server manager
  - wires `RecognizerService`, `SessionManager`, `WsServer`, `CommandController`, readiness coordination, and broadcasters

- `WsServer`
  - owns the asyncio WebSocket server
  - runs on a dedicated daemon thread

- `SessionManager`
  - creates and destroys `ClientSession` instances
  - owns session map
  - broadcasts `server_state` transitions to active sessions
  - handles graceful close and shutdown-wide close

- `RecognizerService`
  - single shared multiplexer for multiple sessions and an ASR inference
  - routes recognizer output back to the owning session

- per-session workers inside `ClientSession`
  - `SoundPreProcessor`
  - `SpeechEndRouter`
  - `IncrementalTextMatcher`
  - `WsResultSender`

- server-wide protocol broadcast helpers
  - `DownloadProgressNotifier`

---

## Session and Audio Pipeline

Each websocket connection gets one `ClientSession`. ClientSession instance owns stateful per-client processing objects.

```text
audio websocket frame
  -> WsAudioReceiver
  -> ClientSession.chunk_queue
  -> SoundPreProcessor
  -> speech_queue
  -> SpeechEndRouter
  -> RecognizerService.input_queue
  -> RecognizerService inference thread
  -> recognizer_output_queue
  -> IncrementalTextMatcher
  -> WsResultSender
  -> websocket text frames
```

### `SoundPreProcessor`

Responsibilities:

- consume raw audio chunks from `chunk_queue`
- apply normalization and VAD
- emit `AudioSegment` windows
- emit `SpeechEndSignal` boundaries

It uses `GrowingWindowAssembler` synchronously, not as a separate thread.

### `SpeechEndRouter`

Responsibilities:

- process in-flight signals
- assign monotonically increasing message IDs
- hold speech-end boundaries until the corresponding in-flight work is acknowledged
- route recognizer output to the matcher queue in protocol-safe order

Message IDs are session-partitioned:

- `message_id = session_index * 10_000_000 + local_id`

That lets `RecognizerService` recover the owning session with:

- `session_index = message_id // 10_000_000`

### `RecognizerService`

Responsibilities:

- own the single shared inference thread
- read `AudioSegment` items from the shared input queue
- call `recognize_window()`
- route `RecognitionTextMessage` and terminal `RecognizerAck` to the correct session output queue


### `IncrementalTextMatcher`

Responsibilities:

- resolve overlap between successive recognition windows
- suppress duplicates
- emit partial and final text updates

It publishes to `WsResultSender`, which is the server-side bridge from synchronous pipeline code to async websocket sends.

### `WsResultSender`

Responsibilities:

- accept sync-thread publish calls
- enqueue encoded messages onto a bounded async send queue
- run one async drain task per session
- support `wait_for_drain()` / `flush()` for ordered session startup and graceful close

---

## Session Lifecycle

### Connect flow

When a client connects:

1. `WsServer` accepts the connection
2. `SessionManager.create_session()` creates a `ClientSession`
3. `ClientSession.start()` registers the session with `RecognizerService` and starts its workers
4. the server queues two welcome messages through the session's send queue:
   - `session_created`
   - `server_state`
5. the send queue is flushed
6. only after that is the session added to the active-session map

This ordering matters: it guarantees `session_created` is the first frame the client sees, and prevents `server_state` broadcasts from interleaving ahead of the connect-time welcome pair.

### Audio acceptance

`WsAudioReceiver` accepts audio only while the server is in `running`.

If the server is in `starting`, `waiting_for_model`, or `shutdown`, audio is rejected with:

- `MODEL_NOT_READY`

If the binary frame contains the wrong `session_id`, the error is:

- `SESSION_ID_MISMATCH`

If `chunk_queue` is full, the error is:

- `BACKPRESSURE_DROP`

### Graceful close

Client-initiated close uses:

- `control_command(command="close_session")`

The graceful close path:

1. stops new ingress from `SoundPreProcessor`
2. waits for pending recognizer work via `RecognizerService.flush_session(...)`
3. waits for matcher and sender queues to drain
4. sends `session_closed`
5. stops session workers and unregisters the session

Server-side error handling reuses the same graceful close path with `reason="error"`.

### Shutdown flow

When `ServerApp.stop()` transitions `ApplicationState` to `shutdown`:

- `SessionManager` fans out `server_state=shutdown`
- `SessionManager.close_all_sessions()` closes active sessions gracefully
- each session attempts to send `session_closed(reason="error", message="server shutdown")`
- `WsServer` then stops serving

---

## Protocol Surface

Protocol dataclasses live in `src/network/types.py`. Codec logic lives in `src/network/codec.py`.

### Client -> Server

- binary `audio_chunk` via `WsAudioFrame`
- JSON `control_command` via `WsControlCommand`

Supported control commands:

- `close_session`
- `list_models`
- `download_model`

### Server -> Client

- `session_created`
- `server_state`
- `recognition_result`
- `session_closed`
- `error`
- `model_list`
- `model_status`
- `download_progress`

### Server state values

- `starting`
- `waiting_for_model`
- `running`
- `shutdown`

### Error codes currently used

- `INVALID_AUDIO_FRAME`
- `UNKNOWN_MESSAGE_TYPE`
- `SESSION_ID_MISMATCH`
- `BACKPRESSURE_DROP`
- `INTERNAL_ERROR`
- `MODEL_NOT_READY`
- `DOWNLOAD_IN_PROGRESS`
- `INVALID_MODEL_NAME`

---

## Server-Controlled Model Download Flow

Model download is part of the live protocol, not just startup bootstrapping.

### Request handling

`CommandController` handles:

- `list_models`
- `download_model`
- `close_session`

For `download_model`:

- if the model is already present, it returns `model_status(status="ready")`
- if a new download starts, it returns `model_status(status="downloading")`
- if another download is already in progress, it returns `DOWNLOAD_IN_PROGRESS`
- if the name is invalid, it returns `INVALID_MODEL_NAME`

If a model is already present but the recognizer is not attached yet, the command path also triggers `ensure_model_ready(...)` so the server can still recover to `running`.

### Progress and completion

`DownloadWorker` runs on a worker thread outside the websocket loop. `DownloadProgressNotifier` receives worker-thread callbacks and publishes `download_progress` events through `IServerMessageBroadcaster`, which is implemented by `SessionManager`.

Broadcast delivery follows the same server send path as normal session messages:

1. `DownloadProgressNotifier` emits a server message through `IServerMessageBroadcaster`
2. `SessionManager.broadcast(...)` snapshots active sessions and fans out the encoded frame
3. `ClientSession.send_encoded(...)` forwards the frame to `WsResultSender`
4. `WsResultSender._enqueue(...)` performs the thread-safety handoff into the per-session async send queue

Progress behavior:

- intermediate `download_progress(status="downloading")` updates are throttled
- final `complete` and `error` updates are always emitted
- broadcasts iterate active sessions and enqueue frames onto each session's existing send queue

### Attach-on-download success

On successful download:

1. `ServerApp` attempts to attach the recognizer
2. if attachment succeeds:
   - broadcast `download_progress(status="complete")`
   - if the server was `waiting_for_model`, transition to `running`
3. if attachment fails:
   - broadcast `download_progress(status="error")`
   - remain out of `running`

---

## State Broadcasting

The server has two main broadcast flows:

### `SessionManager`

- observes `ApplicationState`
- encodes `server_state`
- snapshots its active session map
- implements `IServerMessageBroadcaster`
- exposes `broadcast(...)` as the shared fanout path
- enqueues the encoded frame onto each session's send queue

### `DownloadProgressNotifier`

- receives threaded download callbacks
- throttles progress updates per model
- publishes through `IServerMessageBroadcaster`
- relies on `SessionManager.broadcast(...)` and `ClientSession.send_encoded(...)` for fanout
- relies on `WsResultSender._enqueue(...)` for thread-safe queue handoff

The design intentionally avoids a separate global broadcast queue. Per-session ordering is preserved by reusing each `ClientSession`'s existing outbound queue.

---

## Threading and Concurrency

### Server threads and tasks

- `WsServer`
  - dedicated daemon thread
  - owns the asyncio event loop

- websocket connection handlers
  - coroutines on the `WsServer` loop

- `WsResultSender._drain_loop`
  - async task per session on the `WsServer` loop

- `RecognizerService`
  - one daemon inference thread for the whole server

- `SoundPreProcessor`
  - one thread per session

- `SpeechEndRouter`
  - one thread per session

- `IncrementalTextMatcher`
  - one thread per session

- `DownloadWorker`
  - worker thread used by model download handling

### Important thread-safe bridges

- pipeline threads -> per-session async websocket send queue (`WsResultSender`)
- download worker thread -> `DownloadProgressNotifier` -> `IServerMessageBroadcaster` -> `SessionManager.broadcast(...)` -> `ClientSession.send_encoded(...)` -> `WsResultSender._enqueue(...)`
- `ApplicationState` observers -> synchronous notifications outside the state lock

---

## System Boundary

The Python server is authoritative for:

- readiness to accept audio
- model availability state
- recognition lifecycle
- graceful session close behavior
- session teardown and shutdown ordering

The Tauri client is authoritative for:

- local audio capture
- client UI and desktop behavior
- rendering transcript and model-download state
- local quick-entry and insertion features

Client-specific details live in `src/client/tauri/ARCHITECTURE.md`.

---

## Appendix: Server Code Entity Catalogue by Layer

A cross-reference of server entities grouped by architectural roles.
Client code, tests, and build scripts are excluded.

### Layer 1 — System / Infrastructure

*Direct interaction with OS, filesystem, network, Inference Runtime, logging, CLI.*

Shared:
- [src/PathResolver.py](src/PathResolver.py) — `PathResolver`, `ResolvedPaths`, `DistributionMode`; MSIX/portable/development path resolution
- [src/LoggingSetup.py](src/LoggingSetup.py) — `setup_logging()` (file rotation + console)
- [src/StartupArgs.py](src/StartupArgs.py) — `StartupArgs` dataclass, `from_argv()` (CLI parsing)
- [src/LicenseCollector.py](src/LicenseCollector.py) — `LicenseCollector`

Server I/O (see *Server Runtime Topology* for roles):
- [src/server/WsServer.py](src/server/WsServer.py) — `WsServer`
- [src/server/WsAudioReceiver.py](src/server/WsAudioReceiver.py) — `handle_audio_frame()`
- [src/server/WsResultSender.py](src/server/WsResultSender.py) — `WsResultSender`, `_SupportsAsyncSend` protocol
- [src/server/qr_display.py](src/server/qr_display.py) — `print_qr_code()`, `_build_matrix()`, `_render()` (terminal QR output)

ASR infrastructure:
- [src/asr/ModelLoader.py](src/asr/ModelLoader.py) — `load_model()`, `ModelLoadError` (ONNX load + exception translation)
- [src/asr/VoiceActivityDetector.py](src/asr/VoiceActivityDetector.py) — `VoiceActivityDetector` (Silero VAD ONNX session)
- [src/asr/ExecutionProviderManager.py](src/asr/ExecutionProviderManager.py) — `ExecutionProviderManager` (DirectML/CPU detection & fallback)

Model download I/O:
- [src/downloader/ModelDownloader.py](src/downloader/ModelDownloader.py) — `ModelDownloader` (CDN HTTP fetch, atomic manifest write, partial-file cleanup)
- [src/downloader/ModelDownloadCliDialog.py](src/downloader/ModelDownloadCliDialog.py) — `ModelDownloadCliDialog` (terminal yes/no prompt used by `StartupController` in `--server-only` mode)

### Layer 2 — Protocol / Wire

*Message types and codecs. See also* **Protocol Surface** *above.*

- [src/network/types.py](src/network/types.py) — `WsAudioFrame`, `WsControlCommand`, `WsSessionCreated`, `WsRecognitionResult`, `WsSessionClosed`, `WsServerState`, `WsError`, `WsModelInfo`, `WsModelList`, `WsModelStatus`, `WsDownloadProgress`; `ClientTextMessage`/`ServerMessage` unions
- [src/network/codec.py](src/network/codec.py) — `SessionIdMismatchError`, `decode_audio_frame()`, `encode_server_message()`, `decode_client_message()`
- [src/server/protocols.py](src/server/protocols.py) — `IModelCommandHandler`, `IDownloadProgressEvents`, `IModelReadinessCoordinator`, `IServerMessageBroadcaster` protocols
- [src/server/WsMessageRouter.py](src/server/WsMessageRouter.py) — `RoutedAudioMessage`, `RoutedCommandMessage`, `RoutedProtocolError`; `route_incoming_message()`
- [src/server/broadcast.py](src/server/broadcast.py) — `_SHUTDOWN` sentinel

### Layer 3 — Model / Domain

*Pure logic: algorithms, DTOs, protocol/interface definitions.*

Shared DTOs & interfaces:
- [src/types.py](src/types.py) — `AudioSegment`, `SpeechEndSignal`, `RecognitionResult`, `RecognitionTextMessage`, `RecognizerAck`, `RecognizerFreeSignal`, `DisplayInstructions`; queue-item aliases `ChunkQueueItem`, `SpeechQueueItem`, `RecognizerOutputItem`, `MatcherQueueItem`, `RouterControlItem`
- [src/protocols.py](src/protocols.py) — `TextRecognitionSubscriber` protocol
- [src/RecognitionResultPublisher.py](src/RecognitionResultPublisher.py) — `RecognitionResultPublisher` protocol

Audio / ASR domain logic (see *Session and Audio Pipeline* for responsibilities):
- [src/sound/SoundPreProcessor.py](src/sound/SoundPreProcessor.py) — `ProcessingStatesEnum`, `AudioProcessingState`, `SoundPreProcessor`¹
- [src/sound/GrowingWindowAssembler.py](src/sound/GrowingWindowAssembler.py) — `GrowingWindowAssembler`
- [src/asr/Recognizer.py](src/asr/Recognizer.py) — `Recognizer` (inference + confidence extraction)¹
- [src/asr/ModelDefinitions.py](src/asr/ModelDefinitions.py) — `IModelDefinition`, `ModelStatus`, `BaseModelDefinition`, `ParakeetAsrModel`, `SileroVadModel`

Post-recognition text logic:
- [src/postprocessing/IncrementalTextMatcher.py](src/postprocessing/IncrementalTextMatcher.py) — `IncrementalTextMatcher`¹ (see *Session and Audio Pipeline*)
- [src/postprocessing/TextNormalizer.py](src/postprocessing/TextNormalizer.py) — `TextNormalizer` (case / punctuation / Unicode normalization for duplicate detection)
- [src/postprocessing/PostRecognitionFilter.py](src/postprocessing/PostRecognitionFilter.py) — `PostRecognitionFilter` (filler-word and confidence-gated word filtering)

¹ *`SoundPreProcessor`, `Recognizer`, and `IncrementalTextMatcher` each own a worker thread; the signal-processing or text-matching algorithm is their reason-to-exist.*

### Layer 4 — Service / Orchestration

*Composes domain objects, owns lifecycle, coordinates threads/tasks, wires queues.*

Shared / top-level:
- [src/ApplicationState.py](src/ApplicationState.py) — `ApplicationState` (see *Server Runtime Topology*)
- [src/SpeechEndRouter.py](src/SpeechEndRouter.py) — `SpeechEndRouter` (see *Session and Audio Pipeline*)

Server services (see *Server Runtime Topology*):
- [src/server/SessionManager.py](src/server/SessionManager.py) — `SessionManager`
- [src/server/ClientSession.py](src/server/ClientSession.py) — `ClientSession`
- [src/server/RecognizerService.py](src/server/RecognizerService.py) — `RecognizerService`, `_SessionDrainState`, `_RecognizerInputQueue`
- [src/server/RecognizerReadinessCoordinator.py](src/server/RecognizerReadinessCoordinator.py) — `RecognizerReadinessCoordinator` (deferred recognizer attach and post-download readiness)
- [src/server/CommandController.py](src/server/CommandController.py) — `CommandController` (see *Server-Controlled Model Download Flow*)
- [src/server/ModelCommandHandler.py](src/server/ModelCommandHandler.py) — `ModelCommandHandler`
- [src/server/DownloadProgressNotifier.py](src/server/DownloadProgressNotifier.py) — `DownloadProgressNotifier`

ASR services:
- [src/asr/ModelRegistry.py](src/asr/ModelRegistry.py) — `ModelRegistry`
- [src/asr/ModelManager.py](src/asr/ModelManager.py) — `ModelManager`
- [src/asr/RecognizerFactory.py](src/asr/RecognizerFactory.py) — `RecognizerFactory`, `IRecognizerFactory` protocol
- [src/asr/SessionOptionsFactory.py](src/asr/SessionOptionsFactory.py) — `SessionOptionsFactory`
- [src/asr/SessionOptionsStrategy.py](src/asr/SessionOptionsStrategy.py) — `SessionOptionsStrategy` ABC, `IntegratedGPUStrategy`, `DiscreteGPUStrategy`, `CPUStrategy`

Download orchestration:
- [src/downloader/DownloadWorker.py](src/downloader/DownloadWorker.py) — `DownloadWorker` (see *Server-Controlled Model Download Flow*)

### Layer 5 — Entry-Point / Application

*Top-level composition roots (see* **Startup and Deployment** *above).*

- [main.py](main.py) — entry point
- [src/StartupController.py](src/StartupController.py) ¹ — `StartupController`, `StartupDecisions`; `TauriBinaryNotFoundError`, `MissingModelsError`, `DownloadCancelledError`
- [src/server/ServerApp.py](src/server/ServerApp.py) — `ServerApp`

`StartupController` is application bootstrap, and `ServerApp` is managing websocket server.


### Classifying new entities

When a new module is added, classify it with these litmus questions:
- Touches OS, device, socket, subprocess, or ONNX? → **System**
- Wire-format type or codec? → **Protocol**
- Pure logic / DTO / interface? → **Model / Domain**
- Owns a pipeline, thread, or lifecycle? → **Service**
- Wires every other layer together? → **Entry-Point**
