# STT Pipeline Architecture Documentation

## System Overview

The Speech-to-Text system is a **two-process client-server application**. The server process handles all audio processing, VAD, and ASR inference. The client process handles microphone capture, the Tk GUI, and text insertion. The two processes communicate over a local WebSocket connection (protocol v1).

Launching `python main.py` starts the server in-process and spawns `src/client/client.py` as a subprocess. A daemon watcher thread in `main.py` monitors the subprocess: when it exits (user closes the window or crash), the watcher calls `server_app.stop()`. Running `python main.py --server-only` starts the server headlessly without spawning a client.

Application state is split: `ServerApplicationState` manages server lifecycle (`starting → running → shutdown`, component observers only, no pause, no GUI scheduling). `ClientApplicationState` manages the full client lifecycle (`starting → running ⟷ paused → shutdown`) including GUI observers and tkinter scheduling.

## Router Protocol Overview

`SpeechEndRouter` mediates flow between `SoundPreProcessor`, `RecognizerService`, and `IncrementalTextMatcher`. It enforces one in-flight `AudioSegment` at a time per session, assigns monotonic `message_id` values, forwards `RecognitionTextMessage` payloads to the matcher input queue, and unblocks on terminal `RecognizerAck`. `SpeechEndSignal` boundaries are held until all in-flight audio for that utterance is acknowledged.

In the multi-session server each `SpeechEndRouter` instance receives a `first_message_id` constructor parameter equal to `session_index * 10_000_000 + 1`. This partitions the message ID space so `RecognizerService` can route results back to the correct session by pure arithmetic (`message_id // 10_000_000`), with no routing table lookup on the hot path.

---

## Entry Points and Process Startup

### `main.py`

1. Resolves paths and loads config.
2. Checks for missing ASR models. In `--server-only` mode: prints missing model names to stderr and exits with code 1 if any are absent (no GUI dialogs).
3. Constructs `ServerApp` and calls `server_app.start()` — binds the WebSocket server to an OS-assigned port.
4. **Default mode**: spawns `src/client/client.py` as a subprocess with `--server-url=ws://127.0.0.1:<port>`. Starts a daemon watcher thread calling `client_proc.wait()`, then `server_app.stop()` + `client_proc.terminate()`.
5. **`--server-only` mode**: blocks on `WsServer.join()`.

### `src/client/client.py`

Subprocess entry script. Not intended for direct user invocation.

1. Parses `--server-url=ws://host:port` (required).
2. Displays `LoadingWindow` while connecting to the WebSocket server.
3. Reads the `session_created` JSON frame; calls `ClientApp.from_session_created()`.
4. Sets `transport._loop` to the asyncio loop started by `WsClientTransport`.
5. Calls `client_app.start()` and runs `root.mainloop()`.
6. On `ConnectionClosed`: `WsClientTransport` transitions `ClientApplicationState` to `shutdown`; the observer chain stops `AudioSource` and exits the Tk loop. Exits with code 1.

### CLI Flags

| Flag | Effect |
|---|---|
| *(default)* | Server + Tk client subprocess |
| `--server-only` | Headless server; no GUI |
| `-v` | Verbose logging |
| `--input-file=path` | `FileAudioSource` replaces `AudioSource` (client-side, for testing) |

---

## Network Protocol (v1)

All wire types are defined in `src/network/types.py`; encode/decode is in `src/network/codec.py`.

### Client → Server

| Type | Wire format | Description |
|---|---|---|
| `WsAudioFrame` | Binary | 4-byte little-endian header_len + JSON header + raw float32 PCM payload |
| `WsControlCommand` | JSON text | `{"type": "control_command", "command": "shutdown", "session_id": ..., "timestamp": ...}` |

### Server → Client

| Type | Wire format | Description |
|---|---|---|
| `WsSessionCreated` | JSON text | Sent immediately on connect; contains `session_id`, `protocol_version`, `server_time` |
| `WsRecognitionResult` | JSON text | `status`: `"partial"` or `"final"`; contains text, timing, chunk_ids, utterance_id, confidence fields |
| `WsSessionClosed` | JSON text | Reason: `"shutdown"`, `"timeout"`, or `"error"` |
| `WsError` | JSON text | Non-fatal errors continue the session; `fatal: true` closes the connection |

### Error Codes (`WsError.error_code`)

`INVALID_AUDIO_FRAME`, `UNKNOWN_MESSAGE_TYPE`, `SESSION_ID_MISMATCH`, `BACKPRESSURE_DROP`, `PROTOCOL_VIOLATION`, `INTERNAL_ERROR`

### Session Lifecycle

```
client connects
  → server sends session_created (session_id, protocol_version, server_time)
  → client stamps session_id on all outbound WsAudioFrame headers
  → server streams recognition_result frames (partial + final)
  → client sends control_command shutdown (or closes WebSocket)
  → server sends session_closed, tears down ClientSession
```

---

## Module Organization

### Root (`src/`)

- **`ApplicationState.py`**: Base class; full state machine (`starting → running ⟷ paused → shutdown`); component + GUI observers; thread-safe mutations. Used as base by `ClientApplicationState`.
- **`ServerApplicationState.py`**: 3-state server-only state machine (`starting → running → shutdown`); component observers only; no GUI scheduling; no `paused` state.
- **`RecognitionResultPublisher.py`**: `@runtime_checkable` Protocol with `publish_partial_update()` and `publish_finalization()`. Implemented by `WsResultSender` (server) and `RecognitionResultFanOut` (client).
- **`SpeechEndRouter.py`**: Routes audio and results between `SoundPreProcessor`, `RecognizerService`, and `IncrementalTextMatcher`; enforces single-in-flight per session.
- **`types.py`**: `AudioSegment`, `RecognitionResult`, `RecognitionTextMessage`, `RecognizerAck`, `RecognizerFreeSignal`, `SpeechEndSignal`.
- **`protocols.py`**: `TextRecognitionSubscriber` protocol.

### Server (`src/server/`)

- **`ServerApp.py`**: Orchestrates `WsServer` + `RecognizerService`; server-mode entry point.
- **`WsServer.py`**: `websockets.serve()` asyncio loop on a daemon thread; delegates connections to `SessionManager`.
- **`SessionManager.py`**: Allocates 1-based session indices (atomic); creates and destroys `ClientSession` instances; observes `ServerApplicationState` for shutdown.
- **`ClientSession.py`**: Owns all per-client queues and pipeline workers; `start()` / `close()` lifecycle.
- **`RecognizerService.py`**: Single shared inference daemon thread; routes results by `message_id // 10_000_000`; `register_session()` / `unregister_session()` per-session output queues.
- **`WsAudioReceiver.py`**: Async coroutine running inside the WS connection handler; decodes binary `WsAudioFrame` → puts audio dicts onto `ClientSession.chunk_queue`; sends `BACKPRESSURE_DROP` error if queue is full.
- **`WsResultSender.py`**: Implements `RecognitionResultPublisher` protocol; sync-to-async bridge via bounded `asyncio.Queue(20)`; single drain async task sends JSON to the client WebSocket.

### Network (`src/network/`)

- **`types.py`**: `WsAudioFrame`, `WsControlCommand`, `WsSessionCreated`, `WsRecognitionResult`, `WsSessionClosed`, `WsError`; union aliases `ServerMessage`, `ClientTextMessage`.
- **`codec.py`**: `encode_audio_frame()`, `decode_audio_frame()`, `encode_server_message()`, `decode_client_message()`.

### Client (`src/client/`)

- **`client.py`**: Subprocess entry script; parses `--server-url`; shows `LoadingWindow`; calls `ClientApp.from_session_created()`.
- **`tk/ClientApp.py`**: Wires `AudioSource` + `AudioBridge` thread + `WsClientTransport` + Tk GUI; `from_session_created()` factory.
- **`tk/ClientApplicationState.py`**: Re-exports `ApplicationState`; full 4-state machine with GUI observer support; used by `AudioSource`, `PauseController`, `ControlPanel`, `WsClientTransport`.
- **`tk/WsClientTransport.py`**: Asyncio event loop on daemon thread; sync `send_audio_chunk()` → bounded `asyncio.Queue` → WebSocket; async receive loop → `RemoteRecognitionPublisher`.
- **`tk/RemoteRecognitionPublisher.py`**: Decodes incoming WS JSON text frames → dispatches to `RecognitionResultFanOut`.
- **`tk/RecognitionResultFanOut.py`**: Concrete client-side fan-out publisher; implements `RecognitionResultPublisher` protocol; manages subscribers (`TextFormatter`, `TextInsertionService`, `QuickEntryService`); thread-safe copy-on-notify.
- **`tk/asr/`**: Client copies of `ModelManager` and `DownloadProgressReporter` (used by `client.py` for model download dialogs).
- **`tk/controllers/`**: `PauseController`, `InsertionController` (moved from `src/controllers/`).
- **`tk/gui/`**: All Tk GUI widgets: `ApplicationWindow`, `ControlPanel`, `HeaderPanel`, `InsertionModePanel`, `TextDisplayWidget`, `TextFormatter`, `TextInserter`, `TextInsertionService`, `KeyboardSimulator`, `LoadingWindow`, `ModelDownloadDialog` (moved from `src/gui/`).
- **`tk/quickentry/`**: `QuickEntryService`, `QuickEntryController`, `QuickEntrySubscriber`, `QuickEntryPopup`, `GlobalHotkeyListener`, `FocusTracker`, `windows_effects` (moved from `src/quickentry/`).
- **`tk/sound/`**: `AudioSource`, `FileAudioSource` (moved from `src/sound/`; client-side only).

### Shared Processing (server-side at runtime)

- **`src/sound/`**: `SoundPreProcessor`, `GrowingWindowAssembler` — server-side; one instance per `ClientSession`.
- **`src/asr/`**: `Recognizer` (pure callable; see note below), `VoiceActivityDetector` (stateful, one per session), `ModelManager`, `ExecutionProviderManager`, `SessionOptionsFactory`, `SessionOptionsStrategy`, `DownloadProgressReporter`.
- **`src/postprocessing/`**: `IncrementalTextMatcher`, `PostRecognitionFilter`, `TextNormalizer`.

---

## Architecture Diagram

### Server-Side Session Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│  SERVER PROCESS                                                           │
│  asyncio event loop thread (WsServer)                                     │
│    WsAudioReceiver coroutine (per session)                                │
│      ← binary WsAudioFrame from client                                    │
│      → chunk_queue (per session, maxsize=100)                             │
├──────────────────────────────────────────────────────────────────────────┤
│  Per-Session Daemon Threads (owned by ClientSession)                      │
│                                                                           │
│  SoundPreProcessor thread                                                 │
│    ← chunk_queue.get()                                                    │
│    → _normalize_rms() → VoiceActivityDetector.process_frame()            │
│    → GrowingWindowAssembler.process_segment() [sync, no thread]          │
│    → speech_queue (AudioSegment + SpeechEndSignal)                        │
│                                                                           │
│  SpeechEndRouter thread                                                   │
│    ← speech_queue.get()                                                   │
│    → assigns message_id = session_index * 10_000_000 + local_id          │
│    → RecognizerService.input_queue (shared across all sessions)           │
│    ← recognizer_output_queue (per session, keyed by session_index)       │
│    → matcher_queue (RecognitionResult + SpeechEndSignal)                  │
│                                                                           │
│  IncrementalTextMatcher thread                                            │
│    ← matcher_queue.get()                                                  │
│    → overlap resolution, duplicate detection                              │
│    → WsResultSender.publish_partial_update / publish_finalization         │
├──────────────────────────────────────────────────────────────────────────┤
│  Server-Global                                                            │
│                                                                           │
│  RecognizerService daemon thread (shared, one per server)                 │
│    ← input_queue.get() (AudioSegment from any session)                   │
│    → Recognizer.recognize_window(segment) [blocking inference]            │
│    → session_index = message_id // 10_000_000                            │
│    → session_output_queues[session_index].put_nowait(result / ack)        │
│                                                                           │
│  WsResultSender (per session, asyncio task in WsServer loop)              │
│    ← publish_*() calls from IncrementalTextMatcher thread                 │
│    → loop.call_soon_threadsafe → asyncio.Queue(20) → websocket.send()    │
│    → JSON WsRecognitionResult frame → client                             │
└──────────────────────────────────────────────────────────────────────────┘
```

### Client-Side Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CLIENT PROCESS (src/client/client.py subprocess)                        │
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

### Process Startup Sequence

```
main.py
  → ServerApp.start()
      → WsServer: asyncio loop binds port P (OS-assigned)
      → RecognizerService.start() (inference daemon thread)
  → subprocess.Popen(["python", "src/client/client.py", "--server-url=ws://127.0.0.1:P"])
  → daemon watcher thread: client_proc.wait() → server_app.stop()

src/client/client.py
  → LoadingWindow displayed
  → websockets.connect(ws://127.0.0.1:P) [handshake]
  → reads session_created JSON frame
  → ClientApp.from_session_created(server_url, session_created_json, config)
  → transport._loop = asyncio_loop  (set after WsClientTransport._run_loop starts)
  → client_app.start() → app_state="running", AudioBridge starts, AudioSource starts
  → root.mainloop()
```

---

## Threading Model

### Server Process Threads

| Component | Thread Type | Purpose | Blocking Operations |
|---|---|---|---|
| **WsServer** | Daemon (asyncio event loop) | Accept WS connections; run session handler coroutines | `asyncio.sleep()`, WS I/O |
| **WsAudioReceiver** | Async coroutine (WsServer loop) | Decode binary frames → `chunk_queue` per session | `websocket.__aiter__()` |
| **WsResultSender._drain_loop** | Async task (WsServer loop, per session) | Drain bounded queue → `websocket.send()` | `asyncio.Queue.get()`, `websocket.send()` |
| **RecognizerService** | Daemon (one, shared) | Sequential ASR inference; route results by session_index | `input_queue.get(timeout=0.1)`, `recognize_window()` |
| **SoundPreProcessor** | Daemon (per session) | VAD, RMS normalization, speech buffering | `chunk_queue.get(timeout=0.1)`, `speech_queue.put()` |
| **GrowingWindowAssembler** | **No thread** | Grow recognition windows (sync call from SPP) | Called synchronously by SoundPreProcessor |
| **SpeechEndRouter** | Daemon (per session) | Assign message IDs; enforce in-flight ordering | `speech_queue.get(timeout=0.1)`, `recognizer_output_queue.get()` |
| **IncrementalTextMatcher** | Daemon (per session) | Overlap resolution; publish via WsResultSender | `matcher_queue.get(timeout=0.1)` |
| **SessionManager** | **No thread** | Creates/destroys ClientSession on connect/disconnect; observes ServerApplicationState | — |
| **ServerApplicationState** | **No thread** | Thread-safe state; 3 states | `threading.Lock` |

### Client Process Threads

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

### ServerApplicationState

**File:** `src/ServerApplicationState.py`

**Responsibility:** Server lifecycle management — no GUI, no `paused` state.

**State machine:** `starting → running → shutdown`

**Observer types:** Component observers only (called directly; no `root.after()` scheduling).

**Observed by:** `RecognizerService` (stops inference thread on shutdown), `SessionManager` (closes all sessions on shutdown), `SoundPreProcessor` and `SpeechEndRouter` per session (stop on shutdown).

### ClientApplicationState

**File:** `src/client/tk/ClientApplicationState.py`

**Responsibility:** Full client lifecycle including pause/resume.

**State machine:** `starting → running ⟷ paused → shutdown`

**Observer types:** Component observers (AudioSource, AudioBridge) and GUI observers (ControlPanel; scheduled via `root.after()`).

**Observed by:** `AudioSource` (closes stream on pause, reopens on resume), `WsClientTransport` (transitions to shutdown on connection loss), `ControlPanel` (updates button state).

---

## Data Flow with Types

### Pipeline Types (`src/types.py`)

```python
@dataclass
class AudioSegment:
    type: Literal['incremental']
    data: npt.NDArray[np.float32]       # Speech audio samples
    left_context: npt.NDArray[np.float32]
    right_context: npt.NDArray[np.float32]  # Always empty from SoundPreProcessor
    start_time: float
    end_time: float
    chunk_ids: list[int]
    message_id: int | None = None        # Set by SpeechEndRouter
    utterance_id: int = 0               # Set by SpeechEndRouter

@dataclass
class RecognitionResult:
    text: str
    start_time: float
    end_time: float
    chunk_ids: list[int]
    utterance_id: int = 0
    confidence: float = 0.0
    token_confidences: list[float]
    audio_rms: float = 0.0
    confidence_variance: float = 0.0
```

Also: `SpeechEndSignal`, `RecognitionTextMessage(result, message_id)`, `RecognizerAck(message_id, ok, error)`, `RecognizerFreeSignal`.

### Wire Protocol Types (`src/network/types.py`)

See [Network Protocol (v1)](#network-protocol-v1) above. Key types: `WsAudioFrame`, `WsControlCommand`, `WsSessionCreated`, `WsRecognitionResult`, `WsSessionClosed`, `WsError`.

### Server-Side Data Transformation

```
WsAudioReceiver (async)
  ← binary WsAudioFrame
  → dict {audio: ndarray, timestamp: float, chunk_id: int} → chunk_queue

SoundPreProcessor
  → RMS normalization + VAD → context buffering → speech buffering
  → GrowingWindowAssembler.process_segment() [sync]
      → AudioSegment (type='incremental', growing window) → speech_queue
  → SpeechEndSignal → speech_queue (utterance boundary)

SpeechEndRouter
  → AudioSegment: assigns message_id, puts on RecognizerService.input_queue
  → SpeechEndSignal: held until matching RecognizerAck received

RecognizerService
  → recognize_window(segment) → optional RecognitionResult
  → RecognitionTextMessage + RecognizerAck → session recognizer_output_queue

SpeechEndRouter (from recognizer_output_queue)
  → RecognitionResult → matcher_queue
  → SpeechEndSignal released → matcher_queue
  → RecognizerFreeSignal → control_queue (unblocks SoundPreProcessor)

IncrementalTextMatcher
  → overlap resolution, duplicate detection
  → WsResultSender.publish_partial_update(result)
  → WsResultSender.publish_finalization(result)

WsResultSender
  → encode as WsRecognitionResult JSON
  → asyncio.Queue(20) → websocket.send() → client
```

### Client-Side Data Transformation

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

### Hardware Acceleration Architecture

Uses **Strategy Pattern** for hardware-specific ONNX Runtime optimizations.

**Components:**
- **ExecutionProviderManager**: DXGI-based GPU detection (discrete > integrated priority)
- **SessionOptionsFactory**: Creates hardware-specific strategy
- **SessionOptionsStrategy**: Configures session options per hardware type

**Hardware Configurations:**
- **Integrated GPU** (UMA): Zero-copy shared memory, disable CPU arena
- **Discrete GPU** (PCIe): Staging buffer with CPU arena
- **CPU**: Multi-threaded execution (auto-detect cores)

**Configuration:** `recognition.inference = "auto" | "directml" | "cpu"` with DirectML GPU → CPU fallback

---

### ServerApp

**File:** `src/server/ServerApp.py`

**Responsibility:** Top-level server orchestrator. Creates `WsServer` and `RecognizerService`, manages server lifecycle, exposes `start()` / `stop()`.

---

### WsServer

**File:** `src/server/WsServer.py`

**Responsibility:** WebSocket server lifecycle. Runs `websockets.serve()` on a dedicated asyncio event loop daemon thread. On each new connection, delegates to `SessionManager.handle_connection()`.

---

### SessionManager

**File:** `src/server/SessionManager.py`

**Responsibility:** Session allocation and lifecycle management.

**Behavior:**
- Maintains a 1-based atomic `session_index` counter (protected by `threading.Lock`).
- On connect: creates `ClientSession`, calls `session.start()`, sends `WsSessionCreated` JSON frame.
- On disconnect or shutdown command: calls `session.close()`.
- Observes `ServerApplicationState`: on `shutdown` closes all active sessions.

---

### ClientSession

**File:** `src/server/ClientSession.py`

**Responsibility:** Owns and orchestrates all per-client pipeline components and queues.

**Queues (all `maxsize=100`):** `chunk_queue`, `speech_queue`, `recognizer_output_queue`, `matcher_queue`, `control_queue`.

**`start()` order:** register with `RecognizerService` → `WsResultSender.start()` → `SoundPreProcessor.start()` → `SpeechEndRouter.start()` → `IncrementalTextMatcher.start()`.

**`close()` order:** `SoundPreProcessor.stop()` → `SpeechEndRouter.stop()` + join → `IncrementalTextMatcher.stop()` + join → `WsResultSender.stop()` → unregister from `RecognizerService`.

---

### RecognizerService

**File:** `src/server/RecognizerService.py`

**Responsibility:** Single shared ASR inference thread; session-prefixed result routing.

**Thread loop:**
1. `input_queue.get(timeout=0.1)` — blocks until `AudioSegment` arrives.
2. `recognizer.recognize_window(segment)` — the long blocking inference call.
3. `session_index = segment.message_id // 10_000_000`
4. Routes `RecognitionTextMessage` (if result non-None) + `RecognizerAck` (always) to `session_output_queues[session_index].put_nowait()`. Drops and logs if session is no longer registered.

Observes `ServerApplicationState`; stops inference thread on `shutdown`.

---

### WsAudioReceiver

**File:** `src/server/WsAudioReceiver.py`

**Responsibility:** Async coroutine; bridges WebSocket binary frames to the synchronous `chunk_queue`.

**Behavior:**
- Iterates the WebSocket for binary frames; decodes each as `WsAudioFrame` using `decode_audio_frame()`.
- Puts `{audio, timestamp, chunk_id}` dict onto `ClientSession.chunk_queue` via `put_nowait()`.
- If `chunk_queue` is full: sends `WsError(BACKPRESSURE_DROP)` and continues (non-fatal).
- Text frames decoded as `WsControlCommand`; `"shutdown"` command triggers session close.
- `ConnectionClosed`: exits cleanly.

---

### WsResultSender

**File:** `src/server/WsResultSender.py`

**Responsibility:** Sync-to-async bridge; implements `RecognitionResultPublisher` protocol.

**Design:** Holds a bounded `asyncio.Queue(maxsize=20)`. A single async drain task (`_drain_loop`) calls `await websocket.send(encoded)`. Sync callers (`publish_partial_update`, `publish_finalization`) use `loop.call_soon_threadsafe(queue.put_nowait, encoded)`. If queue is full, the item is dropped and logged (backpressure; no priority discrimination).

---

### ClientApp

**File:** `src/client/tk/ClientApp.py`

**Responsibility:** Wires all client-side components; factory via `from_session_created()`.

**Construction order:** `ClientApplicationState` → `RecognitionResultFanOut` → `TextInsertionService` → `PauseController` → `ApplicationWindow` → subscribe `TextFormatter` to fan-out → `RemoteRecognitionPublisher` → `WsClientTransport` → `AudioSource`.

**`start()`:** Transitions state to `running`, starts `AudioBridge` daemon thread, starts `AudioSource`.

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

**Responsibility:** Concrete client-side implementation of the `RecognitionResultPublisher` protocol. Manages subscriber list and fan-out.

**Thread safety:** Copy-on-notify pattern (lock-protected list, callbacks outside lock). Subscriber exceptions are logged and isolated — one failing subscriber does not affect others.

**Subscribers:** `TextFormatter` (GUI display), `TextInsertionService` (keyboard insertion), `QuickEntrySubscriber` (quick-entry popup).

---

### RecognitionResultPublisher (Protocol)

**File:** `src/RecognitionResultPublisher.py`

**Responsibility:** Structural protocol defining the publisher interface. Any object with `publish_partial_update()` and `publish_finalization()` satisfies it.

```python
@runtime_checkable
class RecognitionResultPublisher(Protocol):
    def publish_partial_update(self, result: RecognitionResult) -> None: ...
    def publish_finalization(self, result: RecognitionResult) -> None: ...
```

**Implementations:**
- `WsResultSender` (server): encodes results as JSON and sends over WebSocket.
- `RecognitionResultFanOut` (client): fans out to local subscribers.

---

### AudioSource

**File:** `src/client/tk/sound/AudioSource.py`

**Responsibility:** Capture 32ms audio frames (16 kHz, 512 samples) from microphone. **Client-side only.**

**Output:** Raw audio dicts `{audio, timestamp, chunk_id}` → `chunk_queue`.

**Observer behavior:** On pause: close stream (release microphone). On resume: create fresh stream.

---

### SoundPreProcessor

**File:** `src/sound/SoundPreProcessor.py`

**Responsibility:** VAD processing, RMS normalization, context-aware speech buffering, windower orchestration. **Server-side only** — one instance per `ClientSession`.

**Input:** Audio dicts from `chunk_queue` (fed by `WsAudioReceiver`).
**Output:** `AudioSegment` + `SpeechEndSignal` → `speech_queue`.

**Key Features:**
- RMS normalization with temporal AGC before VAD.
- 20-chunk circular context buffer (640ms) for left context.
- 3-chunk consecutive speech confirmation prevents false positives.
- Intelligent breakpoint splitting on max duration reached.
- Observer behavior: on pause → flush pending segments.

---

### GrowingWindowAssembler

**File:** `src/sound/GrowingWindowAssembler.py`

**Responsibility:** Assemble speech segments into growing recognition windows. **Server-side only** — one instance per `ClientSession`. **No thread** — called synchronously by `SoundPreProcessor`.

**Strategy:** Each segment extends the current window (emits as `type='incremental'`). Windows grow up to `max_window_duration`; oldest segments dropped when exceeded. Flush emits remaining segments with right context.

---

### Recognizer

**File:** `src/asr/Recognizer.py`

**Responsibility:** Convert audio to text using hardware-accelerated STT model. In server mode, `RecognizerService` calls **only** `recognize_window()` — the Recognizer's own thread, input queue, and `app_state` observer are **not used**.

**Model:** Parakeet ONNX (`nemo-parakeet-tdt-0.6b-v3`, FP16) with `TimestampedResultsAsrAdapter`.

**Hardware Acceleration:** DirectML GPU (auto-select) → CPU fallback.

**Context-Aware Recognition:**
- Concatenates `left_context + data` for better acoustic modeling.
- Filters tokens by timestamp to exclude context hallucinations.
- 0.37s tolerance for VAD warm-up delay.

---

### IncrementalTextMatcher

**File:** `src/postprocessing/IncrementalTextMatcher.py`

**Responsibility:** Handle overlapping text from sliding windows; publish results via `RecognitionResultPublisher`. **Server-side only** — one per `ClientSession`. Output goes to `WsResultSender`.

**Key Features:**
- Overlap resolution using normalized text matching and longest common subsequence.
- Duplicate detection (0.6s time threshold, normalized text comparison).
- Routes by message type: `RecognitionResult` for overlap handling; `SpeechEndSignal` for boundary finalization.

---

### ApplicationState / PauseController / ControlPanel

**Files:**
- `src/ApplicationState.py` — base state class (used by `ClientApplicationState`)
- `src/client/tk/ClientApplicationState.py` — re-exports `ApplicationState`; full 4-state machine
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

## Design Principles

### 1. Single Responsibility Principle (SRP)

Each component has one clear responsibility:
- **AudioSource**: Mic capture only (non-blocking callback, client-side)
- **SoundPreProcessor**: VAD, normalization, speech buffering (server-side, per session)
- **GrowingWindowAssembler**: Growing window assembly (server-side, sync, per session)
- **RecognizerService**: Shared ASR inference + session routing (server-side, global)
- **IncrementalTextMatcher**: Text overlap resolution and result publishing (server-side, per session)
- **RecognitionResultPublisher**: Protocol interface for result delivery
- **WsResultSender**: Sync-to-async bridge to client WebSocket (server-side)
- **RecognitionResultFanOut**: Client-side subscriber management and fan-out
- **TextFormatter**: Formatting logic (client-side MVC controller, no GUI knowledge)
- **TextDisplayWidget**: GUI rendering + thread-safety (client-side MVC view)

### 2. Open/Closed Principle (OCP)

Extension without modification: add new `RecognitionResultPublisher` implementations (e.g., a logging adapter) or new `TextRecognitionSubscriber` instances without changing existing code.

### 3. Dependency Inversion Principle (DIP)

Components depend on abstractions: `RecognitionResultPublisher` Protocol (not concrete class), `TextRecognitionSubscriber` Protocol, windower interface. `IncrementalTextMatcher` accepts any object satisfying the protocol.

### 4. Type Safety

Strongly-typed dataclasses with discriminated unions enable static checking. Wire types in `src/network/types.py` are separate from internal pipeline types in `src/types.py`.

### 5. Hardware Abstraction

Strategy Pattern isolates ONNX Runtime hardware optimizations (`IntegratedGPUStrategy`, `DiscreteGPUStrategy`, `CPUStrategy`) from the recognition pipeline.

### 6. Observer Pattern for State Management

`ServerApplicationState` notifies component observers only (no GUI). `ClientApplicationState` notifies component observers (AudioSource) and GUI observers (ControlPanel, scheduled via `root.after()`). `PauseController` only updates state; components react autonomously. Thread-safe: mutations locked, notifications outside lock.

### 7. Observer Pattern → Protocol for Result Distribution

`RecognitionResultPublisher` is a structural `Protocol` (not a concrete class). `WsResultSender` implements it server-side (sends over WebSocket). `RecognitionResultFanOut` implements it client-side (fans out to local subscribers: TextFormatter, TextInsertionService, QuickEntryService).

### 8. MVC Pattern for Pause/Resume

**Model**: `ClientApplicationState` | **View**: `ControlPanel` | **Controller**: `PauseController` (state-only, no component manipulation)

Flow: `User → ControlPanel → PauseController → ClientApplicationState → Observers → AudioSource`

### 9. MVC Pattern for Text Display

**Model**: `RecognitionResult` | **Controller**: `TextFormatter` (no GUI knowledge) | **View**: `TextDisplayWidget` (thread-safe rendering)

Flow: `IncrementalTextMatcher → WsResultSender → WsClientTransport → RemoteRecognitionPublisher → RecognitionResultFanOut → TextFormatter → TextDisplayWidget → tkinter`

### 10. Process Isolation and Network Protocol

The server owns all ASR inference (GPU stays in the server process). The client owns all GUI and keyboard simulation. The boundary is a local WebSocket with a versioned binary + JSON protocol (`src/network/`). Isolation means future non-Tk clients (CLI, web) can connect without any server changes.

---

## Conclusion

### Key Innovations

**Context Buffer Strategy:**
- 20-chunk circular buffer (640ms) captures pre-speech audio.
- Left context snapshot immune to buffer wraparound.
- Enables better STT for short words (<100ms) without hallucinations.

**Hard-Cut Splitting:**
- Max duration triggers a hard cut: full buffer emitted, reset, stays in `ACTIVE_SPEECH`.
- No silence search; `right_context` is always empty from `SoundPreProcessor`.

**Timestamped Token Filtering:**
- Context concatenation: `left_context + data`.
- Token-level timestamp alignment from `TimestampedResultsAsrAdapter`.
- 0.37s tolerance for VAD warm-up delay.
- Removes context hallucinations from silence padding.

**Consecutive Speech Logic:**
- 3-chunk confirmation prevents false positives.
- Reduces transient noise triggering while preserving responsiveness.

**Observer-Based State Management:**
- `ServerApplicationState` (component-only) and `ClientApplicationState` (component + GUI).
- Lock-protected mutations; notifications outside lock to prevent deadlocks.
- Enables pause/resume without tight coupling between controller and components.

**Protocol-Based Result Distribution:**
- `RecognitionResultPublisher` is a structural Protocol — enables `WsResultSender` (server) and `RecognitionResultFanOut` (client) to satisfy the same interface.
- `RecognitionResultFanOut` uses copy-on-notify; subscriber exceptions are isolated.

**Session-Prefixed Message ID Routing:**
- `session_index * 10_000_000 + local_id` partitions the ID space.
- `RecognizerService` routes results by pure arithmetic (`message_id // 10_000_000`) — no routing table lock on the hot path.
- Capacity: ~57 hours of speech before rollover per session.

**Sync-to-Async Bridges:**
- Both `WsResultSender` (server → client) and `WsClientTransport` (client → server) use the same pattern: sync callers cross the thread boundary via `loop.call_soon_threadsafe(queue.put_nowait, item)`; an async drain task delivers the items without ever blocking the sync caller.
- Bounded queues (maxsize=20) provide natural backpressure with drop-and-log semantics.

**Two-Process Model:**
- Server handles GPU inference; client handles GUI and mic capture.
- Subprocess watcher in `main.py` ensures co-lifecycle: client exit stops server; server exit stops client via `ConnectionClosed`.
- `--server-only` mode enables headless/MSIX deployment with external WebSocket clients.
