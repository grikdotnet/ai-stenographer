# SttClient WinUI — Implementation Documentation

## Overview

SttClient is a Windows desktop application (WinUI 3 / .NET 10) that connects to a Speech-to-Text server over WebSocket, captures microphone audio, streams it to the server, and displays the transcription in real time. It also supports typing transcribed text into any focused window and a Quick Entry popup mode triggered by a global hotkey.

---

## Project Structure

```
src/client/winui/
├── SttClient.sln                    # Solution (SttClient + SttClient.Tests)
├── SttClient/                       # WinUI app (entry point, views, wiring)
│   ├── SttClient.csproj
│   ├── App.xaml / App.xaml.cs       # Application entry — delegates to AppStartup
│   ├── AppStartup.cs                # Component graph construction, startup
│   ├── app.manifest                 # Win32 manifest (asInvoker, Windows 10+)
│   ├── Insertion/
│   │   └── KeyboardSimulator.cs     # Win32 SendInput text injection
│   └── Views/
│       ├── MainWindow.xaml/.cs      # Transcription window
│       ├── LoadingWindow.xaml/.cs   # Startup / error splash
│       ├── LoadingPage.xaml/.cs     # Error display page
│       └── QuickEntryWindow.xaml/.cs# Quick Entry popup
│
├── SttClient.Core/                  # Platform-independent library (all logic)
│   ├── SttClient.Core.csproj
│   ├── App/
│   │   └── ClientOrchestrator.cs
│   ├── Audio/
│   │   ├── AudioChunk.cs
│   │   ├── IAudioSource.cs
│   │   ├── IWaveCapture.cs
│   │   ├── WasapiAudioSource.cs
│   │   └── WasapiCaptureAdapter.cs
│   ├── Formatting/
│   │   ├── DisplayInstructions.cs
│   │   └── TextFormatter.cs
│   ├── Insertion/
│   │   ├── IFocusTracker.cs
│   │   ├── IKeyboardSimulator.cs
│   │   ├── FocusTracker.cs
│   │   ├── TextInserter.cs
│   │   └── InsertionController.cs
│   ├── Protocol/
│   │   ├── WireTypes.cs
│   │   ├── AudioFrameEncoder.cs
│   │   └── ServerMessageDecoder.cs
│   ├── QuickEntry/
│   │   ├── IQuickEntryPopup.cs
│   │   ├── GlobalHotkeyListener.cs
│   │   ├── QuickEntryController.cs
│   │   └── QuickEntrySubscriber.cs
│   ├── Recognition/
│   │   ├── IRecognitionSubscriber.cs
│   │   ├── RecognitionResult.cs
│   │   ├── RecognitionResultFanOut.cs
│   │   └── RemoteRecognitionPublisher.cs
│   ├── State/
│   │   ├── AppState.cs
│   │   └── AppStateManager.cs
│   ├── Transport/
│   │   ├── IWebSocket.cs
│   │   └── WsClientTransport.cs
│   └── ViewModels/
│       ├── IDispatcherQueueAdapter.cs
│       ├── MainWindowViewModel.cs
│       └── QuickEntryViewModel.cs
│
└── SttClient.Tests/                 # xUnit test project
    ├── SttClient.Tests.csproj
    ├── Audio/
    ├── Formatting/
    ├── Insertion/
    ├── Integration/
    ├── Protocol/
    ├── QuickEntry/
    ├── Recognition/
    ├── State/
    ├── Transport/
    └── ViewModels/
```

`SttClient.Core` is not in `SttClient.sln` as a project entry but is referenced by `SttClient.csproj` and `SttClient.Tests.csproj` via `<ProjectReference>` — it builds automatically.

---

## Components

### AppState / AppStateManager

`AppState` is an enum with four values: `Starting → Running ↔ Paused → Shutdown`.

`AppStateManager` is a thread-safe state machine. Components register observers via `AddObserver(Action<AppState, AppState>)` and are notified synchronously (outside the internal lock) on every valid transition. Invalid transitions (e.g. `Shutdown → Running`) throw `InvalidOperationException`. `Shutdown → Shutdown` is a no-op.

---

### WasapiAudioSource

Wraps NAudio's WASAPI capture. On construction it checks whether the device reports 16 kHz mono IEEE float — if not, a `MediaFoundationResampler` is inserted. Audio is emitted in fixed 512-sample chunks via the `ChunkReady` event. Reacts to AppStateManager state changes: pauses capture on `Paused`, resumes on `Running`, stops permanently on `Shutdown`.

---

### AudioFrameEncoder / WsClientTransport

`AudioFrameEncoder` serialises a `WsAudioFrame` (session ID, chunk ID, timestamp, PCM samples) into the binary wire format:

```
[4 bytes LE uint32: header length] [header_length bytes UTF-8 JSON] [float32 PCM samples]
```

`WsClientTransport` owns the WebSocket connection after handshake. It runs two background tasks:

- **DrainLoop** — reads encoded frames from a bounded `Channel<byte[]>` (capacity 20, drops on full) and sends them as binary WebSocket frames.
- **ReceiveLoop** — reads text frames from the server and dispatches to `RemoteRecognitionPublisher`. On receiving a WebSocket Close frame, acknowledges and transitions state to `Shutdown`.

`StopAsync(serverInitiated)` sends a `control_command shutdown` JSON frame (client-initiated only), cancels the internal `CancellationTokenSource`, closes WebSocket output, then awaits both background tasks.

---

### ClientOrchestrator

The top-level coordinator. `ConnectAsync()` performs the full startup sequence:

1. Opens a `ClientWebSocket` to the server URL (10-second timeout).
2. Receives and validates the `session_created` handshake (protocol version must be `v1`).
3. Creates `WsClientTransport`, stores session ID, wires pong sender.
4. Calls `transport.StartAsync()` to launch DrainLoop + ReceiveLoop.
5. Subscribes to `ChunkReady` events from the audio source.
6. Transitions `AppStateManager` to `Running` (which causes `WasapiAudioSource` to start capture).

`StopAsync()` (client-initiated) unsubscribes from audio, calls `transport.StopAsync(serverInitiated: false)`, transitions state to `Shutdown`.

---

### RemoteRecognitionPublisher / RecognitionResultFanOut

`RemoteRecognitionPublisher` is called by `ReceiveLoop` with raw JSON strings. It routes by `type` field:

| `type` | Action |
|--------|--------|
| `recognition_result` (partial) | `fanOut.OnPartialUpdate()` |
| `recognition_result` (final) | `fanOut.OnFinalization()` |
| `session_closed` | `stateManager.SetState(Shutdown)` |
| `ping` | calls `PongSender` to send pong response |
| `error` | logs warning |
| unknown | ignored |

`RecognitionResultFanOut` multiplexes to all registered `IRecognitionSubscriber` implementations. Subscriber exceptions are caught and logged so one failing subscriber does not block others.

---

### TextFormatter

Implements `IRecognitionSubscriber`. Accumulates finalized results, suppresses exact duplicates, inserts paragraph breaks when the gap between utterances exceeds 2 seconds. Produces `DisplayInstructions` which are applied to `MainWindowViewModel` via a callback.

---

### TextInserter / InsertionController

`TextInserter` implements `IRecognitionSubscriber`. When enabled, types each finalized result into the currently focused window via `IKeyboardSimulator`. Ignores partial updates.

`InsertionController` exposes a `Toggle()` method used by the insertion button in the UI.

---

### QuickEntry subsystem

`GlobalHotkeyListener` registers a global hotkey via `RegisterHotKey` on a message-only HWND (`HWND_MESSAGE`). When the registered hotkey fires, it calls `QuickEntryController.OnHotkey()`.

`QuickEntryController` toggles the `QuickEntryWindow` popup. When open, `QuickEntrySubscriber` (an `IRecognitionSubscriber`) accumulates finalized text into `QuickEntryViewModel.LiveText`. On Submit: restores focus to the previously focused window, types the accumulated text; on Cancel: hides without typing.

---

### MainWindowViewModel

Implements `INotifyPropertyChanged`. Properties: `FinalizedText`, `PartialText`, `IsPaused`. All property updates are marshalled to the UI thread via `IDispatcherQueueAdapter`. Observes `AppStateManager` to track pause state.

---

### AppStartup

Constructs the entire component graph manually (no DI container). Order matters:

```
AppStateManager
RecognitionResultFanOut
RemoteRecognitionPublisher
MainWindowViewModel  ← observer registered with AppStateManager
TextFormatter        ← subscriber added to fanOut
KeyboardSimulator
TextInserter         ← subscriber added to fanOut
InsertionController
QuickEntrySubscriber ← subscriber added to fanOut
FocusTracker
QuickEntryController
GlobalHotkeyListener
WasapiAudioSource
AudioFrameEncoder
ClientOrchestrator
MainWindow           ← created, connected to ViewModel + Orchestrator
```

`AppStartup` shows `LoadingWindow`, calls `orchestrator.ConnectAsync()` asynchronously, and on success shows `MainWindow` and closes `LoadingWindow`. On failure, shows the error in `LoadingPage`.

---

## Data Flow

```
WasapiAudioSource  ──ChunkReady──►  ClientOrchestrator.OnChunkReady()
                                          │
                                          ▼
                               WsClientTransport._sendChannel
                                          │ DrainLoop
                                          ▼
                               WebSocket (binary frames) ──► STT Server
                                                                   │
                               WebSocket (text frames)  ◄──────────┘
                                          │ ReceiveLoop
                                          ▼
                               RemoteRecognitionPublisher.Dispatch()
                                          │
                                          ▼
                               RecognitionResultFanOut
                                  │           │           │
                                  ▼           ▼           ▼
                           TextFormatter  TextInserter  QuickEntrySubscriber
                                  │           │           │
                                  ▼           ▼           ▼
                           MainWindow    TypeText()   QuickEntryViewModel
                           ViewModel     (SendInput)   (popup LiveText)
```

---

## Prerequisites

- **Windows 10** version 1809 (build 17763) or later, **x64**
- **.NET 10 SDK** — [https://dotnet.microsoft.com/download](https://dotnet.microsoft.com/download)
- **Windows App SDK 1.7** runtime (installed automatically by NuGet restore on first build)

---

## Build Instructions

All commands are run from `src/client/winui/`.

### Restore dependencies

```bash
dotnet restore SttClient.sln
```

### Build (Debug)

```bash
dotnet build SttClient.sln
```

Debug output lands in `src/client/winui/build/` (configured via `Directory.Build.props`):
- App: `build/SttClient/`
- Core library: `build/SttClient.Core/`

Clean with: `rm -rf build/`

### Build (Release)

```bash
dotnet build SttClient.sln -c Release
```

Release output uses MSBuild defaults (required for MSIX Store packaging):
- App: `SttClient/bin/x64/Release/net10.0-windows10.0.19041.0/`
- Core library: `SttClient.Core/bin/Release/net10.0-windows/`

### Build single project

```bash
dotnet build SttClient.Core/SttClient.Core.csproj
dotnet build SttClient/SttClient.csproj
```

---

## Running Tests

All tests live in `SttClient.Tests/`. The test project references `SttClient.Core` only — no WinUI runtime needed.

### Run all tests

```bash
dotnet test SttClient.Tests/SttClient.Tests.csproj
```

### Run a specific test class

```bash
dotnet test SttClient.Tests/SttClient.Tests.csproj --filter "FullyQualifiedName~WsClientTransportTests"
dotnet test SttClient.Tests/SttClient.Tests.csproj --filter "FullyQualifiedName~ClientOrchestratorTests"
```

### Run a specific test method

```bash
dotnet test SttClient.Tests/SttClient.Tests.csproj --filter "FullyQualifiedName~Connect_ReceivesSessionCreated"
```

### Test parallelism note

`ClientOrchestratorTests` (integration tests using a real `HttpListener` + `ClientWebSocket`) is annotated with `[Collection(nameof(SequentialIntegration))]` to force sequential execution. These tests spin up an in-process fake WebSocket server on a random port and exercise the full orchestrator startup/shutdown cycle.

---

## Running the Application

The app requires a running STT server. The server URL is passed as a command-line argument.

### From the build output directory

```
SttClient.exe --server-url=ws://127.0.0.1:8765
```

### From Visual Studio

Set the project startup argument in project properties → Debug → Application Arguments:
```
--server-url=ws://127.0.0.1:8765
```

### Starting the STT server (Python backend)

From the repo root:

```bash
source venv/Scripts/activate
python main.py
```

This starts the server on `ws://127.0.0.1:8765` by default. The server spawns the WinUI client automatically if configured to do so.

### Logs

Application logs are written to `logs/sttclient-YYYYMMDD.log` relative to the working directory (daily rolling). Minimum level: Debug.

---

## Key Design Decisions

**Why `SttClient.Core` as a separate library?**
WinUI projects can only target `net10.0-windows10.0.19041.0`, which requires the Windows App SDK and cannot be used in test projects without the full MSIX runtime. All testable logic lives in `SttClient.Core` (targeting `net10.0-windows`), which has no WinUI dependency.

**Why manual DI in `AppStartup` instead of a container?**
The component graph is small and fixed. A DI container would add no value and obscure the wiring that drives startup order.

**Why bounded channel with drop-on-full for audio?**
Real-time audio cannot accumulate unbounded backpressure — a stalled network connection should drop frames, not grow memory indefinitely. The capacity-20 channel gives ~640ms of buffer at 32ms/frame before dropping.

**Why `[Collection(DisableParallelization = true)]` on integration tests?**
Background WebSocket tasks from one test's `ClientOrchestrator` can linger briefly after the test returns. When running in parallel these orphaned connections race to accept from the next test's `HttpListener`, stealing the connection and hanging the next test indefinitely. Sequential execution eliminates the race.
