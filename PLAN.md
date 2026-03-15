# Plan: WinUI Model Downloader (Replace Tk download_models.py)

## Context

`main.py` currently spawns `src/client/tk/download_models.py` (a Tkinter dialog) when the Parakeet
model is missing. Goal: replace that Tk subprocess with a standalone WinUI executable
`SttModelDownloader.exe` that does the same job — shows a dialog, downloads the model with a progress
bar, exits with the same code semantics as the Tk dialog. No protocol changes, no server changes, no
changes to the existing WinUI client (`SttClient.exe`).

---

## Exit Code Semantics (matching current Tk behavior)

`main.py` interprets the subprocess return code as:
- `0` → success or active-cancel (re-checks models; exits 1 if still missing)
- non-zero → cancelled before download started (exits 0 — user chose not to download)

Current Tk behavior (from `ModelDownloadDialog.py`):
- User cancels **before** clicking Download → `dialog.quit()` → natural exit → return code **1** via `sys.exit(1 if not success else 0)` in `download_models.py`
- User cancels **during** active download → `os._exit(0)` → return code **0** → main.py re-checks → exits 1 (model still missing)
- Download succeeds → exit code **0**
- Download fails (network error) → re-enables buttons for retry; only exits on explicit cancel or success

WinUI must preserve these exact semantics:
- Pre-download Exit → `Environment.Exit(1)`
- Mid-download cancel (window close or Exit while `Downloading`) → `Environment.Exit(0)`
- Success → `Environment.Exit(0)`
- Post-failure Exit (after error shown) → `Environment.Exit(1)` (download was attempted but failed, user gave up)

---

## New Project: `SttModelDownloader`

Add a third project to `src/client/winui/SttClient.sln`.

```
src/client/winui/
├── SttModelDownloader/               ← new WinUI app project
│   ├── SttModelDownloader.csproj
│   ├── App.xaml / App.xaml.cs        ← entry point, arg parsing, fast-exit if model present
│   ├── Download/
│   │   ├── IHttpMessageHandlerFactory.cs  ← seam for testable HttpClient injection
│   │   ├── ModelDownloadService.cs        ← C# HttpClient streaming download
│   │   └── DownloadViewModel.cs           ← INotifyPropertyChanged, state machine
│   └── Views/
│       └── DownloadWindow.xaml/.cs   ← WinUI dialog (model list, progress bar, license text, buttons)
└── SttModelDownloader.Tests/         ← xUnit test project (same pattern as SttClient.Tests)
```

`SttModelDownloader` targets `net10.0-windows10.0.19041.0` (same as `SttClient`).
`SttModelDownloader.Tests` targets `net10.0-windows` (no WinUI runtime needed for logic tests).

---

## Entry Point: `App.xaml.cs`

```
1. Parse --models-dir=<path> from Environment.GetCommandLineArgs()
2. If --models-dir missing or path invalid: show error dialog, Environment.Exit(2)
3. If model already present (encoder file exists & non-empty): Environment.Exit(0)
4. Create DownloadWindow with DownloadViewModel, show it, run message loop
```

Logging: Serilog rolling daily to `logs/sttdownloader-.log` (same pattern as `SttClient`).

---

## Download Logic (C#)

### Constants (mirror `ModelManager.py`)

```
BASE_URL  = "https://parakeet.grik.net/"
FILES     = ["config.json", "decoder_joint-model.fp16.onnx", "encoder-model.fp16.onnx",
             "nemo128.onnx", "vocab.txt"]
MODEL_DIR = <modelsDir>/parakeet/
```

### `IHttpMessageHandlerFactory` (testability seam)

```csharp
public interface IHttpMessageHandlerFactory
{
    HttpMessageHandler Create();
}
```

Production implementation returns `new HttpClientHandler { UseProxy = true }` (picks up system proxy).
Tests inject a `FakeHttpMessageHandler` that returns canned responses.

### `ModelDownloadService`

```csharp
public sealed class ModelDownloadService(IHttpMessageHandlerFactory handlerFactory)
{
    public bool IsModelMissing(string modelsDir);
    // checks <modelsDir>/parakeet/encoder-model.fp16.onnx exists & size > 0

    public Task DownloadAsync(
        string modelsDir,
        IProgress<DownloadProgressUpdate> progress,
        CancellationToken ct);
}

public sealed record DownloadProgressUpdate(
    long BytesDownloaded, long TotalBytes, double Percentage, string Status);
```

Download algorithm:
1. Pre-compute total expected bytes by issuing HEAD requests (or use Content-Length from first GET)
2. Download files **sequentially** — simpler progress aggregation
3. Write to `<filename>.partial` temp file; rename to final name only on successful file completion
4. On any failure or cancellation: delete all `.partial` files + any fully-written files from the
   current run (use a manifest of files written this session)
5. Classify exceptions for friendly error messages:
   - `HttpRequestException` with SSL inner → "SSL/certificate error — check corporate proxy settings"
   - `HttpRequestException` → "Network error: {message}"
   - `TaskCanceledException` (timeout) → "Download timed out — network too slow"
   - `OperationCanceledException` (user cancel) → rethrow (let ViewModel handle)
   - other → "Unexpected error: {type}: {message}"
6. Log all exceptions at Error level via Serilog before classifying

### `DownloadViewModel`

State enum: `AwaitingConfirmation → Downloading → Complete → Failed`
(Starts in `AwaitingConfirmation` — no `Idle` needed, the window only exists when model is missing.)

Properties (all INPC):
- `DownloadState State`
- `double ProgressPercentage` (0–100)
- `string StatusText`
- `string ErrorMessage` (set in `Failed` state)
- `bool CanDownload` (true only in `AwaitingConfirmation`)
- `bool CanRetry` (true only in `Failed`)
- `bool IsDownloading` (true only in `Downloading`) — drives Exit button behavior

Methods:
- `Task StartDownloadAsync(CancellationToken ct)` — transitions to `Downloading`, runs service
- `void NotifyWindowClosing()` — called by window's Closing handler; returns the exit code to use

Exit code logic in `NotifyWindowClosing()`:
```
Downloading  → Environment.Exit(0)   // mid-download cancel, matches Tk os._exit(0)
Failed       → Environment.Exit(1)   // tried and failed, user gave up
AwaitingConf → Environment.Exit(1)   // never started
Complete     → Environment.Exit(0)   // shouldn't reach here (window auto-closes)
```

`DownloadViewModel` has no WinUI dependency — tested entirely in `SttModelDownloader.Tests`.

---

## UI: `DownloadWindow.xaml`

Simple WinUI window (500×380, no resize, centered):

```
┌─────────────────────────────────────────────┐
│  Speech-to-Text — Required Models           │
│                                             │
│  The following model must be downloaded:    │
│                                             │
│  Parakeet STT Model (~1.5 GB)               │
│  ████████████████░░░░░░  67%                │
│  Downloading... 1.01 GB / 1.50 GB           │
│                                             │
│  [error message shown here on failure]      │
│                                             │
│  Parakeet-TDT-0.6B-v3 by NVIDIA (CC BY 4.0)│
│  Silero VAD by Silero Team (MIT)            │
│                                             │
│          [Download]        [Exit]           │
└─────────────────────────────────────────────┘
```

- `ProgressBar` bound to `ViewModel.ProgressPercentage`, `Visibility` tied to `IsDownloading`
- Status `TextBlock` bound to `ViewModel.StatusText`
- Error `TextBlock` bound to `ViewModel.ErrorMessage`, visible only in `Failed` state
- Download button: calls `ViewModel.StartDownloadAsync()`, visible when `CanDownload || CanRetry`
- Exit button: always visible; calls `ViewModel.NotifyWindowClosing()` then `this.Close()`
- Window `Closing` event → `ViewModel.NotifyWindowClosing()` (handles X button too)
- On `State == Complete`: window closes automatically → `Environment.Exit(0)`

---

## TDD Implementation Order

### 1. `ModelDownloadService` tests + implementation
`SttModelDownloader.Tests/Download/ModelDownloadServiceTests.cs`:
- `IsModelMissing_ReturnsFalse_WhenEncoderFileExistsAndNonEmpty`
- `IsModelMissing_ReturnsTrue_WhenEncoderFileMissing`
- `IsModelMissing_ReturnsTrue_WhenEncoderFileEmpty`
- `DownloadAsync_WritesToPartialFiles_DuringDownload` (fake handler: slow chunked response)
- `DownloadAsync_RenamesPartialToFinal_OnSuccess`
- `DownloadAsync_ReportsProgress_InAscendingOrder`
- `DownloadAsync_DeletesPartialAndCompletedFiles_OnHttpFailure`
- `DownloadAsync_DeletesPartialFiles_OnCancellation`
- `DownloadAsync_SslError_ThrowsWithFriendlyMessage`
- `DownloadAsync_Timeout_ThrowsWithFriendlyMessage`

### 2. `DownloadViewModel` tests + implementation
`SttModelDownloader.Tests/Download/DownloadViewModelTests.cs`:
- `InitialState_IsAwaitingConfirmation`
- `StartDownload_TransitionsToDownloading`
- `ProgressUpdate_UpdatesPercentageAndStatusText`
- `DownloadComplete_TransitionsToComplete`
- `DownloadFailed_TransitionsToFailed_SetsErrorMessage`
- `CanDownload_IsFalse_WhileDownloading`
- `CanRetry_IsTrue_InFailedState`
- `NotifyWindowClosing_WhenDownloading_ExitsWithZero`
- `NotifyWindowClosing_WhenFailed_ExitsWithOne`
- `NotifyWindowClosing_WhenAwaitingConfirmation_ExitsWithOne`

### 3. `DownloadWindow` UI + `App.xaml.cs`
- Implement `DownloadWindow.xaml/.cs` bound to `DownloadViewModel`
- Implement `App.xaml.cs`: arg parsing, model-present fast exit, missing-arg error dialog

### 4. Add projects to solution + build validation
- Add `SttModelDownloader.csproj` and `SttModelDownloader.Tests.csproj` to `SttClient.sln`
- Ensure `dotnet build SttClient.sln` succeeds

---

## Critical Files

| File | Change |
|------|--------|
| `src/client/winui/SttClient.sln` | Add `SttModelDownloader` + `SttModelDownloader.Tests` projects |
| `src/client/winui/SttModelDownloader/` | **New project** — entire directory |
| `src/client/winui/SttModelDownloader.Tests/` | **New test project** |
| `main.py` | **Out of scope** — handled in a separate task |

**Not changed:** `SttClient`, `SttClient.Core`, `SttClient.Tests`, server code, protocol, `client.py`, `main.py`.

---

## Verification

1. Build: `dotnet build SttClient.sln` from `src/client/winui/`
2. C# tests: `dotnet test SttClient.Tests/SttClient.Tests.csproj` + `dotnet test SttModelDownloader.Tests/SttModelDownloader.Tests.csproj`
3. Manual — success path: delete `~/.stt/models/parakeet/`, run `SttModelDownloader.exe --models-dir=<path>`, confirm dialog appears, click Download, observe progress, verify exit code 0
4. Manual — pre-download cancel: open dialog, click Exit immediately → verify exit code 1
5. Manual — mid-download cancel: start download, click Exit mid-way → verify exit code 0, verify partial files cleaned up
6. Manual — network error: point to bad URL (temporarily patch constants), verify friendly error shown and retry button appears
