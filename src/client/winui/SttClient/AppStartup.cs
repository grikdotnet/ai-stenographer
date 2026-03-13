using ILogger = Microsoft.Extensions.Logging.ILogger;
using Microsoft.Extensions.Logging;
using System.Runtime.InteropServices;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Serilog;
using SttClient.Audio;
using SttClient.Formatting;
using SttClient.Orchestration;
using SttClient.Protocol;
using SttClient.QuickEntry;
using SttClient.Recognition;
using SttClient.State;
using SttClient.ViewModels;
using SttClient.Views;
using SttClient.Insertion;

namespace SttClient;

/// <summary>
/// Performs all startup wiring: parses CLI args, constructs the component graph,
/// shows the loading window, and drives the connection handshake.
/// Separated from <see cref="App"/> so the XAML compiler only sees minimal code in App.xaml.cs.
/// </summary>
internal sealed class AppStartup
{
    private MainWindow? _mainWindow;
    private LoadingWindow? _loadingWindow;
    private ClientOrchestrator? _orchestrator;
    private ILoggerFactory? _loggerFactory;
    private GlobalHotkeyListener? _hotkeyListener;
    private DispatcherQueue? _dispatcherQueue;
    private MainWindowViewModel? _pendingViewModel;
    private InsertionController? _pendingInsertionController;
    private QuickEntryController? _pendingQuickEntryController;
    private QuickEntryViewModel? _pendingQuickEntryViewModel;
    private QuickEntrySubscriber? _pendingQuickEntrySubscriber;
    private FocusTracker? _pendingFocusTracker;
    private KeyboardSimulator? _pendingKeyboardSimulator;
    private readonly ManualResetEventSlim _staShutdown = new(false);
    private bool _startupSucceeded;

    /// <summary>Builds all components and initiates the server connection.</summary>
    public void Run()
    {
        ConfigureLogging();

        var logger = _loggerFactory!.CreateLogger<AppStartup>();

        if (!TryParseServerUrl(out var serverUrl, out var urlError))
        {
            ShowFatalError(urlError);
            return;
        }

        _dispatcherQueue = DispatcherQueue.GetForCurrentThread();
        var dispatcherAdapter = new DispatcherQueueAdapter(_dispatcherQueue);
        var loggerFactory = _loggerFactory!;

        var stateManager = new AppStateManager(loggerFactory.CreateLogger<AppStateManager>());
        var fanOut = new RecognitionResultFanOut(loggerFactory.CreateLogger<RecognitionResultFanOut>());
        var publisher = new RemoteRecognitionPublisher(fanOut, stateManager, loggerFactory.CreateLogger<RemoteRecognitionPublisher>());

        var viewModel = new MainWindowViewModel(dispatcherAdapter, loggerFactory.CreateLogger<MainWindowViewModel>());
        stateManager.AddObserver(viewModel.OnStateChanged);

        var formatter = new TextFormatter(viewModel.Apply, loggerFactory.CreateLogger<TextFormatter>());
        fanOut.AddSubscriber(formatter);

        var keyboardSimulator = new KeyboardSimulator(loggerFactory.CreateLogger<KeyboardSimulator>());
        var textInserter = new TextInserter(keyboardSimulator, loggerFactory.CreateLogger<TextInserter>());
        var insertionController = new InsertionController(textInserter, loggerFactory.CreateLogger<InsertionController>());
        fanOut.AddSubscriber(textInserter);

        var quickEntryViewModel = new QuickEntryViewModel(dispatcherAdapter, loggerFactory.CreateLogger<QuickEntryViewModel>());
        var quickEntrySubscriber = new QuickEntrySubscriber(quickEntryViewModel.SetLiveText, loggerFactory.CreateLogger<QuickEntrySubscriber>());
        fanOut.AddSubscriber(quickEntrySubscriber);

        var focusTracker = new FocusTracker(
            GetForegroundWindow,
            SetForegroundWindow,
            AttachThreadInput,
            loggerFactory.CreateLogger<FocusTracker>());

        _hotkeyListener = new GlobalHotkeyListener(
            () => _dispatcherQueue?.TryEnqueue(() => _pendingQuickEntryController?.OnHotkey()),
            loggerFactory.CreateLogger<GlobalHotkeyListener>());

        _pendingViewModel = viewModel;
        _pendingInsertionController = insertionController;
        _pendingQuickEntryViewModel = quickEntryViewModel;
        _pendingQuickEntrySubscriber = quickEntrySubscriber;
        _pendingFocusTracker = focusTracker;
        _pendingKeyboardSimulator = keyboardSimulator;

        stateManager.AddObserver((_, newState) =>
        {
            if (newState != AppState.Shutdown) return;
            _staShutdown.Set();
            _hotkeyListener?.Dispose();
            _dispatcherQueue?.TryEnqueue(() =>
            {
                _loggerFactory?.Dispose();
                Application.Current.Exit();
            });
        });

        logger.LogInformation("Run: constructing LoadingPage");
        var loadingPage = new LoadingPage();
        logger.LogInformation("Run: constructing LoadingWindow");
        _loadingWindow = new LoadingWindow(loadingPage);
        _loadingWindow.Closed += (_, _) => { if (!_startupSucceeded) Application.Current.Exit(); };
        logger.LogInformation("Run: activating LoadingWindow");
        _loadingWindow.Activate();
        logger.LogInformation("Run: LoadingWindow activated, starting hotkey listener");
        _hotkeyListener.Start();
        logger.LogInformation("Run: starting StartupAsync");
        _ = StartupAsync(loadingPage, logger, serverUrl!, stateManager, publisher);
    }

    private async Task StartupAsync(
        LoadingPage loadingPage,
        ILogger logger,
        string serverUrl,
        AppStateManager stateManager,
        RemoteRecognitionPublisher publisher)
    {
        try
        {
            logger.LogInformation("StartupAsync: creating audio on STA thread");
            var audioReady = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var staThread = new Thread(() =>
            {
                logger.LogInformation("STA thread: starting WasapiCaptureAdapter");
                try
                {
                    var capture = new WasapiCaptureAdapter();
                    logger.LogInformation("STA thread: WasapiCaptureAdapter done");
                    var audioSource = new WasapiAudioSource(capture, stateManager, _loggerFactory!.CreateLogger<WasapiAudioSource>());
                    logger.LogInformation("STA thread: WasapiAudioSource done");
                    var encoder = new AudioFrameEncoder(_loggerFactory!.CreateLogger<AudioFrameEncoder>());
                    _orchestrator = new ClientOrchestrator(
                        serverUrl: serverUrl,
                        stateManager: stateManager,
                        publisher: publisher,
                        encoder: encoder,
                        audioSource: audioSource,
                        loggerFactory: _loggerFactory!);
                    logger.LogInformation("STA thread: ClientOrchestrator done, signalling ready");
                    audioReady.SetResult();
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "STA thread: exception during audio init");
                    audioReady.SetException(ex);
                    return;
                }
                // Keep this STA alive so the WASAPI COM object remains on its original apartment.
                // WasapiCapture's callbacks require the owning STA thread to pump messages.
                _staShutdown.Wait();
                logger.LogInformation("STA thread: exiting after shutdown signal");
            });
            staThread.SetApartmentState(ApartmentState.STA);
            staThread.IsBackground = true;
            staThread.Start();
            logger.LogInformation("StartupAsync: waiting for audio ready");
            await audioReady.Task;
            logger.LogInformation("StartupAsync: audio ready, constructing windows on UI thread");

            var mainWindowReady = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            _dispatcherQueue!.TryEnqueue(() =>
            {
                logger.LogInformation("UI thread: constructing QuickEntryWindow");
                var quickEntryWindow = new QuickEntryWindow(_pendingQuickEntryViewModel!);
                logger.LogInformation("UI thread: constructing QuickEntryController");
                _pendingQuickEntryController = new QuickEntryController(
                    _pendingQuickEntrySubscriber!, _pendingFocusTracker!, _pendingKeyboardSimulator!,
                    quickEntryWindow, _loggerFactory!.CreateLogger<QuickEntryController>());
                logger.LogInformation("UI thread: constructing MainWindow");
                _mainWindow = new MainWindow(_pendingViewModel!, _orchestrator!, _pendingInsertionController!);
                logger.LogInformation("UI thread: windows constructed");
                mainWindowReady.SetResult();
            });
            await mainWindowReady.Task;
            logger.LogInformation("StartupAsync: calling ConnectAsync");

            await _orchestrator!.ConnectAsync();

            _startupSucceeded = true;
            _dispatcherQueue!.TryEnqueue(() =>
            {
                _mainWindow!.Activate();
                _loadingWindow!.Close();
            });
        }
        catch (OrchestratorStartupException ex)
        {
            logger.LogError(ex, "Startup failed");
            _loadingWindow!.DispatcherQueue.TryEnqueue(() =>
                loadingPage.ShowError(ex.Message));
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Unexpected startup error");
            _loadingWindow!.DispatcherQueue.TryEnqueue(() =>
                loadingPage.ShowError($"Unexpected error: {ex.Message}"));
        }
    }

    private void ConfigureLogging()
    {
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .WriteTo.File("logs/sttclient-.log", rollingInterval: Serilog.RollingInterval.Day)
            .CreateLogger();

        _loggerFactory = LoggerFactory.Create(builder =>
        {
            builder
                .AddDebug()
                .AddSerilog(dispose: true);
        });
    }

    private static bool TryParseServerUrl(out string? serverUrl, out string error)
    {
        serverUrl = null;
        error = string.Empty;

        foreach (var arg in Environment.GetCommandLineArgs())
        {
            if (arg.StartsWith("--server-url=", StringComparison.OrdinalIgnoreCase))
            {
                var url = arg["--server-url=".Length..];

                if (url.StartsWith("wss://", StringComparison.OrdinalIgnoreCase))
                {
                    error = "TLS (wss://) is not yet supported in protocol v1.";
                    return false;
                }

                if (!url.StartsWith("ws://", StringComparison.OrdinalIgnoreCase))
                {
                    error = $"Invalid server URL scheme. Expected ws://, got: {url}";
                    return false;
                }

                serverUrl = url;
                return true;
            }
        }

        error = "Missing required argument: --server-url=ws://host:port";
        return false;
    }

    private static void ShowFatalError(string message)
    {
        var errorPage = new LoadingPage();
        var win = new LoadingWindow(errorPage);
        win.Closed += (_, _) => Application.Current.Exit();
        win.Activate();
        errorPage.ShowError(message);
    }

    [DllImport("user32.dll")] private static extern nint GetForegroundWindow();
    [DllImport("user32.dll")] private static extern bool SetForegroundWindow(nint hWnd);
    [DllImport("user32.dll")] private static extern bool AttachThreadInput(uint idAttach, uint idAttachTo, bool fAttach);
}

/// <summary>
/// Adapts WinUI <see cref="DispatcherQueue"/> to <see cref="IDispatcherQueueAdapter"/>.
/// </summary>
internal sealed class DispatcherQueueAdapter : IDispatcherQueueAdapter
{
    private readonly DispatcherQueue _queue;

    /// <summary>Initializes the adapter with the UI-thread dispatcher queue.</summary>
    public DispatcherQueueAdapter(DispatcherQueue queue) => _queue = queue;

    /// <inheritdoc/>
    public bool TryEnqueue(Action action) => _queue.TryEnqueue(() => action());
}
