using Microsoft.UI.Xaml;
using Serilog;
using SttModelDownloader.Download;
using SttModelDownloader.Views;
using Windows.UI.Popups;
using WinRT.Interop;

namespace SttModelDownloader;

/// <summary>
/// WinUI application class. Parses command-line args, validates them, performs fast-exit
/// if the model is already present, then wires logging and shows the download window.
///
/// Algorithm:
/// 1. Parse --models-dir from Environment.GetCommandLineArgs().
/// 2. Validate the path; show error dialog and exit(2) if missing or blank.
/// 3. Check IsModelMissing; if model already present exit(0).
/// 4. Configure Serilog rolling-daily file sink.
/// 5. Build DownloadViewModel and show DownloadWindow.
/// </summary>
public partial class App : Application
{
    private DownloadWindow? _window;

    /// <summary>Initializes the application and XAML component tree.</summary>
    public App()
    {
        InitializeComponent();
    }

    /// <inheritdoc />
    protected override async void OnLaunched(LaunchActivatedEventArgs args)
    {
        var modelsDir = ParseModelsDir();

        if (string.IsNullOrWhiteSpace(modelsDir))
        {
            ConfigureSerilog();
            Log.Error("--models-dir argument is missing or empty. Exiting with code 2.");
            await ShowErrorDialogAsync("Missing argument: --models-dir\n\nUsage: SttModelDownloader.exe --models-dir=<path>");
            Environment.Exit(2);
            return;
        }

        var handlerFactory = new HttpClientHandlerFactory();
        var service = new ModelDownloadService(handlerFactory);

        if (!service.IsModelMissing(modelsDir))
        {
            Environment.Exit(0);
            return;
        }

        ConfigureSerilog();

        var viewModel = new DownloadViewModel(service, modelsDir);
        _window = new DownloadWindow(viewModel);
        _window.Activate();
    }

    private static string? ParseModelsDir()
    {
        foreach (var arg in Environment.GetCommandLineArgs())
        {
            if (arg.StartsWith("--models-dir=", StringComparison.OrdinalIgnoreCase))
                return arg["--models-dir=".Length..];
        }
        return null;
    }

    private static void ConfigureSerilog()
    {
        Directory.CreateDirectory("logs");
        Log.Logger = new LoggerConfiguration()
            .WriteTo.File("logs/sttdownloader-.log", rollingInterval: RollingInterval.Day)
            .CreateLogger();
    }

    /// <summary>
    /// Shows a modal error dialog using WinRT MessageDialog attached to a hidden owner window.
    /// </summary>
    /// <param name="message">The error message to display.</param>
    private static async Task ShowErrorDialogAsync(string message)
    {
        var ownerWindow = new Window();
        var hwnd = WindowNative.GetWindowHandle(ownerWindow);

        var dialog = new MessageDialog(message, "SttModelDownloader — Error");
        InitializeWithWindow.Initialize(dialog, hwnd);
        await dialog.ShowAsync();
    }
}
