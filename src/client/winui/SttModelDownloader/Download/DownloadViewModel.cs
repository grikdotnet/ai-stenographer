using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace SttModelDownloader.Download;

/// <summary>
/// ViewModel for the model download dialog.
/// Manages state transitions: AwaitingConfirmation → Downloading → Complete/Failed.
/// Implements INotifyPropertyChanged for WinUI data binding.
/// </summary>
public sealed class DownloadViewModel : INotifyPropertyChanged
{
    private readonly IModelDownloadService _service;
    private readonly string _modelsDir;
    private readonly IEnvironmentExit _environmentExit;

    private DownloadState _state = DownloadState.AwaitingConfirmation;
    private double _progressPercentage;
    private string _statusText = string.Empty;
    private string _errorMessage = string.Empty;

    /// <inheritdoc />
    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>
    /// Initializes a new DownloadViewModel.
    /// </summary>
    /// <param name="service">Service used to perform the download.</param>
    /// <param name="modelsDir">Root directory where model files will be placed.</param>
    /// <param name="environmentExit">Exit abstraction; defaults to real Environment.Exit.</param>
    public DownloadViewModel(
        IModelDownloadService service,
        string modelsDir,
        IEnvironmentExit? environmentExit = null)
    {
        _service = service;
        _modelsDir = modelsDir;
        _environmentExit = environmentExit ?? new EnvironmentExit();
    }

    /// <summary>Gets the current lifecycle state of the download.</summary>
    public DownloadState State
    {
        get => _state;
        private set => SetProperty(ref _state, value);
    }

    /// <summary>Gets the download completion percentage from 0 to 100.</summary>
    public double ProgressPercentage
    {
        get => _progressPercentage;
        private set => SetProperty(ref _progressPercentage, value);
    }

    /// <summary>Gets a human-readable status message describing the current operation.</summary>
    public string StatusText
    {
        get => _statusText;
        private set => SetProperty(ref _statusText, value);
    }

    /// <summary>Gets the error message when State is Failed; empty otherwise.</summary>
    public string ErrorMessage
    {
        get => _errorMessage;
        private set => SetProperty(ref _errorMessage, value);
    }

    /// <summary>Gets whether the download can be initiated (only true in AwaitingConfirmation).</summary>
    public bool CanDownload => State == DownloadState.AwaitingConfirmation;

    /// <summary>Gets whether a retry is available (only true in Failed state).</summary>
    public bool CanRetry => State == DownloadState.Failed;

    /// <summary>Gets whether a download is currently in progress.</summary>
    public bool IsDownloading => State == DownloadState.Downloading;

    /// <summary>
    /// Starts the model download. Transitions state to Downloading, then Complete or Failed.
    /// </summary>
    /// <param name="ct">CancellationToken for cancellation support.</param>
    public async Task StartDownloadAsync(CancellationToken ct)
    {
        State = DownloadState.Downloading;
        NotifyDerivedBoolProperties();

        var progress = new SynchronousProgress<DownloadProgressUpdate>(OnProgressUpdate);

        try
        {
            await _service.DownloadAsync(_modelsDir, progress, ct);
            State = DownloadState.Complete;
            StatusText = "Download complete";
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            State = DownloadState.AwaitingConfirmation;
            StatusText = string.Empty;
        }
        catch (Exception ex)
        {
            State = DownloadState.Failed;
            ErrorMessage = ex.Message;
        }
        finally
        {
            NotifyDerivedBoolProperties();
        }
    }

    /// <summary>
    /// Called by window's Closing event handler. Calls Environment.Exit with appropriate code.
    ///
    /// Algorithm:
    ///   Downloading          → Environment.Exit(0)  // mid-download cancel
    ///   Failed               → Environment.Exit(1)  // tried and failed
    ///   AwaitingConfirmation → Environment.Exit(1)  // never started
    ///   Complete             → Environment.Exit(0)  // shouldn't normally reach here
    /// </summary>
    public void NotifyWindowClosing()
    {
        int exitCode = State switch
        {
            DownloadState.Downloading => 0,
            DownloadState.Complete => 0,
            _ => 1
        };

        _environmentExit.Exit(exitCode);
    }

    private void OnProgressUpdate(DownloadProgressUpdate update)
    {
        ProgressPercentage = update.Percentage;
        StatusText = FormatStatusText(update);
    }

    private static string FormatStatusText(DownloadProgressUpdate update)
    {
        double downloadedMb = update.BytesDownloaded / 1_048_576.0;
        double totalMb = update.TotalBytes / 1_048_576.0;
        return $"Downloading... {downloadedMb:F1} MB / {totalMb:F1} MB";
    }

    private void NotifyDerivedBoolProperties()
    {
        OnPropertyChanged(nameof(CanDownload));
        OnPropertyChanged(nameof(CanRetry));
        OnPropertyChanged(nameof(IsDownloading));
    }

    private void SetProperty<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
            return;
        field = value;
        OnPropertyChanged(propertyName);
    }

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
}

/// <summary>
/// IProgress implementation that invokes the callback synchronously on the reporting thread,
/// bypassing SynchronizationContext scheduling. Used so property changes are immediately
/// visible to callers of StartDownloadAsync without requiring a delay or extra await.
/// </summary>
internal sealed class SynchronousProgress<T>(Action<T> callback) : IProgress<T>
{
    /// <inheritdoc />
    public void Report(T value) => callback(value);
}
