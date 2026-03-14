using System.ComponentModel;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging;
using SttClient.Formatting;
using SttClient.State;

namespace SttClient.ViewModels;

/// <summary>
/// ViewModel for the main transcription window.
///
/// Responsibilities:
/// - Exposes <see cref="FinalizedText"/>, <see cref="PartialText"/>, and <see cref="IsPaused"/> as bindable properties.
/// - Receives <see cref="DisplayInstructions"/> from <see cref="TextFormatter"/> and applies them on the UI thread.
/// - Observes <see cref="AppStateManager"/> to reflect pause/resume state changes.
/// - Implements INotifyPropertyChanged for WinUI data binding.
/// </summary>
public sealed class MainWindowViewModel : INotifyPropertyChanged
{
    private readonly IDispatcherQueueAdapter _dispatcher;
    private readonly ILogger<MainWindowViewModel> _logger;

    private string _finalizedText = string.Empty;
    private string _partialText = string.Empty;
    private bool _isPaused;
    private bool _isRunningOrPaused;

    /// <inheritdoc/>
    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>
    /// Initializes a new <see cref="MainWindowViewModel"/>.
    /// </summary>
    /// <param name="dispatcher">Adapter for marshalling UI updates to the UI thread.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    public MainWindowViewModel(IDispatcherQueueAdapter dispatcher, ILogger<MainWindowViewModel> logger)
    {
        _dispatcher = dispatcher;
        _logger = logger;
    }

    /// <summary>Gets the accumulated finalized transcript text.</summary>
    public string FinalizedText
    {
        get => _finalizedText;
        private set => SetProperty(ref _finalizedText, value);
    }

    /// <summary>Gets the current in-flight partial recognition text shown as a transient overlay.</summary>
    public string PartialText
    {
        get => _partialText;
        private set => SetProperty(ref _partialText, value);
    }

    /// <summary>Gets whether the session is currently paused.</summary>
    public bool IsPaused
    {
        get => _isPaused;
        private set => SetProperty(ref _isPaused, value);
    }

    /// <summary>Gets whether the session is in a state where pause/resume is meaningful (Running or Paused).</summary>
    public bool IsRunningOrPaused
    {
        get => _isRunningOrPaused;
        private set => SetProperty(ref _isRunningOrPaused, value);
    }

    /// <summary>
    /// Applies display instructions produced by <see cref="TextFormatter"/>.
    /// Marshals the update to the UI thread via the dispatcher.
    ///
    /// Algorithm:
    /// 1. Both actions set FinalizedText from instructions.
    /// 2. RerenderAll: sets PartialText from PreliminarySegments (joined by space).
    /// 3. Finalize: clears PartialText.
    /// </summary>
    /// <param name="instructions">Instructions describing the display update to perform.</param>
    public void Apply(DisplayInstructions instructions)
    {
        _dispatcher.TryEnqueue(() =>
        {
            FinalizedText = instructions.FinalizedText;

            PartialText = instructions.Action == DisplayAction.RerenderAll
                ? string.Join(" ", instructions.PreliminarySegments)
                : string.Empty;

            _logger.LogDebug(
                "Display updated: Action={Action} FinalizedLen={Len} PartialLen={PLen}",
                instructions.Action, FinalizedText.Length, PartialText.Length);
        });
    }

    /// <summary>
    /// Observer callback for <see cref="AppStateManager"/> state transitions.
    /// Updates <see cref="IsPaused"/> on the UI thread.
    /// </summary>
    /// <param name="oldState">Previous application state (unused).</param>
    /// <param name="newState">New application state.</param>
    public void OnStateChanged(AppState oldState, AppState newState)
    {
        _dispatcher.TryEnqueue(() =>
        {
            IsPaused = newState == AppState.Paused;
            IsRunningOrPaused = newState == AppState.Running || newState == AppState.Paused;
        });
    }

    private void SetProperty<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
            return;

        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
