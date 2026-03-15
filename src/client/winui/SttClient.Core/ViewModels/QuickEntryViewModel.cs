using Microsoft.Extensions.Logging;

namespace SttClient.ViewModels;

/// <summary>
/// ViewModel for the QuickEntry popup window.
///
/// Responsibilities:
/// - Exposes <see cref="LiveText"/> as the accumulated partial+final transcript shown in the popup.
/// - Provides <see cref="SetLiveText"/> for the QuickEntrySubscriber to push updates.
/// - Implements INotifyPropertyChanged for WinUI data binding.
/// </summary>
public sealed class QuickEntryViewModel : ViewModelBase
{
    private readonly IDispatcherQueueAdapter _dispatcher;
    private readonly ILogger<QuickEntryViewModel> _logger;

    private string _liveText = string.Empty;

    /// <summary>
    /// Initializes a new <see cref="QuickEntryViewModel"/>.
    /// </summary>
    /// <param name="dispatcher">Adapter for marshalling UI updates to the UI thread.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    public QuickEntryViewModel(IDispatcherQueueAdapter dispatcher, ILogger<QuickEntryViewModel> logger)
    {
        _dispatcher = dispatcher;
        _logger = logger;
    }

    /// <summary>Gets the live transcription text shown in the popup.</summary>
    public string LiveText
    {
        get => _liveText;
        private set => SetProperty(ref _liveText, value);
    }

    /// <summary>
    /// Updates the live text shown in the QuickEntry popup.
    /// Marshals the update to the UI thread via the dispatcher.
    /// </summary>
    /// <param name="text">The current accumulated transcription text.</param>
    public void SetLiveText(string text)
    {
        _dispatcher.TryEnqueue(() =>
        {
            LiveText = text;
            _logger.LogDebug("QuickEntry live text updated: length={Len}", text.Length);
        });
    }

    /// <summary>Clears the live text (called on cancel or after submit).</summary>
    public void Clear()
    {
        _dispatcher.TryEnqueue(() => LiveText = string.Empty);
    }

}
