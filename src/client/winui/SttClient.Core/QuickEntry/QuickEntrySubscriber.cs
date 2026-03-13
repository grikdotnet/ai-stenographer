using Microsoft.Extensions.Logging;
using SttClient.Recognition;

namespace SttClient.QuickEntry;

/// <summary>
/// <see cref="IRecognitionSubscriber"/> that accumulates partial and finalized recognition
/// text while the QuickEntry popup is active.
///
/// Responsibilities:
/// - Gates accumulation behind an Activate/Deactivate lifecycle.
/// - Fires a display-text callback (outside the lock) whenever the visible text changes.
/// - Thread-safe: OnPartialUpdate and OnFinalization may be called from the ReceiveLoop task.
/// </summary>
public sealed class QuickEntrySubscriber : IRecognitionSubscriber
{
    private Action<string> _onTextChange;
    private readonly ILogger<QuickEntrySubscriber> _logger;
    private readonly object _lock = new();

    private string _finalizedText = string.Empty;
    private string _partialText = string.Empty;
    private bool _active;

    /// <summary>
    /// Initializes a new <see cref="QuickEntrySubscriber"/>.
    /// </summary>
    /// <param name="onTextChange">
    /// Callback invoked with the current display text (finalized + partial) on every update.
    /// Called outside the internal lock.
    /// </param>
    /// <param name="logger">Logger for diagnostic output.</param>
    public QuickEntrySubscriber(Action<string> onTextChange, ILogger<QuickEntrySubscriber> logger)
    {
        _onTextChange = onTextChange;
        _logger = logger;
    }

    /// <summary>
    /// Overrides the text-change callback. Used by <see cref="QuickEntryController"/> to wire
    /// the popup display update after the subscriber has been constructed.
    /// </summary>
    /// <param name="callback">New callback; replaces the one set at construction.</param>
    public void SetTextChangeCallback(Action<string> callback)
    {
        _onTextChange = callback;
    }

    /// <summary>Activates accumulation and clears any previously accumulated text.</summary>
    public void Activate()
    {
        lock (_lock)
        {
            _active = true;
            _finalizedText = string.Empty;
            _partialText = string.Empty;
        }

        _logger.LogInformation("QuickEntrySubscriber: activated");
    }

    /// <summary>Deactivates accumulation. Incoming results are ignored until next Activate.</summary>
    public void Deactivate()
    {
        lock (_lock)
        {
            _active = false;
        }

        _logger.LogInformation("QuickEntrySubscriber: deactivated");
    }

    /// <summary>Returns the accumulated finalized text (without the current partial).</summary>
    public string GetAccumulatedText()
    {
        lock (_lock)
        {
            return _finalizedText;
        }
    }

    /// <inheritdoc/>
    public void OnPartialUpdate(RecognitionResult result)
    {
        string displayText;

        lock (_lock)
        {
            if (!_active)
                return;

            _partialText = result.Text;
            displayText = BuildDisplayText(_finalizedText, _partialText);
        }

        _onTextChange(displayText);
        _logger.LogDebug("QuickEntrySubscriber: partial '{Text}'", result.Text);
    }

    /// <inheritdoc/>
    public void OnFinalization(RecognitionResult result)
    {
        string displayText;

        lock (_lock)
        {
            if (!_active)
                return;

            _finalizedText = _finalizedText.Length > 0
                ? _finalizedText + " " + result.Text
                : result.Text;
            _partialText = string.Empty;
            displayText = _finalizedText;
        }

        _onTextChange(displayText);
        _logger.LogDebug("QuickEntrySubscriber: finalized '{Text}', total='{Total}'", result.Text, displayText);
    }

    private static string BuildDisplayText(string finalized, string partial) =>
        partial.Length > 0
            ? (finalized.Length > 0 ? finalized + " " + partial : partial)
            : finalized;
}
