using Microsoft.Extensions.Logging;
using SttClient.Recognition;

namespace SttClient.Insertion;

/// <summary>
/// <see cref="IRecognitionSubscriber"/> that types finalized recognition results into
/// the active window via <see cref="IKeyboardSimulator"/>.
///
/// Responsibilities:
/// - Gates insertion on an enabled/disabled toggle managed by <see cref="InsertionController"/>.
/// - Forwards only finalized (committed) text; partial updates are silently ignored.
/// </summary>
public sealed class TextInserter : IRecognitionSubscriber
{
    private readonly IKeyboardSimulator _keyboard;
    private readonly ILogger<TextInserter> _logger;
    private volatile bool _enabled;

    /// <summary>
    /// Initializes a new <see cref="TextInserter"/>.
    /// </summary>
    /// <param name="keyboard">Keyboard simulator used to inject text.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    public TextInserter(IKeyboardSimulator keyboard, ILogger<TextInserter> logger)
    {
        _keyboard = keyboard;
        _logger = logger;
    }

    /// <summary>
    /// Enables or disables text insertion.
    /// </summary>
    /// <param name="enabled">True to enable; false to disable.</param>
    public void SetEnabled(bool enabled)
    {
        _enabled = enabled;
        _logger.LogInformation("TextInserter: insertion {State}", enabled ? "enabled" : "disabled");
    }

    /// <inheritdoc/>
    public void OnPartialUpdate(RecognitionResult result) { }

    /// <summary>
    /// Types the finalized text into the active window if insertion is enabled.
    /// </summary>
    /// <param name="result">The finalized recognition result.</param>
    public void OnFinalization(RecognitionResult result)
    {
        if (!_enabled)
            return;

        _logger.LogDebug("TextInserter: typing {Len} chars", result.Text.Length);
        _keyboard.TypeText(result.Text);
    }
}
