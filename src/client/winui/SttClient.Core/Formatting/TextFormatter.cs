using Microsoft.Extensions.Logging;
using SttClient.Recognition;

namespace SttClient.Formatting;

/// <summary>
/// Transforms raw <see cref="RecognitionResult"/> events into <see cref="DisplayInstructions"/>
/// for the text display layer.
///
/// Responsibilities:
/// - Accumulates finalized text segments separated by spaces.
/// - Detects long pauses between utterances and inserts paragraph breaks.
/// - Suppresses duplicate finalization results.
/// - Tracks the current in-progress partial result for the preliminary overlay.
///
/// Implements <see cref="IRecognitionSubscriber"/> so it can be registered with
/// a <see cref="RecognitionResultFanOut"/> or wired directly.
/// </summary>
public sealed class TextFormatter : IRecognitionSubscriber
{
    // 2.0s gap chosen to match the Python reference implementation paragraph detection threshold
    private const double ParagraphPauseThreshold = 2.0;

    private readonly Action<DisplayInstructions> _onInstructions;
    private readonly ILogger<TextFormatter> _logger;

    private string _finalizedText = string.Empty;
    private string _lastFinalizedText = string.Empty;
    private double _lastFinalizedEndTime;
    private string? _currentPreliminaryText;

    /// <summary>
    /// Initializes a new <see cref="TextFormatter"/>.
    /// </summary>
    /// <param name="onInstructions">Delegate invoked each time display instructions are produced.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    public TextFormatter(Action<DisplayInstructions> onInstructions, ILogger<TextFormatter> logger)
    {
        _onInstructions = onInstructions;
        _logger = logger;
    }

    /// <summary>
    /// Handles a partial recognition update.
    ///
    /// Algorithm:
    /// 1. If a prior finalization exists and the gap from its end time to this result's start time
    ///    meets the paragraph threshold, append a newline to the finalized text.
    /// 2. Reset last-finalized tracking (the partial resets duplicate-suppression context).
    /// 3. Emit a RerenderAll instruction with current finalized text and the partial as overlay.
    /// </summary>
    /// <param name="result">The partial recognition result.</param>
    public void OnPartialUpdate(RecognitionResult result)
    {
        _currentPreliminaryText = result.Text;

        if (_lastFinalizedEndTime != 0
            && (result.StartTime - _lastFinalizedEndTime) >= ParagraphPauseThreshold
            && _lastFinalizedText != string.Empty
            && _finalizedText.Length > 0
            && !_finalizedText.EndsWith('\n'))
        {
            _finalizedText += "\n";
            _logger.LogDebug("Paragraph break inserted after {Gap:F2}s gap.", result.StartTime - _lastFinalizedEndTime);
        }

        _lastFinalizedText = string.Empty;

        _onInstructions(new DisplayInstructions(
            DisplayAction.RerenderAll,
            _finalizedText,
            _currentPreliminaryText is not null ? [_currentPreliminaryText] : []
        ));
    }

    /// <summary>
    /// Handles a final recognition result.
    ///
    /// Algorithm:
    /// 1. Reject duplicate: if the result text equals the last finalized text, do nothing.
    /// 2. Append the new text to the finalized buffer (space-separated).
    /// 3. Update tracking state and clear the preliminary overlay.
    /// 4. Emit a Finalize instruction.
    /// </summary>
    /// <param name="result">The finalized recognition result.</param>
    public void OnFinalization(RecognitionResult result)
    {
        if (result.Text == _lastFinalizedText)
        {
            _logger.LogDebug("Duplicate finalization ignored: \"{Text}\"", result.Text);
            return;
        }

        _finalizedText = _finalizedText.Length > 0
            ? _finalizedText + " " + result.Text
            : result.Text;

        _lastFinalizedText = result.Text;
        _lastFinalizedEndTime = result.EndTime;
        _currentPreliminaryText = null;

        _onInstructions(new DisplayInstructions(
            DisplayAction.Finalize,
            _finalizedText,
            []
        ));
    }
}
