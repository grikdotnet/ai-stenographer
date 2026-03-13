using Microsoft.Extensions.Logging;

namespace SttClient.Insertion;

/// <summary>
/// Controls the enabled/disabled state of <see cref="TextInserter"/>.
/// Exposes a toggle method for the UI insertion button.
/// </summary>
public sealed class InsertionController
{
    private readonly TextInserter _inserter;
    private readonly ILogger<InsertionController> _logger;
    private bool _enabled;

    /// <summary>Gets whether insertion is currently enabled.</summary>
    public bool IsEnabled => _enabled;

    /// <summary>
    /// Initializes a new <see cref="InsertionController"/>.
    /// </summary>
    /// <param name="inserter">The text inserter to control.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    public InsertionController(TextInserter inserter, ILogger<InsertionController> logger)
    {
        _inserter = inserter;
        _logger = logger;
    }

    /// <summary>Toggles insertion on/off and propagates the new state to the inserter.</summary>
    public void Toggle()
    {
        _enabled = !_enabled;
        _inserter.SetEnabled(_enabled);
        _logger.LogInformation("InsertionController: toggled to {State}", _enabled);
    }

    /// <summary>Explicitly sets the enabled state.</summary>
    /// <param name="enabled">Desired enabled state.</param>
    public void SetEnabled(bool enabled)
    {
        _enabled = enabled;
        _inserter.SetEnabled(enabled);
    }
}
