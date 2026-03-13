using Microsoft.Extensions.Logging;

namespace SttClient.Insertion;

/// <summary>
/// Saves the foreground window handle before the QuickEntry popup steals focus,
/// then restores it when the popup closes.
///
/// Responsibilities:
/// - Calls injectable Win32 delegates (GetForegroundWindow / SetForegroundWindow /
///   AttachThreadInput) so unit tests can substitute fakes without P/Invoke.
/// - Implements best-effort restore: if SetForegroundWindow fails, falls back to
///   AttachThreadInput + SetForegroundWindow. Never throws on failure.
/// </summary>
public sealed class FocusTracker : IFocusTracker
{
    private readonly Func<nint> _getForegroundWindow;
    private readonly Func<nint, bool> _setForegroundWindow;
    private readonly Func<uint, uint, bool, bool> _attachThreadInput;
    private readonly ILogger<FocusTracker> _logger;

    private nint _savedHwnd;

    /// <summary>
    /// Initializes a new <see cref="FocusTracker"/> with injectable Win32 delegates.
    /// </summary>
    /// <param name="getForegroundWindow">Returns the handle of the current foreground window.</param>
    /// <param name="setForegroundWindow">Attempts to bring the given window to the foreground.</param>
    /// <param name="attachThreadInput">Attaches/detaches input processing of two threads.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    public FocusTracker(
        Func<nint> getForegroundWindow,
        Func<nint, bool> setForegroundWindow,
        Func<uint, uint, bool, bool> attachThreadInput,
        ILogger<FocusTracker> logger)
    {
        _getForegroundWindow = getForegroundWindow;
        _setForegroundWindow = setForegroundWindow;
        _attachThreadInput = attachThreadInput;
        _logger = logger;
    }

    /// <inheritdoc/>
    public void SaveFocus()
    {
        _savedHwnd = _getForegroundWindow();
        _logger.LogDebug("FocusTracker: saved hwnd=0x{Hwnd:X}", _savedHwnd);
    }

    /// <inheritdoc/>
    public void RestoreFocus()
    {
        if (_savedHwnd == default)
            return;

        if (_setForegroundWindow(_savedHwnd))
        {
            _logger.LogDebug("FocusTracker: restored hwnd=0x{Hwnd:X}", _savedHwnd);
            return;
        }

        // Windows focus rules can block SetForegroundWindow; attach thread input as fallback.
        _logger.LogDebug("FocusTracker: SetForegroundWindow failed, trying AttachThreadInput fallback");
        _attachThreadInput(0, 0, true);
        bool result = _setForegroundWindow(_savedHwnd);
        _attachThreadInput(0, 0, false);

        if (!result)
            _logger.LogWarning("FocusTracker: failed to restore focus to hwnd=0x{Hwnd:X}", _savedHwnd);
    }
}
