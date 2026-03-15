using Microsoft.Extensions.Logging;
using SttClient.Insertion;

namespace SttClient.QuickEntry;

/// <summary>
/// Orchestrates the QuickEntry popup lifecycle: show on hotkey, update live text,
/// submit (type text + restore focus), and cancel.
///
/// Responsibilities:
/// - Toggles popup visibility on each hotkey press.
/// - Delegates text accumulation to <see cref="QuickEntrySubscriber"/>.
/// - On submit: types accumulated text via <see cref="IKeyboardSimulator"/> and restores focus.
/// - On cancel: hides popup and restores focus without typing.
/// - Registers Enter/Escape as global hotkeys while the popup is visible so the user
///   can submit or cancel without the popup having keyboard focus.
/// - Thread-safe: <see cref="OnHotkey"/> may be called from the hotkey listener thread.
/// </summary>
public sealed class QuickEntryController
{
    private readonly QuickEntrySubscriber _subscriber;
    private readonly IFocusTracker _focusTracker;
    private readonly IKeyboardSimulator _keyboard;
    private readonly IQuickEntryPopup _popup;
    private readonly IPopupHotkeyRegistrar _hotkeyRegistrar;
    private readonly ILogger<QuickEntryController> _logger;
    private readonly object _lock = new();

    private bool _isShowing;

    /// <summary>
    /// Initializes a new <see cref="QuickEntryController"/>.
    /// </summary>
    /// <param name="subscriber">Accumulates live transcription text for the popup.</param>
    /// <param name="focusTracker">Saves and restores the target application's focus.</param>
    /// <param name="keyboard">Types submitted text into the restored window.</param>
    /// <param name="popup">The popup view (real WinUI window or test fake).</param>
    /// <param name="hotkeyRegistrar">Registers Enter/Escape as global hotkeys while popup is visible.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    public QuickEntryController(
        QuickEntrySubscriber subscriber,
        IFocusTracker focusTracker,
        IKeyboardSimulator keyboard,
        IQuickEntryPopup popup,
        IPopupHotkeyRegistrar hotkeyRegistrar,
        ILogger<QuickEntryController> logger)
    {
        _subscriber = subscriber;
        _focusTracker = focusTracker;
        _keyboard = keyboard;
        _popup = popup;
        _hotkeyRegistrar = hotkeyRegistrar;
        _logger = logger;

        // Wire the subscriber's text-change callback to update the popup display.
        _subscriber.SetTextChangeCallback(UpdatePopupText);
    }

    /// <summary>Gets whether the popup is currently visible.</summary>
    public bool IsShowing
    {
        get { lock (_lock) { return _isShowing; } }
    }

    /// <summary>
    /// Called by <see cref="GlobalHotkeyListener"/> when the hotkey fires.
    /// Toggles the popup open/closed.
    /// </summary>
    public void OnHotkey()
    {
        lock (_lock)
        {
            if (_isShowing)
                HideAndCancel();
            else
                ShowPopup();
        }
    }

    /// <summary>Submits the accumulated text: types it and hides the popup.</summary>
    public void Submit()
    {
        string text;

        lock (_lock)
        {
            text = _subscriber.GetAccumulatedText();
            ClosePopup();
        }

        if (text.Trim().Length > 0)
        {
            _logger.LogInformation("QuickEntryController: submitting '{Text}'", text);
            // Type while our process still owns the foreground — SendInput is blocked by
            // Windows if the calling process is not the foreground process. Restoring focus
            // to the target window first would surrender foreground and cause SendInput to
            // return 0. We type first, then hand foreground back.
            _keyboard.TypeText(text);
        }

        _focusTracker.RestoreFocus();
    }

    /// <summary>Cancels the QuickEntry session: hides the popup without typing.</summary>
    public void Cancel()
    {
        lock (_lock)
        {
            ClosePopup();
        }

        _focusTracker.RestoreFocus();
        _logger.LogInformation("QuickEntryController: popup hidden (cancelled)");
    }

    private void ShowPopup()
    {
        _focusTracker.SaveFocus();
        _subscriber.Activate();
        _popup.Show(onSubmit: Submit, onCancel: Cancel);
        _hotkeyRegistrar.RegisterPopupHotkeys(onSubmit: Submit, onCancel: Cancel);
        _isShowing = true;
        _logger.LogInformation("QuickEntryController: popup shown");
    }

    private void HideAndCancel()
    {
        ClosePopup();
        _focusTracker.RestoreFocus();
        _logger.LogInformation("QuickEntryController: popup hidden (hotkey toggle)");
    }

    private void ClosePopup()
    {
        _subscriber.Deactivate();
        _popup.Hide();
        _popup.SetText(string.Empty);
        _hotkeyRegistrar.UnregisterPopupHotkeys();
        _isShowing = false;
    }

    private void UpdatePopupText(string text)
    {
        lock (_lock)
        {
            if (_isShowing)
                _popup.SetText(text);
        }
    }
}
