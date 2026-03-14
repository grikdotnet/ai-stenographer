namespace SttClient.QuickEntry;

/// <summary>
/// Registers and unregisters transient global hotkeys (Enter/Escape) while the
/// QuickEntry popup is visible, so the user can submit or cancel without the
/// popup having keyboard focus.
/// </summary>
public interface IPopupHotkeyRegistrar
{
    /// <summary>
    /// Registers Enter and Escape as global hotkeys.
    /// </summary>
    /// <param name="onSubmit">Invoked when Enter is pressed.</param>
    /// <param name="onCancel">Invoked when Escape is pressed.</param>
    void RegisterPopupHotkeys(Action onSubmit, Action onCancel);

    /// <summary>Unregisters the Enter and Escape global hotkeys.</summary>
    void UnregisterPopupHotkeys();
}
