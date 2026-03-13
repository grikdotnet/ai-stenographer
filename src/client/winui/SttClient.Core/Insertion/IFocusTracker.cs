namespace SttClient.Insertion;

/// <summary>
/// Saves and restores the foreground window handle so focus can be returned
/// to the user's target application after the QuickEntry popup closes.
/// </summary>
public interface IFocusTracker
{
    /// <summary>Captures the current foreground window handle for later restoration.</summary>
    void SaveFocus();

    /// <summary>
    /// Attempts to restore focus to the previously saved window.
    /// Best-effort: never throws even if focus restoration fails.
    /// </summary>
    void RestoreFocus();
}
