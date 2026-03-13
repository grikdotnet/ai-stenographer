namespace SttClient.QuickEntry;

/// <summary>
/// Abstraction over the QuickEntry popup window. Decouples <see cref="QuickEntryController"/>
/// from the WinUI <c>QuickEntryWindow</c> so unit tests can inject a fake.
/// </summary>
public interface IQuickEntryPopup
{
    /// <summary>Makes the popup visible and registers submit/cancel callbacks.</summary>
    /// <param name="onSubmit">Invoked when the user presses Enter to submit.</param>
    /// <param name="onCancel">Invoked when the user presses Escape to cancel.</param>
    void Show(Action onSubmit, Action onCancel);

    /// <summary>Hides the popup without triggering submit or cancel.</summary>
    void Hide();

    /// <summary>Updates the live transcription text shown inside the popup.</summary>
    /// <param name="text">Current accumulated text to display.</param>
    void SetText(string text);
}
