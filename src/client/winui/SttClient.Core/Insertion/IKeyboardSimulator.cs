namespace SttClient.Insertion;

/// <summary>
/// Abstraction over keyboard text injection. Decouples <see cref="TextInserter"/>
/// from the Win32 SendInput P/Invoke so unit tests can inject a fake.
/// </summary>
public interface IKeyboardSimulator
{
    /// <summary>Types the given text into the currently focused window.</summary>
    /// <param name="text">The text to inject via simulated key events.</param>
    void TypeText(string text);
}
