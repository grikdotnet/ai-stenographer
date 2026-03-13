using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;

namespace SttClient.Insertion;

/// <summary>
/// Types text into the currently focused window using Win32 <c>SendInput</c>.
///
/// Responsibilities:
/// - Converts each character to a Unicode keyboard event pair (key-down + key-up).
/// - Does not require <c>uiAccess="true"</c> in the app manifest.
/// </summary>
public sealed class KeyboardSimulator : IKeyboardSimulator
{
    private readonly ILogger<KeyboardSimulator> _logger;

    [StructLayout(LayoutKind.Sequential)]
    private struct INPUT
    {
        public uint Type;
        public INPUTUNION Data;
    }

    [StructLayout(LayoutKind.Explicit)]
    private struct INPUTUNION
    {
        [FieldOffset(0)] public KEYBDINPUT Keyboard;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct KEYBDINPUT
    {
        public ushort VirtualKey;
        public ushort ScanCode;
        public uint Flags;
        public uint Time;
        public nint ExtraInfo;
    }

    private const uint INPUT_KEYBOARD = 1;
    private const uint KEYEVENTF_UNICODE = 0x0004;
    private const uint KEYEVENTF_KEYUP = 0x0002;

    [DllImport("user32.dll", SetLastError = true)]
    private static extern uint SendInput(uint nInputs, INPUT[] pInputs, int cbSize);

    /// <summary>
    /// Initializes a new <see cref="KeyboardSimulator"/>.
    /// </summary>
    /// <param name="logger">Logger for diagnostic output.</param>
    public KeyboardSimulator(ILogger<KeyboardSimulator> logger)
    {
        _logger = logger;
    }

    /// <inheritdoc/>
    public void TypeText(string text)
    {
        if (string.IsNullOrEmpty(text))
            return;

        var inputs = new INPUT[text.Length * 2];
        for (int i = 0; i < text.Length; i++)
        {
            inputs[i * 2] = MakeKeyEvent(text[i], keyUp: false);
            inputs[i * 2 + 1] = MakeKeyEvent(text[i], keyUp: true);
        }

        uint sent = SendInput((uint)inputs.Length, inputs, Marshal.SizeOf<INPUT>());
        if (sent != inputs.Length)
            _logger.LogWarning("KeyboardSimulator: SendInput sent {Sent}/{Total} events", sent, inputs.Length);
        else
            _logger.LogDebug("KeyboardSimulator: typed {Len} chars", text.Length);
    }

    private static INPUT MakeKeyEvent(char c, bool keyUp) => new()
    {
        Type = INPUT_KEYBOARD,
        Data = new INPUTUNION
        {
            Keyboard = new KEYBDINPUT
            {
                ScanCode = c,
                Flags = KEYEVENTF_UNICODE | (keyUp ? KEYEVENTF_KEYUP : 0),
            }
        }
    };
}
