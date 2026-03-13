using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;

namespace SttClient.QuickEntry;

/// <summary>
/// Registers a global system hotkey via Win32 <c>RegisterHotKey</c> on a message-only HWND
/// and fires a callback whenever the hotkey is pressed by the user.
///
/// Responsibilities:
/// - Runs a Win32 message loop on a dedicated background thread.
/// - Exposes <see cref="SimulateHotkeyForTest"/> so unit tests can trigger the callback
///   without a real Win32 environment.
/// - Swallows callback exceptions to keep the message loop alive.
/// </summary>
public sealed class GlobalHotkeyListener : IDisposable
{
    // Ctrl+Space: MOD_CONTROL = 0x0002
    private const int HotkeyId = 9001;
    private const uint ModControl = 0x0002;
    private const uint VkSpace = 0x20;
    private const uint WmHotkey = 0x0312;
    private const uint WmQuit = 0x0012;

    private readonly Action _callback;
    private readonly ILogger<GlobalHotkeyListener> _logger;

    private long _hwnd;
    private Thread? _thread;

    [DllImport("user32.dll", SetLastError = true)]
    private static extern nint CreateWindowExW(
        uint dwExStyle, string lpClassName, string lpWindowName,
        uint dwStyle, int x, int y, int nWidth, int nHeight,
        nint hWndParent, nint hMenu, nint hInstance, nint lpParam);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool DestroyWindow(nint hWnd);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool RegisterHotKey(nint hWnd, int id, uint fsModifiers, uint vk);

    [DllImport("user32.dll")]
    private static extern bool UnregisterHotKey(nint hWnd, int id);

    [DllImport("user32.dll")]
    private static extern bool GetMessage(out NativeMsg lpMsg, nint hWnd, uint wMsgFilterMin, uint wMsgFilterMax);

    [DllImport("user32.dll")]
    private static extern bool TranslateMessage(ref NativeMsg lpMsg);

    [DllImport("user32.dll")]
    private static extern nint DispatchMessage(ref NativeMsg lpmsg);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool PostMessage(nint hWnd, uint msg, nint wParam, nint lParam);

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeMsg
    {
        public nint hwnd;
        public uint message;
        public nint wParam;
        public nint lParam;
        public uint time;
        public int ptX;
        public int ptY;
    }

    // HWND_MESSAGE sentinel for message-only windows
    private static readonly nint HwndMessage = new(-3);

    /// <summary>
    /// Initializes a new <see cref="GlobalHotkeyListener"/>.
    /// </summary>
    /// <param name="callback">Action invoked on each hotkey press.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    public GlobalHotkeyListener(Action callback, ILogger<GlobalHotkeyListener> logger)
    {
        _callback = callback;
        _logger = logger;
    }

    /// <summary>
    /// Triggers the hotkey callback directly — for unit tests only.
    /// Swallows any exception thrown by the callback and logs it as a warning.
    /// </summary>
    public void SimulateHotkeyForTest()
    {
        InvokeCallback();
    }

    /// <summary>Starts the Win32 message loop on a dedicated thread (production use).</summary>
    public void Start()
    {
        _logger.LogInformation("GlobalHotkeyListener: starting message loop thread");
        _thread = new Thread(RunMessageLoop) { IsBackground = true, Name = "HotkeyMessageLoop" };
        _thread.Start();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        var hwnd = (nint)Interlocked.Exchange(ref _hwnd, 0);
        if (hwnd != 0)
        {
            UnregisterHotKey(hwnd, HotkeyId);
            PostMessage(hwnd, WmQuit, 0, 0);
        }
    }

    private void RunMessageLoop()
    {
        // Message-only window: receives WM_HOTKEY without appearing on screen or taskbar.
        _hwnd = (long)CreateWindowExW(0, "STATIC", string.Empty, 0, 0, 0, 0, 0, HwndMessage, 0, 0, 0);

        if (_hwnd == 0)
        {
            _logger.LogError("GlobalHotkeyListener: CreateWindowEx failed (error {Error})", Marshal.GetLastWin32Error());
            return;
        }

        if (!RegisterHotKey((nint)_hwnd, HotkeyId, ModControl, VkSpace))
        {
            _logger.LogWarning("GlobalHotkeyListener: RegisterHotKey failed (error {Error}) — hotkey unavailable", Marshal.GetLastWin32Error());
        }
        else
        {
            _logger.LogInformation("GlobalHotkeyListener: Ctrl+Space registered (id={Id})", HotkeyId);
        }

        while (GetMessage(out var msg, (nint)_hwnd, 0, 0))
        {
            if (msg.message == WmHotkey && msg.wParam == HotkeyId)
                InvokeCallback();

            TranslateMessage(ref msg);
            DispatchMessage(ref msg);
        }

        var hwnd = (nint)Interlocked.Exchange(ref _hwnd, 0);
        UnregisterHotKey(hwnd, HotkeyId);
        DestroyWindow(hwnd);
        _logger.LogDebug("GlobalHotkeyListener: message loop exited");
    }

    private void InvokeCallback()
    {
        try
        {
            _callback();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "GlobalHotkeyListener: callback threw an exception");
        }
    }
}
