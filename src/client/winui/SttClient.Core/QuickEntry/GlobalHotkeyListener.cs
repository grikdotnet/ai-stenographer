using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;

namespace SttClient.QuickEntry;

/// <summary>
/// Registers a global system hotkey via Win32 <c>RegisterHotKey(hWnd=0)</c> on a dedicated
/// background thread and fires a callback whenever the hotkey is pressed by the user.
///
/// Responsibilities:
/// - Runs a Win32 message loop on a dedicated background thread.
/// - Uses a null HWND with RegisterHotKey so WM_HOTKEY is posted to the thread queue directly,
///   avoiding the need to create a message-only window.
/// - Supports registering transient popup hotkeys (Enter/Escape) while the QuickEntry popup
///   is visible, so the popup can be submitted or cancelled without keyboard focus.
/// - Exposes <see cref="SimulateHotkeyForTest"/> so unit tests can trigger the callback
///   without a real Win32 environment.
/// - Swallows callback exceptions to keep the message loop alive.
/// </summary>
public sealed class GlobalHotkeyListener : IPopupHotkeyRegistrar, IDisposable
{
    // Ctrl+Space: MOD_CONTROL = 0x0002
    private const int HotkeyId = 9001;
    private const int PopupSubmitHotkeyId = 9002;
    private const int PopupCancelHotkeyId = 9003;
    private const uint ModNone = 0x4000; // MOD_NOREPEAT, no modifier
    private const uint ModControl = 0x0002;
    private const uint VkSpace = 0x20;
    private const uint VkReturn = 0x0D;
    private const uint VkEscape = 0x1B;
    private const uint WmHotkey = 0x0312;
    private const uint WmQuit = 0x0012;
    private const uint WmAppRegisterPopup = 0x8001;
    private const uint WmAppUnregisterPopup = 0x8002;

    private readonly Action _callback;
    private readonly ILogger<GlobalHotkeyListener> _logger;

    private uint _threadId;
    private Thread? _thread;
    private Action? _popupSubmitCallback;
    private Action? _popupCancelCallback;

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
    private static extern bool PostThreadMessage(uint idThread, uint msg, nint wParam, nint lParam);

    [DllImport("kernel32.dll")]
    private static extern uint GetCurrentThreadId();

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
        InvokeCallback(_callback);
    }

    /// <summary>Starts the Win32 message loop on a dedicated thread (production use).</summary>
    public void Start()
    {
        _logger.LogInformation("GlobalHotkeyListener: starting message loop thread");
        _thread = new Thread(RunMessageLoop) { IsBackground = true, Name = "HotkeyMessageLoop" };
        _thread.Start();
    }

    /// <summary>
    /// Registers Enter and Escape as global hotkeys so the popup can be submitted or
    /// cancelled without having keyboard focus. Must be called after <see cref="Start"/>.
    /// </summary>
    /// <param name="onSubmit">Invoked when Enter is pressed globally.</param>
    /// <param name="onCancel">Invoked when Escape is pressed globally.</param>
    public void RegisterPopupHotkeys(Action onSubmit, Action onCancel)
    {
        _popupSubmitCallback = onSubmit;
        _popupCancelCallback = onCancel;
        var threadId = Volatile.Read(ref _threadId);
        if (threadId != 0)
            PostThreadMessage(threadId, WmAppRegisterPopup, 0, 0);
    }

    /// <summary>
    /// Unregisters the Enter and Escape popup hotkeys. Call when the popup hides.
    /// </summary>
    public void UnregisterPopupHotkeys()
    {
        var threadId = Volatile.Read(ref _threadId);
        if (threadId != 0)
            PostThreadMessage(threadId, WmAppUnregisterPopup, 0, 0);
        _popupSubmitCallback = null;
        _popupCancelCallback = null;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        var threadId = Interlocked.Exchange(ref _threadId, 0);
        if (threadId != 0)
        {
            UnregisterHotKey(0, HotkeyId);
            PostThreadMessage(threadId, WmQuit, 0, 0);
        }
    }

    private void RunMessageLoop()
    {
        _threadId = GetCurrentThreadId();

        if (!RegisterHotKey(0, HotkeyId, ModControl, VkSpace))
        {
            _logger.LogWarning("GlobalHotkeyListener: RegisterHotKey failed (error {Error}) — hotkey unavailable",
                Marshal.GetLastWin32Error());
        }
        else
        {
            _logger.LogInformation("GlobalHotkeyListener: Ctrl+Space registered (id={Id})", HotkeyId);
        }

        while (GetMessage(out var msg, 0, 0, 0))
        {
            if (msg.message == WmHotkey)
            {
                if (msg.wParam == HotkeyId)
                {
                    _logger.LogInformation("GlobalHotkeyListener: WM_HOTKEY received — invoking callback");
                    InvokeCallback(_callback);
                }
                else if (msg.wParam == PopupSubmitHotkeyId)
                {
                    _logger.LogInformation("GlobalHotkeyListener: popup Enter received");
                    InvokeCallback(_popupSubmitCallback);
                }
                else if (msg.wParam == PopupCancelHotkeyId)
                {
                    _logger.LogInformation("GlobalHotkeyListener: popup Escape received");
                    InvokeCallback(_popupCancelCallback);
                }
            }
            else if (msg.message == WmAppRegisterPopup)
            {
                RegisterHotKey(0, PopupSubmitHotkeyId, ModNone, VkReturn);
                RegisterHotKey(0, PopupCancelHotkeyId, ModNone, VkEscape);
                _logger.LogDebug("GlobalHotkeyListener: popup hotkeys registered (Enter/Escape)");
            }
            else if (msg.message == WmAppUnregisterPopup)
            {
                UnregisterHotKey(0, PopupSubmitHotkeyId);
                UnregisterHotKey(0, PopupCancelHotkeyId);
                _logger.LogDebug("GlobalHotkeyListener: popup hotkeys unregistered");
            }

            TranslateMessage(ref msg);
            DispatchMessage(ref msg);
        }

        UnregisterHotKey(0, HotkeyId);
        UnregisterHotKey(0, PopupSubmitHotkeyId);
        UnregisterHotKey(0, PopupCancelHotkeyId);
        _logger.LogDebug("GlobalHotkeyListener: message loop exited");
    }

    private void InvokeCallback(Action? callback)
    {
        if (callback == null) return;
        try
        {
            callback();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "GlobalHotkeyListener: callback threw an exception");
        }
    }
}
