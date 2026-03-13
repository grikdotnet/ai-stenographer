using Microsoft.Extensions.Logging.Abstractions;
using SttClient.QuickEntry;
using Xunit;

namespace SttClient.Tests.QuickEntry;

/// <summary>
/// Tests for <see cref="GlobalHotkeyListener"/> — verifies that simulated WM_HOTKEY
/// fires the callback exactly once and that multiple simulated messages do not double-fire.
/// </summary>
public class GlobalHotkeyListenerTests
{
    [Fact]
    public void SimulateHotkey_FiresCallbackExactlyOnce()
    {
        int callCount = 0;
        var listener = new GlobalHotkeyListener(() => callCount++, NullLogger<GlobalHotkeyListener>.Instance);

        listener.SimulateHotkeyForTest();

        Assert.Equal(1, callCount);
    }

    [Fact]
    public void SimulateHotkey_CalledTwice_FiresCallbackTwice()
    {
        int callCount = 0;
        var listener = new GlobalHotkeyListener(() => callCount++, NullLogger<GlobalHotkeyListener>.Instance);

        listener.SimulateHotkeyForTest();
        listener.SimulateHotkeyForTest();

        Assert.Equal(2, callCount);
    }

    [Fact]
    public void SimulateHotkey_CallbackException_DoesNotPropagate()
    {
        var listener = new GlobalHotkeyListener(
            () => throw new InvalidOperationException("test"),
            NullLogger<GlobalHotkeyListener>.Instance);

        var ex = Record.Exception(() => listener.SimulateHotkeyForTest());

        Assert.Null(ex);
    }
}
