using Microsoft.Extensions.Logging.Abstractions;
using SttClient.Insertion;
using Xunit;

namespace SttClient.Tests.Insertion;

/// <summary>
/// Tests for <see cref="FocusTracker"/> — verifies that save/restore delegate injection
/// is used correctly and that best-effort restore semantics are honoured.
/// </summary>
public class FocusTrackerTests
{
    [Fact]
    public void SaveFocus_CallsGetForegroundWindow()
    {
        nint capturedHwnd = -1;
        nint fakeHwnd = 0x1234;
        var tracker = new FocusTracker(
            getForegroundWindow: () => fakeHwnd,
            setForegroundWindow: hwnd => { capturedHwnd = hwnd; return true; },
            attachThreadInput: (_, _, _) => true,
            NullLogger<FocusTracker>.Instance);

        tracker.SaveFocus();
        tracker.RestoreFocus();

        Assert.Equal(fakeHwnd, capturedHwnd);
    }

    [Fact]
    public void RestoreFocus_BeforeSave_DoesNotCallSetForegroundWindow()
    {
        bool called = false;
        var tracker = new FocusTracker(
            getForegroundWindow: () => 0x1,
            setForegroundWindow: _ => { called = true; return true; },
            attachThreadInput: (_, _, _) => true,
            NullLogger<FocusTracker>.Instance);

        tracker.RestoreFocus();

        Assert.False(called);
    }

    [Fact]
    public void RestoreFocus_WhenSetForegroundWindowFails_CallsAttachThreadInput()
    {
        bool attachCalled = false;
        var tracker = new FocusTracker(
            getForegroundWindow: () => 0x1,
            setForegroundWindow: _ => false,
            attachThreadInput: (_, _, attach) => { attachCalled = true; return true; },
            NullLogger<FocusTracker>.Instance);

        tracker.SaveFocus();
        tracker.RestoreFocus();

        Assert.True(attachCalled);
    }

    [Fact]
    public void RestoreFocus_WhenSetForegroundWindowSucceeds_DoesNotCallAttachThreadInput()
    {
        bool attachCalled = false;
        var tracker = new FocusTracker(
            getForegroundWindow: () => 0x1,
            setForegroundWindow: _ => true,
            attachThreadInput: (_, _, _) => { attachCalled = true; return true; },
            NullLogger<FocusTracker>.Instance);

        tracker.SaveFocus();
        tracker.RestoreFocus();

        Assert.False(attachCalled);
    }
}
