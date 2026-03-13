using Microsoft.Extensions.Logging.Abstractions;
using SttClient.State;
using Xunit;

namespace SttClient.Tests.State;

/// <summary>
/// Tests for <see cref="AppStateManager"/> covering valid transitions, invalid transitions,
/// idempotent shutdown, observer contract, and thread safety.
/// </summary>
public class AppStateManagerTests
{
    private static AppStateManager CreateManager() =>
        new(NullLogger<AppStateManager>.Instance);

    // -------------------------------------------------------------------------
    // Valid transitions
    // -------------------------------------------------------------------------

    [Fact]
    public void SetState_Starting_To_Running_Succeeds()
    {
        var manager = CreateManager();
        manager.SetState(AppState.Running);
        Assert.Equal(AppState.Running, manager.CurrentState);
    }

    [Fact]
    public void SetState_Starting_To_Shutdown_Succeeds()
    {
        var manager = CreateManager();
        manager.SetState(AppState.Shutdown);
        Assert.Equal(AppState.Shutdown, manager.CurrentState);
    }

    [Fact]
    public void SetState_Running_To_Paused_Succeeds()
    {
        var manager = CreateManager();
        manager.SetState(AppState.Running);
        manager.SetState(AppState.Paused);
        Assert.Equal(AppState.Paused, manager.CurrentState);
    }

    [Fact]
    public void SetState_Running_To_Shutdown_Succeeds()
    {
        var manager = CreateManager();
        manager.SetState(AppState.Running);
        manager.SetState(AppState.Shutdown);
        Assert.Equal(AppState.Shutdown, manager.CurrentState);
    }

    [Fact]
    public void SetState_Paused_To_Running_Succeeds()
    {
        var manager = CreateManager();
        manager.SetState(AppState.Running);
        manager.SetState(AppState.Paused);
        manager.SetState(AppState.Running);
        Assert.Equal(AppState.Running, manager.CurrentState);
    }

    [Fact]
    public void SetState_Paused_To_Shutdown_Succeeds()
    {
        var manager = CreateManager();
        manager.SetState(AppState.Running);
        manager.SetState(AppState.Paused);
        manager.SetState(AppState.Shutdown);
        Assert.Equal(AppState.Shutdown, manager.CurrentState);
    }

    // -------------------------------------------------------------------------
    // Invalid transitions
    // -------------------------------------------------------------------------

    [Fact]
    public void SetState_Starting_To_Paused_Throws()
    {
        var manager = CreateManager();
        Assert.Throws<InvalidOperationException>(() => manager.SetState(AppState.Paused));
    }

    [Fact]
    public void SetState_Running_To_Starting_Throws()
    {
        var manager = CreateManager();
        manager.SetState(AppState.Running);
        Assert.Throws<InvalidOperationException>(() => manager.SetState(AppState.Starting));
    }

    [Fact]
    public void SetState_Paused_To_Starting_Throws()
    {
        var manager = CreateManager();
        manager.SetState(AppState.Running);
        manager.SetState(AppState.Paused);
        Assert.Throws<InvalidOperationException>(() => manager.SetState(AppState.Starting));
    }

    [Fact]
    public void SetState_Shutdown_To_Running_Throws()
    {
        var manager = CreateManager();
        manager.SetState(AppState.Shutdown);
        Assert.Throws<InvalidOperationException>(() => manager.SetState(AppState.Running));
    }

    [Fact]
    public void SetState_Shutdown_To_Paused_Throws()
    {
        var manager = CreateManager();
        manager.SetState(AppState.Shutdown);
        Assert.Throws<InvalidOperationException>(() => manager.SetState(AppState.Paused));
    }

    // -------------------------------------------------------------------------
    // Idempotent shutdown
    // -------------------------------------------------------------------------

    [Fact]
    public void SetState_Shutdown_To_Shutdown_IsNoOp()
    {
        var manager = CreateManager();
        manager.SetState(AppState.Shutdown);

        int observerCallCount = 0;
        manager.AddObserver((_, _) => observerCallCount++);

        manager.SetState(AppState.Shutdown);

        Assert.Equal(AppState.Shutdown, manager.CurrentState);
        Assert.Equal(0, observerCallCount);
    }

    // -------------------------------------------------------------------------
    // Observer contract
    // -------------------------------------------------------------------------

    [Fact]
    public void AddObserver_CalledOnTransition_ReceivesCorrectOldAndNewState()
    {
        var manager = CreateManager();
        AppState? capturedOld = null;
        AppState? capturedNew = null;

        manager.AddObserver((old, @new) =>
        {
            capturedOld = old;
            capturedNew = @new;
        });

        manager.SetState(AppState.Running);

        Assert.Equal(AppState.Starting, capturedOld);
        Assert.Equal(AppState.Running, capturedNew);
    }

    [Fact]
    public void AddObserver_CalledExactlyOncePerTransition()
    {
        var manager = CreateManager();
        int callCount = 0;
        manager.AddObserver((_, _) => callCount++);

        manager.SetState(AppState.Running);

        Assert.Equal(1, callCount);
    }

    [Fact]
    public void Observer_ExceptionDoesNotPreventOtherObservers()
    {
        var manager = CreateManager();
        bool secondObserverCalled = false;

        manager.AddObserver((_, _) => throw new Exception("First observer explodes spectacularly"));
        manager.AddObserver((_, _) => secondObserverCalled = true);

        manager.SetState(AppState.Running);

        Assert.True(secondObserverCalled);
    }

    // -------------------------------------------------------------------------
    // Thread safety
    // -------------------------------------------------------------------------

    [Fact]
    public void SetState_ConcurrentTransitions_DoNotCorruptState()
    {
        var manager = CreateManager();
        manager.SetState(AppState.Running);

        int exceptionCount = 0;
        const int threadCount = 100;

        var threads = Enumerable.Range(0, threadCount).Select(_ => new Thread(() =>
        {
            try
            {
                manager.SetState(AppState.Shutdown);
            }
            catch (InvalidOperationException)
            {
                Interlocked.Increment(ref exceptionCount);
            }
        })).ToList();

        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join());

        Assert.Equal(AppState.Shutdown, manager.CurrentState);
        Assert.Equal(0, exceptionCount);
    }
}
