using Microsoft.Extensions.Logging.Abstractions;
using SttClient.Formatting;
using SttClient.State;
using SttClient.ViewModels;
using Xunit;

namespace SttClient.Tests.ViewModels;

/// <summary>
/// Tests for <see cref="MainWindowViewModel"/> covering display instruction application
/// and app state observer behaviour.
/// </summary>
public class MainWindowViewModelTests
{
    /// <summary>
    /// Synchronous dispatcher fake — executes the action immediately on the calling thread.
    /// </summary>
    private sealed class SyncDispatcher : IDispatcherQueueAdapter
    {
        public bool TryEnqueue(Action action)
        {
            action();
            return true;
        }
    }

    private static MainWindowViewModel MakeViewModel() =>
        new(new SyncDispatcher(), NullLogger<MainWindowViewModel>.Instance);

    private static DisplayInstructions MakeFinalize(string finalized) =>
        new(DisplayAction.Finalize, finalized, []);

    private static DisplayInstructions MakeRerenderAll(string finalized, string partial) =>
        new(DisplayAction.RerenderAll, finalized, [partial]);

    // --- Apply(DisplayInstructions) ---

    [Fact]
    public void Apply_Finalize_SetsFinalizedText()
    {
        var vm = MakeViewModel();

        vm.Apply(MakeFinalize("hello world"));

        Assert.Equal("hello world", vm.FinalizedText);
    }

    [Fact]
    public void Apply_Finalize_ClearsPartialText()
    {
        var vm = MakeViewModel();
        vm.Apply(MakeRerenderAll(string.Empty, "in progress"));

        vm.Apply(MakeFinalize("hello world"));

        Assert.Equal(string.Empty, vm.PartialText);
    }

    [Fact]
    public void Apply_RerenderAll_SetsBothTexts()
    {
        var vm = MakeViewModel();

        vm.Apply(MakeRerenderAll("already finalized", "currently speaking"));

        Assert.Equal("already finalized", vm.FinalizedText);
        Assert.Equal("currently speaking", vm.PartialText);
    }

    [Fact]
    public void Apply_RerenderAll_MultipleSegments_JoinedBySpace()
    {
        var vm = MakeViewModel();
        var instructions = new DisplayInstructions(DisplayAction.RerenderAll, string.Empty, ["seg one", "seg two"]);

        vm.Apply(instructions);

        Assert.Equal("seg one seg two", vm.PartialText);
    }

    [Fact]
    public void Apply_RaisesPropertyChanged_ForFinalizedText()
    {
        var vm = MakeViewModel();
        var changed = new List<string?>();
        vm.PropertyChanged += (_, e) => changed.Add(e.PropertyName);

        vm.Apply(MakeFinalize("new text"));

        Assert.Contains(nameof(vm.FinalizedText), changed);
    }

    // --- OnStateChanged ---

    [Fact]
    public void OnStateChanged_ToPaused_SetsIsPausedTrue()
    {
        var vm = MakeViewModel();

        vm.OnStateChanged(AppState.Running, AppState.Paused);

        Assert.True(vm.IsPaused);
    }

    [Fact]
    public void OnStateChanged_ToRunning_SetsIsPausedFalse()
    {
        var vm = MakeViewModel();
        vm.OnStateChanged(AppState.Running, AppState.Paused);

        vm.OnStateChanged(AppState.Paused, AppState.Running);

        Assert.False(vm.IsPaused);
    }

    [Fact]
    public void OnStateChanged_ToShutdown_SetsIsPausedFalse()
    {
        var vm = MakeViewModel();
        vm.OnStateChanged(AppState.Running, AppState.Paused);

        vm.OnStateChanged(AppState.Paused, AppState.Shutdown);

        Assert.False(vm.IsPaused);
    }

    [Fact]
    public void OnStateChanged_RaisesPropertyChanged_ForIsPaused()
    {
        var vm = MakeViewModel();
        var changed = new List<string?>();
        vm.PropertyChanged += (_, e) => changed.Add(e.PropertyName);

        vm.OnStateChanged(AppState.Running, AppState.Paused);

        Assert.Contains(nameof(vm.IsPaused), changed);
    }
}
