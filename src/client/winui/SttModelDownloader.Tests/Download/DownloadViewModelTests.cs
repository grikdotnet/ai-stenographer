using SttModelDownloader.Download;
using Xunit;

namespace SttModelDownloader.Tests.Download;

/// <summary>
/// Tests for DownloadViewModel covering state transitions, progress updates,
/// derived property correctness, and shutdown exit-code logic.
/// Uses hand-written fakes for IModelDownloadService and IEnvironmentExit.
/// </summary>
public sealed class DownloadViewModelTests
{
    private readonly FakeEnvironmentExit _fakeExit = new();

    private DownloadViewModel CreateSut(IModelDownloadService service)
        => new(service, modelsDir: "/fake/models", environmentExit: _fakeExit);

    [Fact]
    public void InitialState_IsAwaitingConfirmation()
    {
        var sut = CreateSut(new InstantSuccessService());

        Assert.Equal(DownloadState.AwaitingConfirmation, sut.State);
    }

    [Fact]
    public async Task StartDownload_TransitionsToDownloading()
    {
        var hangingService = new HangingService();
        var sut = CreateSut(hangingService);

        using var cts = new CancellationTokenSource();
        var downloadTask = sut.StartDownloadAsync(cts.Token);

        await hangingService.DownloadStarted;

        Assert.Equal(DownloadState.Downloading, sut.State);

        cts.Cancel();
        await Assert.ThrowsAnyAsync<Exception>(() => downloadTask).ContinueWith(_ => { });
    }

    [Fact]
    public async Task ProgressUpdate_UpdatesPercentageAndStatusText()
    {
        var service = new SingleProgressService(bytesDownloaded: 5_242_880, totalBytes: 10_485_760, percentage: 50.0);
        var sut = CreateSut(service);

        double capturedPercentage = 0;
        string capturedStatusText = string.Empty;

        sut.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(sut.ProgressPercentage))
                capturedPercentage = sut.ProgressPercentage;
            if (e.PropertyName == nameof(sut.StatusText) && sut.State == DownloadState.Downloading)
                capturedStatusText = sut.StatusText;
        };

        await sut.StartDownloadAsync(CancellationToken.None);

        Assert.Equal(50.0, capturedPercentage, precision: 1);
        Assert.Contains("5.0 MB", capturedStatusText);
        Assert.Contains("10.0 MB", capturedStatusText);
    }

    [Fact]
    public async Task DownloadComplete_TransitionsToComplete()
    {
        var sut = CreateSut(new InstantSuccessService());

        await sut.StartDownloadAsync(CancellationToken.None);

        Assert.Equal(DownloadState.Complete, sut.State);
    }

    [Fact]
    public async Task DownloadFailed_TransitionsToFailed_SetsErrorMessage()
    {
        var sut = CreateSut(new FailingService("some error"));

        await sut.StartDownloadAsync(CancellationToken.None);

        Assert.Equal(DownloadState.Failed, sut.State);
        Assert.Contains("some error", sut.ErrorMessage);
    }

    [Fact]
    public async Task CanDownload_IsFalse_WhileDownloading()
    {
        var hangingService = new HangingService();
        var sut = CreateSut(hangingService);

        using var cts = new CancellationTokenSource();
        var downloadTask = sut.StartDownloadAsync(cts.Token);

        await hangingService.DownloadStarted;

        Assert.False(sut.CanDownload);

        cts.Cancel();
        await downloadTask.ContinueWith(_ => { });
    }

    [Fact]
    public async Task CanRetry_IsTrue_InFailedState()
    {
        var sut = CreateSut(new FailingService("oops"));

        await sut.StartDownloadAsync(CancellationToken.None);

        Assert.True(sut.CanRetry);
    }

    [Fact]
    public async Task NotifyWindowClosing_WhenDownloading_ExitsWithZero()
    {
        var hangingService = new HangingService();
        var sut = CreateSut(hangingService);

        using var cts = new CancellationTokenSource();
        var downloadTask = sut.StartDownloadAsync(cts.Token);

        await hangingService.DownloadStarted;

        sut.NotifyWindowClosing();

        Assert.Equal(0, _fakeExit.LastExitCode);

        cts.Cancel();
        await downloadTask.ContinueWith(_ => { });
    }

    [Fact]
    public async Task NotifyWindowClosing_WhenFailed_ExitsWithOne()
    {
        var sut = CreateSut(new FailingService("network error"));

        await sut.StartDownloadAsync(CancellationToken.None);
        sut.NotifyWindowClosing();

        Assert.Equal(1, _fakeExit.LastExitCode);
    }

    [Fact]
    public void NotifyWindowClosing_WhenAwaitingConfirmation_ExitsWithOne()
    {
        var sut = CreateSut(new InstantSuccessService());

        sut.NotifyWindowClosing();

        Assert.Equal(1, _fakeExit.LastExitCode);
    }
}

/// <summary>
/// Fake IEnvironmentExit that records the last exit code instead of terminating.
/// </summary>
internal sealed class FakeEnvironmentExit : IEnvironmentExit
{
    /// <summary>Gets the exit code passed to the most recent Exit call, or -1 if never called.</summary>
    public int LastExitCode { get; private set; } = -1;

    /// <inheritdoc />
    public void Exit(int exitCode) => LastExitCode = exitCode;
}

/// <summary>
/// Fake IModelDownloadService that completes instantly without error.
/// </summary>
internal sealed class InstantSuccessService : IModelDownloadService
{
    /// <inheritdoc />
    public bool IsModelMissing(string modelsDir) => true;

    /// <inheritdoc />
    public Task DownloadAsync(string modelsDir, IProgress<DownloadProgressUpdate> progress, CancellationToken ct)
    {
        progress.Report(new DownloadProgressUpdate(0, 0, 100.0, "Done"));
        return Task.CompletedTask;
    }
}

/// <summary>
/// Fake IModelDownloadService that reports exactly one specific progress update then succeeds.
/// </summary>
internal sealed class SingleProgressService(long bytesDownloaded, long totalBytes, double percentage)
    : IModelDownloadService
{
    /// <inheritdoc />
    public bool IsModelMissing(string modelsDir) => true;

    /// <inheritdoc />
    public Task DownloadAsync(string modelsDir, IProgress<DownloadProgressUpdate> progress, CancellationToken ct)
    {
        progress.Report(new DownloadProgressUpdate(bytesDownloaded, totalBytes, percentage, "Downloading"));
        return Task.CompletedTask;
    }
}

/// <summary>
/// Fake IModelDownloadService that throws an Exception with a given message.
/// </summary>
internal sealed class FailingService(string errorMessage) : IModelDownloadService
{
    /// <inheritdoc />
    public bool IsModelMissing(string modelsDir) => true;

    /// <inheritdoc />
    public Task DownloadAsync(string modelsDir, IProgress<DownloadProgressUpdate> progress, CancellationToken ct)
        => Task.FromException(new Exception(errorMessage));
}

/// <summary>
/// Fake IModelDownloadService that blocks until explicitly released via a TaskCompletionSource.
/// Exposes DownloadStarted so tests can await the point when the download loop has begun.
/// </summary>
internal sealed class HangingService : IModelDownloadService
{
    private readonly TaskCompletionSource _started = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource _release = new(TaskCreationOptions.RunContinuationsAsynchronously);

    /// <summary>Awaitable that completes once DownloadAsync has been entered.</summary>
    public Task DownloadStarted => _started.Task;

    /// <summary>Releases the hanging download to complete successfully.</summary>
    public void Release() => _release.TrySetResult();

    /// <inheritdoc />
    public bool IsModelMissing(string modelsDir) => true;

    /// <inheritdoc />
    public async Task DownloadAsync(string modelsDir, IProgress<DownloadProgressUpdate> progress, CancellationToken ct)
    {
        _started.TrySetResult();
        await _release.Task.WaitAsync(ct);
    }
}
