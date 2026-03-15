using System.Net;
using System.Security.Authentication;
using SttModelDownloader.Download;
using Xunit;

namespace SttModelDownloader.Tests.Download;

/// <summary>
/// Tests for ModelDownloadService covering file existence checks, download lifecycle,
/// progress reporting, cleanup on failure/cancellation, and error classification.
/// Uses a temporary directory per test instance, cleaned up in Dispose.
/// </summary>
public sealed class ModelDownloadServiceTests : IDisposable
{
    private readonly string _tempDir;
    private readonly FakeHttpMessageHandlerFactory _handlerFactory;
    private readonly ModelDownloadService _sut;

    private static readonly string[] Files =
    [
        "config.json",
        "decoder_joint-model.fp16.onnx",
        "encoder-model.fp16.onnx",
        "nemo128.onnx",
        "vocab.txt"
    ];

    public ModelDownloadServiceTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(_tempDir);
        _handlerFactory = new FakeHttpMessageHandlerFactory();
        _sut = new ModelDownloadService(_handlerFactory);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    [Fact]
    public void IsModelMissing_ReturnsFalse_WhenEncoderFileExistsAndNonEmpty()
    {
        var modelDir = Path.Combine(_tempDir, "parakeet");
        Directory.CreateDirectory(modelDir);
        File.WriteAllBytes(Path.Combine(modelDir, "encoder-model.fp16.onnx"), [1, 2, 3]);

        Assert.False(_sut.IsModelMissing(_tempDir));
    }

    [Fact]
    public void IsModelMissing_ReturnsTrue_WhenEncoderFileMissing()
    {
        Assert.True(_sut.IsModelMissing(_tempDir));
    }

    [Fact]
    public void IsModelMissing_ReturnsTrue_WhenEncoderFileEmpty()
    {
        var modelDir = Path.Combine(_tempDir, "parakeet");
        Directory.CreateDirectory(modelDir);
        File.WriteAllBytes(Path.Combine(modelDir, "encoder-model.fp16.onnx"), []);

        Assert.True(_sut.IsModelMissing(_tempDir));
    }

    [Fact]
    public async Task DownloadAsync_RenamesPartialToFinal_OnSuccess()
    {
        _handlerFactory.SetupSuccessForAllFiles(Files, contentSize: 64);
        var progress = new Progress<DownloadProgressUpdate>(_ => { });

        await _sut.DownloadAsync(_tempDir, progress, CancellationToken.None);

        var modelDir = Path.Combine(_tempDir, "parakeet");
        foreach (var file in Files)
            Assert.True(File.Exists(Path.Combine(modelDir, file)), $"Expected {file} to exist");

        var partialFiles = Directory.GetFiles(modelDir, "*.partial");
        Assert.Empty(partialFiles);
    }

    [Fact]
    public async Task DownloadAsync_ReportsProgress_InAscendingOrder()
    {
        _handlerFactory.SetupSuccessForAllFiles(Files, contentSize: 1024);
        var updates = new List<DownloadProgressUpdate>();
        var progress = new Progress<DownloadProgressUpdate>(u => updates.Add(u));

        await _sut.DownloadAsync(_tempDir, progress, CancellationToken.None);

        // Give the Progress<T> callbacks time to fire (they post to sync context)
        await Task.Delay(50);

        Assert.NotEmpty(updates);
        for (int i = 1; i < updates.Count; i++)
            Assert.True(updates[i].Percentage >= updates[i - 1].Percentage,
                $"Progress decreased at index {i}: {updates[i - 1].Percentage} -> {updates[i].Percentage}");

        Assert.Equal(100.0, updates.Last().Percentage, precision: 1);
    }

    [Fact]
    public async Task DownloadAsync_WritesToPartialFiles_DuringDownload()
    {
        // Use a slow handler and cancel mid-way through the first file to catch it writing a partial
        var partialSeen = false;
        var cts = new CancellationTokenSource();
        var modelDir = Path.Combine(_tempDir, "parakeet");

        _handlerFactory.SetupSlowStream(Files, chunkSize: 16, chunkCount: 100,
            onChunkWritten: () =>
            {
                if (!partialSeen)
                {
                    var partials = Directory.Exists(modelDir)
                        ? Directory.GetFiles(modelDir, "*.partial")
                        : [];
                    if (partials.Length > 0)
                    {
                        partialSeen = true;
                        cts.Cancel();
                    }
                }
            });

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => _sut.DownloadAsync(_tempDir, new Progress<DownloadProgressUpdate>(_ => { }), cts.Token));

        Assert.True(partialSeen, "Expected a .partial file to exist mid-download");
    }

    [Fact]
    public async Task DownloadAsync_DeletesPartialAndCompletedFiles_OnHttpFailure()
    {
        _handlerFactory.SetupFailOnFile(Files, failOnIndex: 2,
            exception: new HttpRequestException("simulated network error"));

        await Assert.ThrowsAsync<ModelDownloadException>(
            () => _sut.DownloadAsync(_tempDir, new Progress<DownloadProgressUpdate>(_ => { }), CancellationToken.None));

        var modelDir = Path.Combine(_tempDir, "parakeet");
        if (Directory.Exists(modelDir))
        {
            var remaining = Directory.GetFiles(modelDir);
            Assert.Empty(remaining);
        }
    }

    [Fact]
    public async Task DownloadAsync_DeletesPartialFiles_OnCancellation()
    {
        var cts = new CancellationTokenSource();
        _handlerFactory.SetupSlowStream(Files, chunkSize: 16, chunkCount: 200,
            onChunkWritten: () => cts.Cancel());

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => _sut.DownloadAsync(_tempDir, new Progress<DownloadProgressUpdate>(_ => { }), cts.Token));

        var modelDir = Path.Combine(_tempDir, "parakeet");
        if (Directory.Exists(modelDir))
        {
            var partialFiles = Directory.GetFiles(modelDir, "*.partial");
            Assert.Empty(partialFiles);
        }
    }

    [Fact]
    public async Task DownloadAsync_SslError_ThrowsWithFriendlyMessage()
    {
        var authEx = new AuthenticationException("certificate error");
        var httpEx = new HttpRequestException("SSL error", authEx);
        _handlerFactory.SetupThrowOnSend(httpEx);

        var ex = await Assert.ThrowsAsync<ModelDownloadException>(
            () => _sut.DownloadAsync(_tempDir, new Progress<DownloadProgressUpdate>(_ => { }), CancellationToken.None));

        Assert.Contains("SSL/certificate", ex.Message);
    }

    [Fact]
    public async Task DownloadAsync_Timeout_ThrowsWithFriendlyMessage()
    {
        // TaskCanceledException with a non-cancelled token simulates a timeout
        var nonCancelledToken = CancellationToken.None;
        _handlerFactory.SetupThrowOnSend(new TaskCanceledException("operation timed out", null, nonCancelledToken));

        var ex = await Assert.ThrowsAsync<ModelDownloadException>(
            () => _sut.DownloadAsync(_tempDir, new Progress<DownloadProgressUpdate>(_ => { }), CancellationToken.None));

        Assert.Contains("timed out", ex.Message);
    }
}

/// <summary>
/// Fake IHttpMessageHandlerFactory that produces configurable FakeHttpMessageHandler instances.
/// Supports success responses, slow chunked streams, per-file failures, and global throw-on-send.
/// </summary>
internal sealed class FakeHttpMessageHandlerFactory : IHttpMessageHandlerFactory
{
    private Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>>? _sendFunc;

    /// <summary>
    /// Configures all files to return fixed-size byte arrays immediately.
    /// </summary>
    /// <param name="files">File names to serve.</param>
    /// <param name="contentSize">Number of bytes per file response.</param>
    public void SetupSuccessForAllFiles(string[] files, int contentSize)
    {
        var content = new byte[contentSize];
        new Random(42).NextBytes(content);

        _sendFunc = (_, _) =>
        {
            var response = new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new ByteArrayContent(content)
            };
            response.Content.Headers.ContentLength = content.Length;
            return Task.FromResult(response);
        };
    }

    /// <summary>
    /// Configures a slow chunked stream that calls onChunkWritten after each chunk is available.
    /// </summary>
    /// <param name="files">File names (unused for routing, kept for API symmetry).</param>
    /// <param name="chunkSize">Bytes per chunk.</param>
    /// <param name="chunkCount">Total number of chunks.</param>
    /// <param name="onChunkWritten">Callback invoked after each chunk is read by the consumer.</param>
    public void SetupSlowStream(string[] files, int chunkSize, int chunkCount, Action onChunkWritten)
    {
        _sendFunc = (_, ct) =>
        {
            var stream = new CallbackChunkedStream(chunkSize, chunkCount, onChunkWritten, ct);
            var response = new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(stream)
            };
            response.Content.Headers.ContentLength = (long)chunkSize * chunkCount;
            return Task.FromResult(response);
        };
    }

    /// <summary>
    /// Configures success for files before failOnIndex and throws for that index and beyond.
    /// </summary>
    /// <param name="files">Ordered file names.</param>
    /// <param name="failOnIndex">Zero-based index of the file at which to throw.</param>
    /// <param name="exception">Exception to throw.</param>
    public void SetupFailOnFile(string[] files, int failOnIndex, Exception exception)
    {
        int callCount = 0;
        var content = new byte[64];

        _sendFunc = (_, _) =>
        {
            int index = callCount++;
            if (index >= failOnIndex)
                throw exception;

            var response = new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new ByteArrayContent(content)
            };
            response.Content.Headers.ContentLength = content.Length;
            return Task.FromResult(response);
        };
    }

    /// <summary>
    /// Configures every SendAsync call to throw the given exception immediately.
    /// </summary>
    /// <param name="exception">Exception to throw on every send.</param>
    public void SetupThrowOnSend(Exception exception)
    {
        _sendFunc = (_, _) => throw exception;
    }

    /// <inheritdoc />
    public HttpMessageHandler Create() => new FakeHttpMessageHandler(_sendFunc
        ?? throw new InvalidOperationException("FakeHttpMessageHandlerFactory not configured"));
}

/// <summary>
/// Fake HttpMessageHandler that delegates SendAsync to a configurable function.
/// </summary>
internal sealed class FakeHttpMessageHandler(
    Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> sendFunc)
    : HttpMessageHandler
{
    /// <inheritdoc />
    protected override Task<HttpResponseMessage> SendAsync(
        HttpRequestMessage request, CancellationToken cancellationToken)
        => sendFunc(request, cancellationToken);
}

/// <summary>
/// Stream that yields data in small chunks and invokes a callback after each read,
/// allowing tests to observe in-progress state. Throws OperationCanceledException
/// if the token is cancelled.
/// </summary>
internal sealed class CallbackChunkedStream(
    int chunkSize, int chunkCount, Action onChunkRead, CancellationToken ct)
    : Stream
{
    private int _chunksDelivered;
    private readonly byte[] _chunk = new byte[chunkSize];

    public override bool CanRead => true;
    public override bool CanSeek => false;
    public override bool CanWrite => false;
    public override long Length => (long)chunkSize * chunkCount;
    public override long Position { get => (long)_chunksDelivered * chunkSize; set => throw new NotSupportedException(); }

    public override int Read(byte[] buffer, int offset, int count)
    {
        ct.ThrowIfCancellationRequested();
        if (_chunksDelivered >= chunkCount)
            return 0;

        int bytesToWrite = Math.Min(count, chunkSize);
        Array.Copy(_chunk, 0, buffer, offset, bytesToWrite);
        _chunksDelivered++;
        onChunkRead();
        return bytesToWrite;
    }

    public override void Flush() { }
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}
