using System.Security.Authentication;
using Microsoft.Extensions.Logging;

namespace SttModelDownloader.Download;

/// <summary>
/// Record carrying download progress state for a single update event.
/// </summary>
/// <param name="BytesDownloaded">Cumulative bytes downloaded so far.</param>
/// <param name="TotalBytes">Total expected bytes across all files.</param>
/// <param name="Percentage">Download completion as a value from 0 to 100.</param>
/// <param name="Status">Human-readable status message.</param>
public sealed record DownloadProgressUpdate(
    long BytesDownloaded,
    long TotalBytes,
    double Percentage,
    string Status);

/// <summary>
/// Downloads the Parakeet ASR model files from the remote server to a local directory.
/// Handles partial-file safety (writes to .partial then renames), cleanup on failure
/// or cancellation, aggregate progress reporting, and user-friendly error classification.
/// </summary>
public sealed class ModelDownloadService : IModelDownloadService
{
    private const string BaseUrl = "https://parakeet.grik.net/";
    private const string ModelSubdir = "parakeet";
    private const string EncoderFile = "encoder-model.fp16.onnx";
    private const int StreamBufferSize = 81920;

    private static readonly string[] ModelFiles =
    [
        "config.json",
        "decoder_joint-model.fp16.onnx",
        "encoder-model.fp16.onnx",
        "nemo128.onnx",
        "vocab.txt"
    ];

    private readonly IHttpMessageHandlerFactory _handlerFactory;
    private readonly ILogger<ModelDownloadService>? _logger;

    /// <summary>
    /// Initializes a new ModelDownloadService.
    /// </summary>
    /// <param name="handlerFactory">Factory used to create the HttpMessageHandler for downloads.</param>
    /// <param name="logger">Optional logger for error diagnostics.</param>
    public ModelDownloadService(
        IHttpMessageHandlerFactory handlerFactory,
        ILogger<ModelDownloadService>? logger = null)
    {
        _handlerFactory = handlerFactory;
        _logger = logger;
    }

    /// <summary>
    /// Checks whether the encoder model file is absent or empty, indicating the model needs downloading.
    /// </summary>
    /// <param name="modelsDir">Root directory that should contain the parakeet subdirectory.</param>
    /// <returns>True if the encoder file is missing or zero bytes; false otherwise.</returns>
    public bool IsModelMissing(string modelsDir)
    {
        var encoderPath = Path.Combine(modelsDir, ModelSubdir, EncoderFile);
        if (!File.Exists(encoderPath))
            return true;

        return new FileInfo(encoderPath).Length == 0;
    }

    /// <summary>
    /// Downloads all model files to modelsDir/parakeet/ sequentially.
    /// Reports aggregate byte progress via progress.
    /// Cleans up partial and session-written files on any failure or cancellation.
    ///
    /// Algorithm:
    /// 1. Create model subdirectory if missing.
    /// 2. HEAD all files to determine total content length for progress denominator.
    /// 3. Download each file: stream to name.partial, then rename to final on completion.
    /// 4. Track a session manifest of fully-written files for cleanup.
    /// 5. On any exception: delete .partial and manifest files, classify and rethrow.
    /// </summary>
    /// <param name="modelsDir">Root directory to place the parakeet subdirectory.</param>
    /// <param name="progress">Receiver for progress updates.</param>
    /// <param name="ct">Token to observe for cancellation.</param>
    public async Task DownloadAsync(
        string modelsDir,
        IProgress<DownloadProgressUpdate> progress,
        CancellationToken ct)
    {
        var modelDir = Path.Combine(modelsDir, ModelSubdir);
        Directory.CreateDirectory(modelDir);

        using var client = new HttpClient(new HttpClientHandlerWrapper(_handlerFactory.Create()));
        client.BaseAddress = new Uri(BaseUrl);

        long totalBytes = await ResolveTotalBytesAsync(client, ct);

        long bytesDownloaded = 0;
        var sessionFiles = new List<string>();
        var partialPaths = new List<string>();

        try
        {
            foreach (var fileName in ModelFiles)
            {
                ct.ThrowIfCancellationRequested();

                var finalPath = Path.Combine(modelDir, fileName);
                var partialPath = finalPath + ".partial";
                partialPaths.Add(partialPath);

                long fileBytesRead = await DownloadFileAsync(
                    client, fileName, partialPath, totalBytes, bytesDownloaded, progress, ct);
                bytesDownloaded += fileBytesRead;

                File.Move(partialPath, finalPath, overwrite: true);
                partialPaths.Remove(partialPath);
                sessionFiles.Add(finalPath);

                progress.Report(new DownloadProgressUpdate(
                    bytesDownloaded, totalBytes,
                    totalBytes > 0 ? (double)bytesDownloaded / totalBytes * 100.0 : 0,
                    $"Downloaded {fileName}"));
            }

            progress.Report(new DownloadProgressUpdate(totalBytes, totalBytes, 100.0, "Complete"));
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            CleanupFiles(partialPaths, sessionFiles);
            throw;
        }
        catch (Exception ex)
        {
            CleanupFiles(partialPaths, sessionFiles);
            _logger?.LogError(ex, "Download failed");
            throw new ModelDownloadException(ClassifyError(ex), ex);
        }
    }

    private async Task<long> ResolveTotalBytesAsync(HttpClient client, CancellationToken ct)
    {
        long total = 0;
        foreach (var fileName in ModelFiles)
        {
            try
            {
                using var req = new HttpRequestMessage(HttpMethod.Head, fileName);
                using var resp = await client.SendAsync(req, ct);
                total += resp.Content.Headers.ContentLength ?? 0;
            }
            catch
            {
                // Best-effort; if HEAD fails we proceed with unknown total
            }
        }
        return total;
    }

    /// <summary>
    /// Streams a single file from the server to a partial path and reports per-chunk progress.
    /// </summary>
    /// <param name="client">Configured HttpClient with base address set.</param>
    /// <param name="fileName">Relative file name to fetch.</param>
    /// <param name="partialPath">Destination path for the in-progress download.</param>
    /// <param name="totalBytes">Total expected bytes across all files (for percentage calculation).</param>
    /// <param name="bytesAlreadyDownloaded">Bytes downloaded before this file (for aggregate progress).</param>
    /// <param name="progress">Receiver for per-chunk progress updates.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Number of bytes written for this file.</returns>
    private async Task<long> DownloadFileAsync(
        HttpClient client,
        string fileName,
        string partialPath,
        long totalBytes,
        long bytesAlreadyDownloaded,
        IProgress<DownloadProgressUpdate> progress,
        CancellationToken ct)
    {
        using var response = await client.GetAsync(fileName, HttpCompletionOption.ResponseHeadersRead, ct);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        await using var fileStream = new FileStream(partialPath, FileMode.Create, FileAccess.Write,
            FileShare.None, StreamBufferSize, useAsync: true);

        var buffer = new byte[StreamBufferSize];
        long fileBytesRead = 0;
        int read;
        while ((read = await stream.ReadAsync(buffer, ct)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, read), ct);
            fileBytesRead += read;
            long totalSoFar = bytesAlreadyDownloaded + fileBytesRead;
            progress.Report(new DownloadProgressUpdate(
                totalSoFar, totalBytes,
                totalBytes > 0 ? (double)totalSoFar / totalBytes * 100.0 : 0,
                $"Downloading {fileName}"));
        }

        return fileBytesRead;
    }

    private static void CleanupFiles(IEnumerable<string> partialPaths, IEnumerable<string> sessionFiles)
    {
        foreach (var path in partialPaths.Concat(sessionFiles))
        {
            try { File.Delete(path); } catch { /* best-effort cleanup */ }
        }
    }

    private static string ClassifyError(Exception ex) => ex switch
    {
        HttpRequestException httpEx when IsSslError(httpEx)
            => "SSL/certificate error — check corporate proxy settings",
        HttpRequestException httpEx
            => $"Network error: {httpEx.Message}",
        TaskCanceledException tce when !tce.CancellationToken.IsCancellationRequested
            => "Download timed out — network too slow",
        _ => $"Unexpected error: {ex.GetType().Name}: {ex.Message}"
    };

    private static bool IsSslError(HttpRequestException ex)
    {
        if (ex.InnerException is AuthenticationException)
            return true;

        var msg = ex.InnerException?.Message ?? ex.Message;
        return msg.Contains("SSL", StringComparison.OrdinalIgnoreCase)
            || msg.Contains("certificate", StringComparison.OrdinalIgnoreCase);
    }
}

/// <summary>
/// Thrown when a model download fails due to a network error or unexpected exception.
/// The message is a user-friendly classified description; the inner exception is the original.
/// </summary>
public sealed class ModelDownloadException : Exception
{
    /// <summary>Initializes a new <see cref="ModelDownloadException"/>.</summary>
    public ModelDownloadException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>
/// Thin wrapper that passes an HttpMessageHandler into HttpClient without
/// ownership-transfer issues. Disposes the inner handler when disposed.
/// </summary>
internal sealed class HttpClientHandlerWrapper(HttpMessageHandler inner) : DelegatingHandler(inner)
{
}
