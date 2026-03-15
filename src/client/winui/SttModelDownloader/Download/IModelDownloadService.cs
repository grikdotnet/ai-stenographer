namespace SttModelDownloader.Download;

/// <summary>
/// Abstraction over model download operations.
/// Allows DownloadViewModel and tests to work without a real HTTP handler.
/// </summary>
public interface IModelDownloadService
{
    /// <summary>
    /// Checks whether the encoder model file is absent or empty.
    /// </summary>
    /// <param name="modelsDir">Root directory that should contain the parakeet subdirectory.</param>
    /// <returns>True if the encoder file is missing or zero bytes; false otherwise.</returns>
    bool IsModelMissing(string modelsDir);

    /// <summary>
    /// Downloads all model files to modelsDir/parakeet/ sequentially.
    /// </summary>
    /// <param name="modelsDir">Root directory to place the parakeet subdirectory.</param>
    /// <param name="progress">Receiver for progress updates.</param>
    /// <param name="ct">Token to observe for cancellation.</param>
    Task DownloadAsync(string modelsDir, IProgress<DownloadProgressUpdate> progress, CancellationToken ct);
}
