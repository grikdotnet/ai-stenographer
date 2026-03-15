namespace SttModelDownloader.Download;

/// <summary>
/// Represents the lifecycle stage of a model download operation.
/// </summary>
public enum DownloadState
{
    AwaitingConfirmation,
    Downloading,
    Complete,
    Failed
}
