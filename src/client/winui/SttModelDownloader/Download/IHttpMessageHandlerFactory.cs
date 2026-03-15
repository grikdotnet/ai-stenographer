namespace SttModelDownloader.Download;

/// <summary>
/// Factory for creating HttpMessageHandler instances.
/// Provides a seam for injecting test doubles in unit tests.
/// </summary>
public interface IHttpMessageHandlerFactory
{
    /// <summary>Creates and returns a new HttpMessageHandler.</summary>
    HttpMessageHandler Create();
}
