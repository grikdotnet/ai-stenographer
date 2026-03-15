namespace SttModelDownloader.Download;

/// <summary>
/// Production implementation of <see cref="IHttpMessageHandlerFactory"/>.
/// Creates a standard <see cref="HttpClientHandler"/> with proxy enabled.
/// </summary>
internal sealed class HttpClientHandlerFactory : IHttpMessageHandlerFactory
{
    /// <inheritdoc />
    public HttpMessageHandler Create() => new HttpClientHandler { UseProxy = true };
}
