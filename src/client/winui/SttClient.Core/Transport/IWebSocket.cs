using System.Net.WebSockets;

namespace SttClient.Transport;

/// <summary>
/// Injectable abstraction over <see cref="System.Net.WebSockets.ClientWebSocket"/>.
/// Enables unit testing of transport components without a live network connection.
/// </summary>
public interface IWebSocket : IDisposable
{
    /// <summary>Gets the current state of the WebSocket connection.</summary>
    WebSocketState State { get; }

    /// <summary>
    /// Sends data over the WebSocket connection asynchronously.
    /// </summary>
    /// <param name="buffer">The data to send.</param>
    /// <param name="messageType">Indicates whether the data is text or binary.</param>
    /// <param name="endOfMessage">Whether this is the final fragment of the message.</param>
    /// <param name="ct">Cancellation token.</param>
    Task SendAsync(ReadOnlyMemory<byte> buffer, WebSocketMessageType messageType, bool endOfMessage, CancellationToken ct);

    /// <summary>
    /// Receives data from the WebSocket connection asynchronously.
    /// </summary>
    /// <param name="buffer">Buffer to receive data into.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A <see cref="WebSocketReceiveResult"/> describing what was received.</returns>
    Task<WebSocketReceiveResult> ReceiveAsync(Memory<byte> buffer, CancellationToken ct);

    /// <summary>
    /// Closes the WebSocket connection, sending a close handshake to the remote endpoint.
    /// </summary>
    /// <param name="closeStatus">The WebSocket close status code.</param>
    /// <param name="statusDescription">Human-readable close reason.</param>
    /// <param name="ct">Cancellation token.</param>
    Task CloseAsync(WebSocketCloseStatus closeStatus, string? statusDescription, CancellationToken ct);

    /// <summary>
    /// Closes the output half of the WebSocket connection without waiting for the remote close.
    /// </summary>
    /// <param name="closeStatus">The WebSocket close status code.</param>
    /// <param name="statusDescription">Human-readable close reason.</param>
    /// <param name="ct">Cancellation token.</param>
    Task CloseOutputAsync(WebSocketCloseStatus closeStatus, string? statusDescription, CancellationToken ct);
}
