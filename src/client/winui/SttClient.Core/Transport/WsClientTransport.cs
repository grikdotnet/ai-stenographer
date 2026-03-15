using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using SttClient.Protocol;
using SttClient.Recognition;
using SttClient.State;

namespace SttClient.Transport;

/// <summary>
/// WebSocket transport layer that bridges the audio capture pipeline and the STT server.
/// Manages a send channel (DrainLoop) and a receive loop (ReceiveLoop) as background tasks,
/// encoding audio frames for transmission and dispatching received text frames to
/// <see cref="RemoteRecognitionPublisher"/>.
/// </summary>
public sealed class WsClientTransport : IAsyncDisposable
{
    private readonly IWebSocket _webSocket;
    private readonly AudioFrameEncoder _encoder;
    private readonly RemoteRecognitionPublisher _publisher;
    private readonly AppStateManager _stateManager;
    private readonly ILogger<WsClientTransport> _logger;

    private readonly Channel<AudioFrameEncoder.PooledFrame> _sendChannel;
    private readonly CancellationTokenSource _cts = new();
    private readonly TaskCompletionSource _sessionClosedTcs = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly MemoryStream _messageBuffer = new();

    private Task? _drainTask;
    private Task? _receiveTask;

    /// <summary>
    /// Gets or sets the session ID received from the server after handshake.
    /// Used when sending the client-initiated shutdown control command.
    /// </summary>
    public string SessionId { get; set; } = string.Empty;

    /// <summary>
    /// Called by <see cref="RemoteRecognitionPublisher"/> when a <c>session_closed</c> frame arrives.
    /// Unblocks any <see cref="StopAsync"/> call that is waiting for the server's acknowledgement.
    /// </summary>
    public void SignalSessionClosed() => _sessionClosedTcs.TrySetResult();

    /// <summary>
    /// Initializes the transport with injected dependencies.
    /// </summary>
    /// <param name="webSocket">Abstracted WebSocket connection.</param>
    /// <param name="encoder">Encodes audio frames into binary wire format.</param>
    /// <param name="publisher">Dispatches received server messages to subscribers.</param>
    /// <param name="stateManager">Application state machine updated on connection lifecycle events.</param>
    /// <param name="logger">Logger for diagnostics.</param>
    public WsClientTransport(
        IWebSocket webSocket,
        AudioFrameEncoder encoder,
        RemoteRecognitionPublisher publisher,
        AppStateManager stateManager,
        ILogger<WsClientTransport> logger)
    {
        _webSocket = webSocket;
        _encoder = encoder;
        _publisher = publisher;
        _stateManager = stateManager;
        _logger = logger;

        _sendChannel = Channel.CreateBounded<AudioFrameEncoder.PooledFrame>(new BoundedChannelOptions(20)
        {
            FullMode = BoundedChannelFullMode.DropWrite
        });
    }

    /// <summary>
    /// Starts the DrainLoop and ReceiveLoop background tasks.
    /// </summary>
    public void StartAsync()
    {
        _drainTask = Task.Run(() => DrainLoop(_cts.Token));
        _receiveTask = Task.Run(() => ReceiveLoop(_cts.Token));
    }

    /// <summary>
    /// Stops both background tasks.
    /// When <paramref name="serverInitiated"/> is false, sends a <c>control_command shutdown</c>
    /// JSON frame before closing the WebSocket.
    ///
    /// Algorithm:
    /// 1. If client-initiated: send control_command shutdown, then wait up to 2 s for session_closed.
    /// 2. Cancel the background loops via the CancellationTokenSource.
    /// 3. Close the WebSocket output gracefully.
    /// 4. Await both background tasks to ensure clean teardown.
    /// </summary>
    /// <param name="serverInitiated">
    /// True if the server triggered the shutdown (no control command needed).
    /// False if the client is initiating the shutdown.
    /// </param>
    public async Task StopAsync(bool serverInitiated)
    {
        if (!serverInitiated)
        {
            await SendControlCommandShutdownAsync();
            await Task.WhenAny(_sessionClosedTcs.Task, Task.Delay(TimeSpan.FromSeconds(2)));
        }

        await _cts.CancelAsync();

        var wsState = _webSocket.State;
        if (wsState == WebSocketState.Open || wsState == WebSocketState.CloseReceived)
        {
            try
            {
                await _webSocket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "Client shutdown", CancellationToken.None);
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "WebSocket close output failed during StopAsync");
            }
        }

        if (_drainTask is not null)
            await _drainTask.ConfigureAwait(false);

        if (_receiveTask is not null)
            await _receiveTask.ConfigureAwait(false);
    }

    /// <summary>
    /// Encodes the audio chunk and enqueues it for transmission.
    /// Drops the frame silently if the channel is full (capacity 20).
    /// Thread-safe; returns immediately without blocking.
    /// </summary>
    /// <param name="sessionId">Session identifier assigned by the server.</param>
    /// <param name="chunkId">Monotonically increasing chunk counter.</param>
    /// <param name="timestamp">Wall-clock timestamp in seconds (Unix epoch, double precision).</param>
    /// <param name="samples">Raw PCM float samples.</param>
    public void SendAudioChunkAsync(string sessionId, int chunkId, double timestamp, float[] samples)
    {
        var pooled = _encoder.Encode(new WsAudioFrame(sessionId, chunkId, timestamp, samples));

        if (!_sendChannel.Writer.TryWrite(pooled))
        {
            pooled.Dispose();
            _logger.LogDebug("Audio send channel full — dropping chunk_id={ChunkId}", chunkId);
        }
    }

    /// <summary>
    /// Sends a pre-formed JSON text frame directly over the WebSocket.
    /// Used by <see cref="RemoteRecognitionPublisher.PongSender"/> to reply to ping messages.
    /// </summary>
    /// <param name="json">The JSON string to send as a WebSocket text frame.</param>
    public async Task SendTextAsync(string json)
    {
        var bytes = Encoding.UTF8.GetBytes(json);
        await _webSocket.SendAsync(bytes, WebSocketMessageType.Text, endOfMessage: true, CancellationToken.None);
    }

    /// <summary>
    /// Reads binary frames from the send channel and writes them to the WebSocket.
    /// Exits when the cancellation token is triggered or the channel is completed.
    /// </summary>
    private async Task DrainLoop(CancellationToken ct)
    {
        try
        {
            await foreach (var pooled in _sendChannel.Reader.ReadAllAsync(ct))
            {
                await _webSocket.SendAsync(pooled.Data, WebSocketMessageType.Binary, endOfMessage: true, ct);
                pooled.Dispose();
            }
        }
        catch (OperationCanceledException)
        {
            _logger.LogDebug("DrainLoop cancelled");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "DrainLoop terminated unexpectedly");
        }
    }

    /// <summary>
    /// Reads text and control frames from the WebSocket and dispatches them.
    /// On receiving a Close frame, transitions the application state to Shutdown.
    ///
    /// Algorithm:
    /// 1. Receive into a fixed 64 KB chunk buffer, reassembling multi-frame messages
    ///    until EndOfMessage is true.
    /// 2. If Text: pass the fully reassembled string to publisher.Dispatch.
    /// 3. If Close: acknowledge, cancel loops, call stateManager.SetState(Shutdown), and exit.
    /// 4. On cancellation or WebSocket error: exit loop cleanly.
    /// </summary>
    private async Task ReceiveLoop(CancellationToken ct)
    {
        var chunk = new byte[65536];

        try
        {
            while (!ct.IsCancellationRequested)
            {
                _messageBuffer.SetLength(0);
                WebSocketReceiveResult result;

                do
                {
                    result = await _webSocket.ReceiveAsync(chunk.AsMemory(), ct);

                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        _logger.LogInformation("WebSocket Close received — transitioning to Shutdown");
                        try { await _webSocket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "Acknowledged", CancellationToken.None); }
                        catch { /* best-effort close handshake */ }
                        await _cts.CancelAsync();
                        _stateManager.SetState(AppState.Shutdown);
                        return;
                    }

                    if (result.Count > 0)
                        _messageBuffer.Write(chunk, 0, result.Count);
                }
                while (!result.EndOfMessage);

                if (result.MessageType == WebSocketMessageType.Text)
                {
                    var text = Encoding.UTF8.GetString(_messageBuffer.GetBuffer(), 0, (int)_messageBuffer.Length);
                    _publisher.Dispatch(text);
                }
            }
        }
        catch (OperationCanceledException)
        {
            _logger.LogDebug("ReceiveLoop cancelled");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "ReceiveLoop terminated unexpectedly");
        }
    }

    private async Task SendControlCommandShutdownAsync()
    {
        try
        {
            var ts = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
            var cmd = new ControlCommand("shutdown", SessionId, ts);
            var json = JsonSerializer.Serialize(cmd, WireTypesJsonContext.Default.ControlCommand);
            await SendTextAsync(json);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to send control_command shutdown");
        }
    }

    /// <inheritdoc/>
    public async ValueTask DisposeAsync()
    {
        await StopAsync(serverInitiated: true);
        _cts.Dispose();
        _webSocket.Dispose();
    }
}
