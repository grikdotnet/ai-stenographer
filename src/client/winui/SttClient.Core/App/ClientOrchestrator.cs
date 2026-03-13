using System.Net.WebSockets;
using Microsoft.Extensions.Logging;
using SttClient.Audio;
using SttClient.Protocol;
using SttClient.Recognition;
using SttClient.State;
using SttClient.Transport;

namespace SttClient.Orchestration;

/// <summary>
/// Wires all client components into a running session.
///
/// Responsibilities:
/// - Connects to the STT server WebSocket, validates the <c>session_created</c> handshake,
///   and starts the audio pipeline.
/// - Coordinates graceful shutdown (client- or server-initiated).
/// - Exposes <see cref="PauseAsync"/> and <see cref="ResumeAsync"/> for UI controls.
///
/// Algorithm for <see cref="ConnectAsync"/>:
/// 1. Connect the WebSocket within a 10-second timeout.
/// 2. Receive and decode the first frame; assert it is <c>session_created</c>.
/// 3. Validate protocol version is <c>v1</c>; on mismatch close with code 1002 and throw.
/// 4. Store session_id in transport and publisher; wire pong sender.
/// 5. Start DrainLoop + ReceiveLoop.
/// 6. Transition state machine to Running → audio capture begins.
/// </summary>
public sealed class ClientOrchestrator : IAsyncDisposable
{
    private readonly string _serverUrl;
    private readonly AppStateManager _stateManager;
    private readonly RemoteRecognitionPublisher _publisher;
    private readonly AudioFrameEncoder _encoder;
    private readonly IAudioSource _audioSource;
    private readonly ILoggerFactory _loggerFactory;
    private readonly ILogger<ClientOrchestrator> _logger;

    private WsClientTransport? _transport;
    private int _chunkId;
    private bool _clientShutdownInitiated;

    /// <summary>
    /// Initializes a new <see cref="ClientOrchestrator"/>.
    /// </summary>
    /// <param name="serverUrl">WebSocket server URL (must use ws:// scheme).</param>
    /// <param name="stateManager">Application state machine.</param>
    /// <param name="publisher">Dispatches incoming server messages.</param>
    /// <param name="encoder">Encodes audio frames for the wire.</param>
    /// <param name="audioSource">Microphone capture source.</param>
    /// <param name="loggerFactory">Factory used to create per-component loggers.</param>
    public ClientOrchestrator(
        string serverUrl,
        AppStateManager stateManager,
        RemoteRecognitionPublisher publisher,
        AudioFrameEncoder encoder,
        IAudioSource audioSource,
        ILoggerFactory loggerFactory)
    {
        _serverUrl = serverUrl;
        _stateManager = stateManager;
        _publisher = publisher;
        _encoder = encoder;
        _audioSource = audioSource;
        _loggerFactory = loggerFactory;
        _logger = loggerFactory.CreateLogger<ClientOrchestrator>();

        _stateManager.AddObserver(OnStateChanged);
    }

    /// <summary>Gets the error message if startup failed, or null on success.</summary>
    public string? StartupError { get; private set; }

    /// <summary>
    /// Connects to the server, validates the handshake, and starts the audio pipeline.
    /// Throws <see cref="OrchestratorStartupException"/> on protocol errors.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token (10-second timeout recommended).</param>
    public async Task ConnectAsync(CancellationToken cancellationToken = default)
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cts.CancelAfter(TimeSpan.FromSeconds(10));

        var ws = new ClientWebSocket();

        try
        {
            _logger.LogInformation("Connecting to {Url}", _serverUrl);
            await ws.ConnectAsync(new Uri(_serverUrl), cts.Token);
        }
        catch (OperationCanceledException)
        {
            ws.Dispose();
            throw new OrchestratorStartupException("Connection timed out.");
        }
        catch (Exception ex)
        {
            ws.Dispose();
            throw new OrchestratorStartupException($"Connection failed: {ex.Message}", ex);
        }

        var sessionCreated = await ReceiveSessionCreatedAsync(ws, cts.Token);

        if (sessionCreated.ProtocolVersion != "v1")
        {
            _logger.LogError("Protocol version mismatch: expected v1, got {Version}", sessionCreated.ProtocolVersion);
            await ws.CloseAsync(WebSocketCloseStatus.ProtocolError, "Unsupported protocol version", CancellationToken.None);
            ws.Dispose();
            throw new OrchestratorStartupException(
                $"Unsupported server protocol version '{sessionCreated.ProtocolVersion}'. This client requires protocol v1.");
        }

        var sessionId = sessionCreated.SessionId;
        _logger.LogInformation("Session created: {SessionId}", sessionId);

        var wsAdapter = new ClientWebSocketAdapter(ws);
        _transport = new WsClientTransport(wsAdapter, _encoder, _publisher, _stateManager,
            _loggerFactory.CreateLogger<WsClientTransport>());
        _transport.SessionId = sessionId;

        _publisher.SetSessionId(sessionId);
        _publisher.PongSender = json => _transport.SendTextAsync(json);
        _publisher.SessionClosedCallback = _transport.SignalSessionClosed;

        _transport.StartAsync();

        _audioSource.ChunkReady += OnChunkReady;
        _stateManager.SetState(AppState.Running);

        _logger.LogInformation("Orchestrator running");
    }

    /// <summary>Pauses audio capture (Running → Paused).</summary>
    public void Pause()
    {
        _stateManager.SetState(AppState.Paused);
    }

    /// <summary>Resumes audio capture (Paused → Running).</summary>
    public void Resume()
    {
        _stateManager.SetState(AppState.Running);
    }

    /// <summary>Initiates client-side shutdown: sends control_command, closes WebSocket.</summary>
    public async Task StopAsync()
    {
        _clientShutdownInitiated = true;
        _audioSource.ChunkReady -= OnChunkReady;

        if (_transport is not null)
            await _transport.StopAsync(serverInitiated: false);

        _stateManager.SetState(AppState.Shutdown);
    }

    /// <inheritdoc/>
    public async ValueTask DisposeAsync()
    {
        if (_transport is not null)
            await _transport.DisposeAsync();
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private async Task<SessionCreated> ReceiveSessionCreatedAsync(ClientWebSocket ws, CancellationToken ct)
    {
        var buffer = new byte[4096];
        ValueWebSocketReceiveResult result;

        try
        {
            result = await ws.ReceiveAsync(buffer.AsMemory(), ct);
        }
        catch (OperationCanceledException)
        {
            ws.Dispose();
            throw new OrchestratorStartupException("Timed out waiting for session_created from server.");
        }

        if (result.MessageType != WebSocketMessageType.Text)
        {
            ws.Dispose();
            throw new OrchestratorStartupException("Expected text frame for session_created handshake.");
        }

        var json = System.Text.Encoding.UTF8.GetString(buffer, 0, result.Count);
        var decoder = new ServerMessageDecoder(Microsoft.Extensions.Logging.Abstractions.NullLogger<ServerMessageDecoder>.Instance);
        var message = decoder.Decode(json);

        if (message is not SessionCreated sessionCreated)
        {
            ws.Dispose();
            throw new OrchestratorStartupException($"Expected session_created, got: {message?.Type ?? "null"}");
        }

        return sessionCreated;
    }

    private void OnChunkReady(AudioChunk chunk)
    {
        if (_transport is null) return;

        var chunkId = Interlocked.Increment(ref _chunkId);
        var timestamp = chunk.Timestamp.ToUnixTimeMilliseconds();
        _transport.SendAudioChunkAsync(_transport.SessionId, chunkId, timestamp, chunk.Samples);
    }

    private void OnStateChanged(AppState oldState, AppState newState)
    {
        if (newState == AppState.Running && oldState != AppState.Paused)
            _audioSource.Start();

        if (newState == AppState.Shutdown)
        {
            _audioSource.Stop();
            _audioSource.ChunkReady -= OnChunkReady;

            if (!_clientShutdownInitiated && _transport is not null)
                _ = _transport.StopAsync(serverInitiated: true);
        }
    }
}

/// <summary>Exception thrown when the orchestrator cannot establish a valid session.</summary>
public sealed class OrchestratorStartupException : Exception
{
    /// <summary>Initializes a new <see cref="OrchestratorStartupException"/> with a message.</summary>
    public OrchestratorStartupException(string message) : base(message) { }

    /// <summary>Initializes a new <see cref="OrchestratorStartupException"/> with a message and inner exception.</summary>
    public OrchestratorStartupException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>Adapts <see cref="ClientWebSocket"/> to <see cref="IWebSocket"/>.</summary>
internal sealed class ClientWebSocketAdapter : IWebSocket
{
    private readonly ClientWebSocket _ws;

    public ClientWebSocketAdapter(ClientWebSocket ws) => _ws = ws;

    public WebSocketState State => _ws.State;

    public Task SendAsync(ReadOnlyMemory<byte> buffer, WebSocketMessageType messageType, bool endOfMessage, CancellationToken ct) =>
        _ws.SendAsync(buffer, messageType, endOfMessage, ct).AsTask();

    public async Task<WebSocketReceiveResult> ReceiveAsync(Memory<byte> buffer, CancellationToken ct)
    {
        var result = await _ws.ReceiveAsync(buffer, ct);
        return new WebSocketReceiveResult(result.Count, result.MessageType, result.EndOfMessage);
    }

    public Task CloseAsync(WebSocketCloseStatus closeStatus, string? statusDescription, CancellationToken ct) =>
        _ws.CloseAsync(closeStatus, statusDescription, ct);

    public Task CloseOutputAsync(WebSocketCloseStatus closeStatus, string? statusDescription, CancellationToken ct) =>
        _ws.CloseOutputAsync(closeStatus, statusDescription, ct);

    public void Dispose() => _ws.Dispose();
}

