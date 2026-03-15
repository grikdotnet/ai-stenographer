using System.Text.Json;
using Microsoft.Extensions.Logging;
using SttClient.Protocol;
using SttClient.State;

namespace SttClient.Recognition;

/// <summary>
/// Decodes JSON text frames received from the STT server and routes them to the appropriate handler.
/// Handles recognition results (partial and final), session lifecycle events, ping/pong keepalive, and errors.
/// Called by the transport's ReceiveLoop; Dispatch must never throw.
/// </summary>
public sealed class RemoteRecognitionPublisher
{
    private readonly RecognitionResultFanOut _fanOut;
    private readonly AppStateManager _stateManager;
    private readonly ServerMessageDecoder _decoder;
    private readonly ILogger<RemoteRecognitionPublisher> _logger;

    private string? _sessionId;

    /// <summary>
    /// Injected by the transport layer to send pong responses back to the server.
    /// When set, receives a fully-formed JSON pong string and sends it over the connection.
    /// </summary>
    public Func<string, Task>? PongSender { get; set; }

    /// <summary>
    /// Invoked when a <c>session_closed</c> message is received, before state is set to Shutdown.
    /// Used by <see cref="Transport.WsClientTransport"/> to unblock the shutdown wait.
    /// </summary>
    public Action? SessionClosedCallback { get; set; }

    /// <summary>
    /// Initializes the publisher with the fan-out, state manager, and decoder it routes events through.
    /// </summary>
    /// <param name="fanOut">Fan-out that dispatches recognition results to all subscribers.</param>
    /// <param name="stateManager">State machine updated on session lifecycle events.</param>
    /// <param name="decoder">Decodes raw JSON into typed ServerMessage records.</param>
    /// <param name="logger">Logger for warnings and diagnostics.</param>
    public RemoteRecognitionPublisher(
        RecognitionResultFanOut fanOut,
        AppStateManager stateManager,
        ServerMessageDecoder decoder,
        ILogger<RemoteRecognitionPublisher> logger)
    {
        _fanOut = fanOut;
        _stateManager = stateManager;
        _decoder = decoder;
        _logger = logger;
    }

    /// <summary>
    /// Stores the session ID received in the session_created handshake.
    /// Used to validate incoming recognition_result frames.
    /// </summary>
    /// <param name="sessionId">The session identifier assigned by the server.</param>
    public void SetSessionId(string sessionId)
    {
        _sessionId = sessionId;
    }

    /// <summary>
    /// Parses and dispatches one JSON text frame from the server.
    ///
    /// Algorithm:
    /// 1. Decode JSON via ServerMessageDecoder into a typed record.
    /// 2. Route to the appropriate handler via pattern matching.
    /// 3. Swallow all exceptions after logging — never rethrow.
    /// </summary>
    /// <param name="json">Raw JSON string received from the server WebSocket.</param>
    public void Dispatch(string json)
    {
        try
        {
            var message = _decoder.Decode(json);
            switch (message)
            {
                case RecognitionResultMessage r: HandleRecognitionResult(r); break;
                case SessionClosed:              HandleSessionClosed(); break;
                case PingMessage p:              HandlePing(p); break;
                case ErrorMessage e:             HandleError(e); break;
                case null:                       break;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Unexpected error dispatching server message");
        }
    }

    private void HandleRecognitionResult(RecognitionResultMessage r)
    {
        if (r.SessionId != _sessionId)
        {
            _logger.LogWarning(
                "Discarding recognition_result: session_id mismatch (expected {Expected}, got {Got})",
                _sessionId, r.SessionId);
            return;
        }

        var result = new RecognitionResult(
            r.Text, r.StartTime, r.EndTime, r.UtteranceId, r.ChunkIds, r.TokenConfidences);

        if (r.Status == "partial")
            _fanOut.OnPartialUpdate(result);
        else if (r.Status == "final")
            _fanOut.OnFinalization(result);
        else
            _logger.LogWarning("Unrecognized recognition_result status: {Status}", r.Status);
    }

    private void HandleSessionClosed()
    {
        try
        {
            SessionClosedCallback?.Invoke();
            _stateManager.SetState(AppState.Shutdown);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to set Shutdown state on session_closed");
        }
    }

    private void HandlePing(PingMessage p)
    {
        if (PongSender is null)
            return;

        var pong = new PongMessage(_sessionId ?? string.Empty, p.Timestamp);
        var pongJson = JsonSerializer.Serialize(pong, WireTypesJsonContext.Default.PongMessage);
        _ = PongSender(pongJson);
    }

    private void HandleError(ErrorMessage e)
    {
        _logger.LogWarning("Server error [{ErrorCode}]: {Message}", e.ErrorCode, e.Message);
    }
}
