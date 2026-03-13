using System.Text.Json;
using Microsoft.Extensions.Logging;
using SttClient.State;

namespace SttClient.Recognition;

/// <summary>
/// Decodes JSON text frames received from the STT server and routes them to the appropriate handler.
/// Handles recognition results (partial and final), session lifecycle events, ping/pong keepalive, and errors.
/// Called by the transport's ReceiveLoop; Dispatch must never throw.
/// </summary>
public sealed class RemoteRecognitionPublisher
{
    private static readonly JsonSerializerOptions SnakeCase = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    private readonly RecognitionResultFanOut _fanOut;
    private readonly AppStateManager _stateManager;
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
    /// Initializes the publisher with the fan-out and state manager it routes events to.
    /// </summary>
    /// <param name="fanOut">Fan-out that dispatches recognition results to all subscribers.</param>
    /// <param name="stateManager">State machine updated on session lifecycle events.</param>
    /// <param name="logger">Logger for warnings and diagnostics.</param>
    public RemoteRecognitionPublisher(
        RecognitionResultFanOut fanOut,
        AppStateManager stateManager,
        ILogger<RemoteRecognitionPublisher> logger)
    {
        _fanOut = fanOut;
        _stateManager = stateManager;
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
    /// 1. Parse JSON and peek at "type" field.
    /// 2. Route to the appropriate handler based on type.
    /// 3. Swallow all exceptions after logging — never rethrow.
    /// </summary>
    /// <param name="json">Raw JSON string received from the server WebSocket.</param>
    public void Dispatch(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (!root.TryGetProperty("type", out var typeProp))
            {
                _logger.LogWarning("Received message with no 'type' field");
                return;
            }

            var type = typeProp.GetString();

            switch (type)
            {
                case "recognition_result":
                    HandleRecognitionResult(root);
                    break;
                case "session_closed":
                    HandleSessionClosed();
                    break;
                case "ping":
                    HandlePing(root);
                    break;
                case "error":
                    HandleError(root);
                    break;
                default:
                    _logger.LogDebug("Ignoring unknown message type: {Type}", type);
                    break;
            }
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "Received malformed JSON frame");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Unexpected error dispatching server message");
        }
    }

    private void HandleRecognitionResult(JsonElement root)
    {
        if (root.TryGetProperty("session_id", out var sessionIdProp))
        {
            var incomingSessionId = sessionIdProp.GetString();
            if (incomingSessionId != _sessionId)
            {
                _logger.LogWarning(
                    "Discarding recognition_result: session_id mismatch (expected {Expected}, got {Got})",
                    _sessionId, incomingSessionId);
                return;
            }
        }

        var status = root.TryGetProperty("status", out var statusProp) ? statusProp.GetString() : null;
        var result = DeserializeRecognitionResult(root);

        if (status == "partial")
            _fanOut.OnPartialUpdate(result);
        else if (status == "final")
            _fanOut.OnFinalization(result);
        else
            _logger.LogWarning("Unrecognized recognition_result status: {Status}", status);
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

    private void HandlePing(JsonElement root)
    {
        if (PongSender is null)
            return;

        var timestamp = root.TryGetProperty("timestamp", out var tsProp) ? tsProp.GetDouble() : 0.0;
        var pongJson = JsonSerializer.Serialize(new
        {
            type = "pong",
            session_id = _sessionId ?? string.Empty,
            timestamp
        });

        _ = PongSender(pongJson);
    }

    private void HandleError(JsonElement root)
    {
        var errorCode = root.TryGetProperty("error_code", out var codeProp) ? codeProp.GetString() : "unknown";
        var message = root.TryGetProperty("message", out var msgProp) ? msgProp.GetString() : string.Empty;
        _logger.LogWarning("Server error [{ErrorCode}]: {Message}", errorCode, message);
    }

    private static RecognitionResult DeserializeRecognitionResult(JsonElement root)
    {
        var text = root.TryGetProperty("text", out var textProp) ? textProp.GetString() ?? string.Empty : string.Empty;
        var startTime = root.TryGetProperty("start_time", out var stProp) ? stProp.GetDouble() : 0.0;
        var endTime = root.TryGetProperty("end_time", out var etProp) ? etProp.GetDouble() : 0.0;
        int? utteranceId = root.TryGetProperty("utterance_id", out var uidProp) && uidProp.ValueKind != JsonValueKind.Null
            ? uidProp.GetInt32()
            : null;

        int[] chunkIds = [];
        if (root.TryGetProperty("chunk_ids", out var chunksProp) && chunksProp.ValueKind == JsonValueKind.Array)
            chunkIds = chunksProp.Deserialize<int[]>(SnakeCase) ?? [];

        double[]? tokenConfidences = null;
        if (root.TryGetProperty("token_confidences", out var tcProp) && tcProp.ValueKind == JsonValueKind.Array)
            tokenConfidences = tcProp.Deserialize<double[]>(SnakeCase);

        return new RecognitionResult(text, startTime, endTime, utteranceId, chunkIds, tokenConfidences);
    }
}
