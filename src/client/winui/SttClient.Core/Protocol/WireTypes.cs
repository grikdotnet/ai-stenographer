using System.Text.Json.Serialization;

namespace SttClient.Protocol;

/// <summary>
/// Base record for all messages sent from server to client.
/// Discriminated by the Type field to allow polymorphic deserialization.
/// </summary>
public abstract record ServerMessage(
    [property: JsonPropertyName("type")] string Type
);

/// <summary>
/// Sent by the server immediately after a WebSocket connection is accepted,
/// confirming the session identity and negotiated protocol version.
/// </summary>
public sealed record SessionCreated(
    [property: JsonPropertyName("session_id")] string SessionId,
    [property: JsonPropertyName("protocol_version")] string ProtocolVersion,
    [property: JsonPropertyName("server_time")] double ServerTime
) : ServerMessage("session_created");

/// <summary>
/// Carries a partial or final ASR recognition result for a speech segment,
/// including timing, confidence scores, and chunk tracking information.
/// </summary>
public sealed record RecognitionResultMessage(
    [property: JsonPropertyName("session_id")] string SessionId,
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("start_time")] double StartTime,
    [property: JsonPropertyName("end_time")] double EndTime,
    [property: JsonPropertyName("chunk_ids")] int[] ChunkIds,
    [property: JsonPropertyName("utterance_id")] int? UtteranceId,
    [property: JsonPropertyName("confidence")] double? Confidence,
    [property: JsonPropertyName("token_confidences")] double[]? TokenConfidences,
    [property: JsonPropertyName("audio_rms")] double? AudioRms,
    [property: JsonPropertyName("confidence_variance")] double? ConfidenceVariance
) : ServerMessage("recognition_result");

/// <summary>
/// Sent by the server when a session is terminated, carrying the reason
/// and an optional human-readable message for diagnostics.
/// </summary>
public sealed record SessionClosed(
    [property: JsonPropertyName("session_id")] string SessionId,
    [property: JsonPropertyName("reason")] string Reason,
    [property: JsonPropertyName("message")] string? Message
) : ServerMessage("session_closed");

/// <summary>
/// Sent by the server when a recoverable or fatal protocol or processing error occurs.
/// Fatal errors indicate the session cannot continue.
/// </summary>
public sealed record ErrorMessage(
    [property: JsonPropertyName("session_id")] string SessionId,
    [property: JsonPropertyName("error_code")] string ErrorCode,
    [property: JsonPropertyName("message")] string Message,
    [property: JsonPropertyName("fatal")] bool Fatal
) : ServerMessage("error");

/// <summary>
/// Sent by the server to measure round-trip latency. The client should
/// respond with a pong carrying the same timestamp.
/// </summary>
public sealed record PingMessage(
    [property: JsonPropertyName("timestamp")] double Timestamp
) : ServerMessage("ping");

/// <summary>
/// Binary audio frame sent from client to server. Contains a session identifier,
/// monotonically increasing chunk counter, wall-clock timestamp, and raw PCM samples.
/// </summary>
public sealed record WsAudioFrame(
    [property: JsonPropertyName("session_id")] string SessionId,
    [property: JsonPropertyName("chunk_id")] int ChunkId,
    [property: JsonPropertyName("timestamp")] double Timestamp,
    float[] Audio
);
