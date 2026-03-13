using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace SttClient.Protocol;

/// <summary>
/// Decodes JSON strings received from the STT server into typed ServerMessage records.
/// Peeks at the "type" discriminator field to route deserialization without allocating
/// intermediate objects. Returns null for unknown types or malformed input instead of throwing.
/// </summary>
public sealed class ServerMessageDecoder
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    private readonly ILogger<ServerMessageDecoder> _logger;

    public ServerMessageDecoder(ILogger<ServerMessageDecoder> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Decodes a JSON string from the server into the appropriate ServerMessage subtype.
    /// </summary>
    /// <param name="json">Raw JSON text received over WebSocket.</param>
    /// <returns>
    /// A concrete ServerMessage subtype, or null if the JSON is malformed,
    /// the "type" field is absent, or the type value is unrecognized.
    /// </returns>
    public ServerMessage? Decode(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            _logger.LogWarning("Received empty or whitespace JSON — ignoring");
            return null;
        }

        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(json);
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "Failed to parse server message JSON");
            return null;
        }

        using (document)
        {
            if (!document.RootElement.TryGetProperty("type", out JsonElement typeElement))
            {
                _logger.LogWarning("Server message missing 'type' field");
                return null;
            }

            string? type = typeElement.GetString();
            return type switch
            {
                "session_created" => Deserialize<SessionCreated>(json),
                "recognition_result" => Deserialize<RecognitionResultMessage>(json),
                "session_closed" => Deserialize<SessionClosed>(json),
                "error" => Deserialize<ErrorMessage>(json),
                "ping" => Deserialize<PingMessage>(json),
                _ => LogUnknown(type),
            };
        }
    }

    private ServerMessage? Deserialize<T>(string json) where T : ServerMessage
    {
        try
        {
            return JsonSerializer.Deserialize<T>(json, Options);
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "Failed to deserialize server message as {Type}", typeof(T).Name);
            return null;
        }
    }

    private ServerMessage? LogUnknown(string? type)
    {
        _logger.LogWarning("Unknown server message type: {Type}", type);
        return null;
    }
}
