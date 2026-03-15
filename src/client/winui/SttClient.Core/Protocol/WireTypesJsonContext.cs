using System.Text.Json.Serialization;

namespace SttClient.Protocol;

/// <summary>
/// Source-generated JSON serialization context for all server-to-client wire types.
/// Enables trim-safe deserialization without runtime reflection.
/// </summary>
[JsonSerializable(typeof(SessionCreated))]
[JsonSerializable(typeof(RecognitionResultMessage))]
[JsonSerializable(typeof(SessionClosed))]
[JsonSerializable(typeof(ErrorMessage))]
[JsonSerializable(typeof(PingMessage))]
[JsonSerializable(typeof(ControlCommand))]
[JsonSerializable(typeof(PongMessage))]
[JsonSourceGenerationOptions(PropertyNameCaseInsensitive = true)]
internal sealed partial class WireTypesJsonContext : JsonSerializerContext
{
}
