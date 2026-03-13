using Microsoft.Extensions.Logging.Abstractions;
using SttClient.Protocol;
using Xunit;

namespace SttClient.Tests.Protocol;

/// <summary>
/// Tests for ServerMessageDecoder JSON deserialization into typed ServerMessage records.
/// Covers all known message types, null-safety for unknown types, and malformed input handling.
/// </summary>
public sealed class ServerMessageDecoderTests
{
    private readonly ServerMessageDecoder _decoder = new(NullLogger<ServerMessageDecoder>.Instance);

    [Fact]
    public void Decode_SessionCreated_ReturnsCorrectRecord()
    {
        const string json = """
            {
                "type": "session_created",
                "session_id": "abc-123",
                "protocol_version": "1.0",
                "server_time": 1735689600.0
            }
            """;

        var result = _decoder.Decode(json);

        var msg = Assert.IsType<SessionCreated>(result);
        Assert.Equal("session_created", msg.Type);
        Assert.Equal("abc-123", msg.SessionId);
        Assert.Equal("1.0", msg.ProtocolVersion);
        Assert.Equal(1735689600.0, msg.ServerTime);
    }

    [Fact]
    public void Decode_RecognitionResult_Partial_ReturnsCorrectRecord()
    {
        const string json = """
            {
                "type": "recognition_result",
                "session_id": "abc-123",
                "status": "partial",
                "text": "hello world",
                "start_time": 0.0,
                "end_time": 1.5,
                "chunk_ids": [1, 2, 3],
                "utterance_id": null,
                "confidence": null,
                "token_confidences": null,
                "audio_rms": 0.12,
                "confidence_variance": null
            }
            """;

        var result = _decoder.Decode(json);

        var msg = Assert.IsType<RecognitionResultMessage>(result);
        Assert.Equal("recognition_result", msg.Type);
        Assert.Equal("abc-123", msg.SessionId);
        Assert.Equal("partial", msg.Status);
        Assert.Equal("hello world", msg.Text);
        Assert.Equal(0.0, msg.StartTime);
        Assert.Equal(1.5, msg.EndTime);
        Assert.Equal([1, 2, 3], msg.ChunkIds);
        Assert.Null(msg.UtteranceId);
        Assert.Null(msg.Confidence);
        Assert.Null(msg.TokenConfidences);
        Assert.Equal(0.12, msg.AudioRms);
        Assert.Null(msg.ConfidenceVariance);
    }

    [Fact]
    public void Decode_RecognitionResult_Final_ReturnsCorrectRecord()
    {
        const string json = """
            {
                "type": "recognition_result",
                "session_id": "abc-123",
                "status": "final",
                "text": "the quick brown fox",
                "start_time": 0.5,
                "end_time": 2.8,
                "chunk_ids": [4, 5, 6, 7],
                "utterance_id": 3,
                "confidence": 0.97,
                "token_confidences": [0.98, 0.95, 0.99, 0.96],
                "audio_rms": 0.23,
                "confidence_variance": 0.002
            }
            """;

        var result = _decoder.Decode(json);

        var msg = Assert.IsType<RecognitionResultMessage>(result);
        Assert.Equal("final", msg.Status);
        Assert.Equal("the quick brown fox", msg.Text);
        Assert.Equal(3, msg.UtteranceId);
        Assert.Equal(0.97, msg.Confidence);
        Assert.NotNull(msg.TokenConfidences);
        Assert.Equal(4, msg.TokenConfidences!.Length);
        Assert.Equal(0.95, msg.TokenConfidences[1]);
        Assert.Equal(0.002, msg.ConfidenceVariance);
    }

    [Fact]
    public void Decode_SessionClosed_ReturnsCorrectRecord()
    {
        const string json = """
            {
                "type": "session_closed",
                "session_id": "abc-123",
                "reason": "client_disconnect",
                "message": "Normal closure"
            }
            """;

        var result = _decoder.Decode(json);

        var msg = Assert.IsType<SessionClosed>(result);
        Assert.Equal("session_closed", msg.Type);
        Assert.Equal("abc-123", msg.SessionId);
        Assert.Equal("client_disconnect", msg.Reason);
        Assert.Equal("Normal closure", msg.Message);
    }

    [Fact]
    public void Decode_ErrorMessage_ReturnsCorrectRecord()
    {
        const string json = """
            {
                "type": "error",
                "session_id": "abc-123",
                "error_code": "OVERLOAD",
                "message": "Server is overloaded, retry later",
                "fatal": false
            }
            """;

        var result = _decoder.Decode(json);

        var msg = Assert.IsType<ErrorMessage>(result);
        Assert.Equal("error", msg.Type);
        Assert.Equal("abc-123", msg.SessionId);
        Assert.Equal("OVERLOAD", msg.ErrorCode);
        Assert.Equal("Server is overloaded, retry later", msg.Message);
        Assert.False(msg.Fatal);
    }

    [Fact]
    public void Decode_PingMessage_ReturnsCorrectRecord()
    {
        const string json = """
            {
                "type": "ping",
                "timestamp": 1735689600.456
            }
            """;

        var result = _decoder.Decode(json);

        var msg = Assert.IsType<PingMessage>(result);
        Assert.Equal("ping", msg.Type);
        Assert.Equal(1735689600.456, msg.Timestamp);
    }

    [Fact]
    public void Decode_UnknownType_ReturnsNull()
    {
        const string json = """
            {
                "type": "teleportation_request",
                "destination": "Mars"
            }
            """;

        var result = _decoder.Decode(json);

        Assert.Null(result);
    }

    [Fact]
    public void Decode_MalformedJson_ReturnsNull_NoException()
    {
        const string malformed = "{ this is not valid JSON :::";

        var ex = Record.Exception(() => _decoder.Decode(malformed));
        Assert.Null(ex);

        var result = _decoder.Decode(malformed);
        Assert.Null(result);
    }

    [Fact]
    public void Decode_EmptyString_ReturnsNull_NoException()
    {
        var ex = Record.Exception(() => _decoder.Decode(string.Empty));
        Assert.Null(ex);

        var result = _decoder.Decode(string.Empty);
        Assert.Null(result);
    }
}
