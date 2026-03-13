using System.Buffers;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using SttClient.Protocol;
using Xunit;

namespace SttClient.Tests.Protocol;

/// <summary>
/// Tests for AudioFrameEncoder binary wire format correctness.
/// Validates header length prefix, JSON key order, PCM payload fidelity,
/// and golden byte sequence for a known input.
/// </summary>
public sealed class AudioFrameEncoderTests
{
    private readonly AudioFrameEncoder _encoder = new(NullLogger<AudioFrameEncoder>.Instance);

    [Fact]
    public void Encode_ProducesCorrectHeaderLengthPrefix()
    {
        var frame = new WsAudioFrame("sess-1", 1, 1000.0, [0.1f]);

        using var pooled = _encoder.Encode(frame);
        var data = pooled.Data.ToArray();

        uint reportedHeaderLen = BinaryPrimitives_ReadUInt32LE(data, 0);
        Assert.True(reportedHeaderLen > 0);
        Assert.Equal((uint)(data.Length - 4 - sizeof(float)), reportedHeaderLen);
    }

    [Fact]
    public void Encode_ProducesJsonKeysInFixedOrder()
    {
        var frame = new WsAudioFrame("sess-2", 7, 2000.5, [0.2f, 0.3f]);

        using var pooled = _encoder.Encode(frame);
        var data = pooled.Data.ToArray();

        uint headerLen = BinaryPrimitives_ReadUInt32LE(data, 0);
        string json = Encoding.UTF8.GetString(data, 4, (int)headerLen);

        var keyOrder = ExtractKeyOrder(json);
        Assert.Equal(["type", "session_id", "chunk_id", "timestamp"], keyOrder);
    }

    [Fact]
    public void Encode_ProducesCorrectFloatPayload()
    {
        float[] audio = [1.0f, -1.0f, 0.5f, 0.0f];
        var frame = new WsAudioFrame("sess-3", 3, 3000.0, audio);

        using var pooled = _encoder.Encode(frame);
        var data = pooled.Data.ToArray();

        uint headerLen = BinaryPrimitives_ReadUInt32LE(data, 0);
        int audioOffset = 4 + (int)headerLen;
        int audioByteLen = data.Length - audioOffset;

        Assert.Equal(audio.Length * sizeof(float), audioByteLen);

        byte[] expectedAudioBytes = new byte[audio.Length * sizeof(float)];
        MemoryMarshal.Cast<float, byte>(audio).CopyTo(expectedAudioBytes);

        Assert.Equal(expectedAudioBytes, data[audioOffset..]);
    }

    [Fact]
    public void Encode_GoldenByteSequence()
    {
        var frame = new WsAudioFrame("test-session", 42, 1735689600.123, [1.0f, -1.0f, 0.5f]);

        using var pooled = _encoder.Encode(frame);
        var actual = pooled.Data.ToArray();

        byte[] golden = BuildGolden("test-session", 42, 1735689600.123, [1.0f, -1.0f, 0.5f]);
        Assert.Equal(golden, actual);
    }

    [Fact]
    public void Encode_KeyOrderDoesNotVaryAcrossRuns()
    {
        var frame = new WsAudioFrame("sess-stable", 99, 9999.9, [0.5f]);

        for (int i = 0; i < 100; i++)
        {
            using var pooled = _encoder.Encode(frame);
            var data = pooled.Data.ToArray();
            uint headerLen = BinaryPrimitives_ReadUInt32LE(data, 0);
            string json = Encoding.UTF8.GetString(data, 4, (int)headerLen);
            var keyOrder = ExtractKeyOrder(json);
            Assert.Equal(["type", "session_id", "chunk_id", "timestamp"], keyOrder);
        }
    }

    [Fact]
    public void PooledFrame_Dispose_DoesNotThrow()
    {
        var frame = new WsAudioFrame("sess-dispose", 0, 0.0, []);
        var pooled = _encoder.Encode(frame);

        var ex = Record.Exception(() =>
        {
            pooled.Dispose();
            pooled.Dispose();
        });

        Assert.Null(ex);
    }

    private static byte[] BuildGolden(string sessionId, int chunkId, double timestamp, float[] audio)
    {
        using var ms = new MemoryStream();
        using var writer = new Utf8JsonWriter(ms);
        writer.WriteStartObject();
        writer.WriteString("type", "audio_chunk");
        writer.WriteString("session_id", sessionId);
        writer.WriteNumber("chunk_id", chunkId);
        writer.WriteNumber("timestamp", timestamp);
        writer.WriteEndObject();
        writer.Flush();
        byte[] headerBytes = ms.ToArray();

        int audioByteLen = audio.Length * sizeof(float);
        byte[] result = new byte[4 + headerBytes.Length + audioByteLen];
        result[0] = (byte)(headerBytes.Length & 0xFF);
        result[1] = (byte)((headerBytes.Length >> 8) & 0xFF);
        result[2] = (byte)((headerBytes.Length >> 16) & 0xFF);
        result[3] = (byte)((headerBytes.Length >> 24) & 0xFF);
        headerBytes.CopyTo(result, 4);
        MemoryMarshal.Cast<float, byte>(audio).CopyTo(result.AsSpan(4 + headerBytes.Length));
        return result;
    }

    private static List<string> ExtractKeyOrder(string json)
    {
        var keys = new List<string>();
        using var doc = JsonDocument.Parse(json);
        foreach (var prop in doc.RootElement.EnumerateObject())
            keys.Add(prop.Name);
        return keys;
    }

    private static uint BinaryPrimitives_ReadUInt32LE(byte[] data, int offset) =>
        (uint)(data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24));
}
