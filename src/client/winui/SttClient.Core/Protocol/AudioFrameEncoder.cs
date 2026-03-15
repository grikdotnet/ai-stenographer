using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace SttClient.Protocol;

/// <summary>
/// Encodes WsAudioFrame values into the binary wire format expected by the STT server.
/// Uses ArrayPool to avoid per-frame heap allocations at 31 frames/second sustained throughput.
///
/// Binary layout:
///   [0..3]          uint32 LE — byte length of the UTF-8 JSON header
///   [4..4+hlen)     UTF-8 JSON header with keys in fixed order: type, session_id, chunk_id, timestamp
///   [4+hlen..]      float32 LE PCM samples (raw MemoryMarshal cast, no conversion)
/// </summary>
public sealed class AudioFrameEncoder
{
    private readonly ILogger<AudioFrameEncoder> _logger;

    public AudioFrameEncoder(ILogger<AudioFrameEncoder> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Encodes a WsAudioFrame into a pooled binary buffer ready for WebSocket transmission.
    /// </summary>
    /// <param name="frame">The audio frame to encode.</param>
    /// <returns>
    /// A PooledFrame whose Data slice contains the complete binary message.
    /// The caller must dispose the PooledFrame to return the rented array to the pool.
    /// </returns>
    public PooledFrame Encode(WsAudioFrame frame)
    {
        byte[] headerBytes = BuildHeader(frame);
        int headerLen = headerBytes.Length;
        int audioByteLen = frame.Audio.Length * sizeof(float);
        int totalLen = 4 + headerLen + audioByteLen;

        byte[] rented = ArrayPool<byte>.Shared.Rent(totalLen);

        BinaryPrimitives.WriteUInt32LittleEndian(rented, (uint)headerLen);
        headerBytes.CopyTo(rented, 4);
        MemoryMarshal.Cast<float, byte>(frame.Audio).CopyTo(rented.AsSpan(4 + headerLen));

        return new PooledFrame(rented, totalLen);
    }

    private static byte[] BuildHeader(WsAudioFrame frame)
    {
        using var ms = new System.IO.MemoryStream();
        using var writer = new Utf8JsonWriter(ms);

        writer.WriteStartObject();
        writer.WriteString("type", "audio_chunk");
        writer.WriteString("session_id", frame.SessionId);
        writer.WriteNumber("chunk_id", frame.ChunkId);
        writer.WriteNumber("timestamp", frame.Timestamp);
        writer.WriteEndObject();
        writer.Flush();

        return ms.ToArray();
    }

    /// <summary>
    /// A rented byte array slice wrapping an encoded audio frame.
    /// Implements IDisposable to return the underlying array to ArrayPool on release.
    /// Callers must dispose after the WebSocket send completes.
    /// </summary>
    public sealed class PooledFrame : IDisposable
    {
        private byte[]? _rentedArray;
        private bool _disposed;

        /// <summary>
        /// Gets the exact slice of the rented array containing the encoded frame.
        /// Valid only until Dispose is called.
        /// </summary>
        public Memory<byte> Data { get; }

        internal PooledFrame(byte[] rentedArray, int length)
        {
            _rentedArray = rentedArray;
            Data = rentedArray.AsMemory(0, length);
        }

        /// <summary>
        /// Returns the rented array to ArrayPool. Safe to call multiple times.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (_rentedArray is not null)
            {
                ArrayPool<byte>.Shared.Return(_rentedArray);
                _rentedArray = null;
            }
        }
    }
}
