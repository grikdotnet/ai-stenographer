namespace SttClient.Audio;

/// <summary>
/// Abstraction over an audio capture device.
/// Raises <see cref="ChunkReady"/> when a 512-sample float32 PCM chunk is available.
/// </summary>
public interface IAudioSource
{
    /// <summary>
    /// Fired on the capture thread each time a 512-sample chunk is ready.
    /// Handlers must be non-blocking (&lt;1 ms) to avoid dropouts.
    /// </summary>
    event Action<AudioChunk> ChunkReady;

    /// <summary>Begins audio capture and starts raising <see cref="ChunkReady"/>.</summary>
    void Start();

    /// <summary>Stops audio capture. No further <see cref="ChunkReady"/> events are raised.</summary>
    void Stop();
}
