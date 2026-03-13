namespace SttClient.Audio;

/// <summary>
/// Immutable snapshot of a single audio capture event.
/// Contains 512 float32 PCM samples at 16 kHz mono and the wall-clock capture timestamp.
/// </summary>
/// <param name="Samples">512 float32 PCM samples at 16 kHz mono.</param>
/// <param name="Timestamp">Wall-clock capture time (UTC).</param>
public sealed record AudioChunk(float[] Samples, DateTimeOffset Timestamp);
