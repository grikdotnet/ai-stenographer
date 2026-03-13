using NAudio.Wave;

namespace SttClient.Audio;

/// <summary>
/// Abstraction over a NAudio wave capture device (e.g. <c>WasapiCapture</c>).
/// Extracted for testability: production code uses <see cref="WasapiCaptureAdapter"/>;
/// tests supply <c>FakeWaveCapture</c>.
/// </summary>
public interface IWaveCapture : IDisposable
{
    /// <summary>Gets the wave format reported by the capture device.</summary>
    WaveFormat WaveFormat { get; }

    /// <summary>Fired by the device when a buffer of audio data is available.</summary>
    event EventHandler<WaveInEventArgs>? DataAvailable;

    /// <summary>Starts audio capture on the device.</summary>
    void StartRecording();

    /// <summary>Stops audio capture on the device.</summary>
    void StopRecording();
}
