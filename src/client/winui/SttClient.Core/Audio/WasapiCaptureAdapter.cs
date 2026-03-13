using NAudio.Wave;
using NAudio.CoreAudioApi;

namespace SttClient.Audio;

/// <summary>
/// Production adapter that wraps <see cref="WasapiCapture"/> to implement <see cref="IWaveCapture"/>.
/// Created by <see cref="WasapiAudioSource"/> when no explicit capture device is injected.
/// </summary>
public sealed class WasapiCaptureAdapter : IWaveCapture
{
    private readonly WasapiCapture _capture;

    /// <summary>
    /// Initializes the adapter, requesting 16 kHz mono IEEE-float shared-mode capture.
    /// </summary>
    public WasapiCaptureAdapter()
    {
        _capture = new WasapiCapture();
    }

    /// <inheritdoc/>
    public WaveFormat WaveFormat => _capture.WaveFormat;

    /// <inheritdoc/>
    public event EventHandler<WaveInEventArgs>? DataAvailable
    {
        add => _capture.DataAvailable += value;
        remove => _capture.DataAvailable -= value;
    }

    /// <inheritdoc/>
    public void StartRecording() => _capture.StartRecording();

    /// <inheritdoc/>
    public void StopRecording() => _capture.StopRecording();

    /// <inheritdoc/>
    public void Dispose() => _capture.Dispose();
}
