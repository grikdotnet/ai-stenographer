using Microsoft.Extensions.Logging;
using NAudio.Wave;
using SttClient.State;
using System.Runtime.InteropServices;

namespace SttClient.Audio;

/// <summary>
/// Captures audio via a <see cref="IWaveCapture"/> device and raises <see cref="ChunkReady"/>
/// with fixed 512-sample float32 PCM buffers at 16 kHz mono.
///
/// If the capture device reports a format other than 16 kHz mono IEEE-float, the raw bytes are
/// piped through a <see cref="MediaFoundationResampler"/> before chunking, so callers always
/// receive normalised 16 kHz float32 data.
///
/// Registers with <see cref="AppStateManager"/> to react to state transitions:
/// <list type="bullet">
///   <item>Running → Paused: stops capture (releases microphone).</item>
///   <item>Paused  → Running: resumes capture.</item>
///   <item>Any     → Shutdown: stops capture permanently.</item>
/// </list>
/// </summary>
public sealed class WasapiAudioSource : IAudioSource, IDisposable
{
    private const int TargetSampleRate = 16000;
    private const int TargetChunkSamples = 512;

    private readonly IWaveCapture _capture;
    private readonly AppStateManager _stateManager;
    private readonly ILogger<WasapiAudioSource> _logger;

    private readonly bool _needsResampling;
    private readonly IWaveProvider? _resamplerProvider;
    private readonly RawSourceWaveStream? _rawSource;

    private readonly float[] _accumulator = new float[TargetChunkSamples * 4];
    private int _accumulatedSamples;

    private bool _active;
    private bool _shutdown;

    /// <inheritdoc/>
    public event Action<AudioChunk>? ChunkReady;

    /// <summary>
    /// Initializes the audio source with an injected capture device.
    /// Determines at construction time whether resampling is needed.
    /// </summary>
    /// <param name="capture">Capture device (WASAPI in production, fake in tests).</param>
    /// <param name="stateManager">Application state machine; observer registered here.</param>
    /// <param name="logger">Logger for diagnostics.</param>
    public WasapiAudioSource(
        IWaveCapture capture,
        AppStateManager stateManager,
        ILogger<WasapiAudioSource> logger)
    {
        _capture = capture;
        _stateManager = stateManager;
        _logger = logger;

        var targetFormat = WaveFormat.CreateIeeeFloatWaveFormat(TargetSampleRate, 1);
        _needsResampling = !capture.WaveFormat.Equals(targetFormat);

        if (_needsResampling)
        {
            _logger.LogInformation(
                "Device format {Format} differs from target; MediaFoundationResampler inserted",
                capture.WaveFormat);

            _rawSource = new RawSourceWaveStream(Stream.Null, capture.WaveFormat);
            _resamplerProvider = new MediaFoundationResampler(_rawSource, targetFormat)
            {
                ResamplerQuality = 60
            };
        }

        _capture.DataAvailable += OnDataAvailable;
        _stateManager.AddObserver(OnStateChange);
    }

    /// <inheritdoc/>
    public void Start()
    {
        if (_shutdown) return;
        _active = true;
        _capture.StartRecording();
        _logger.LogInformation("Audio capture started");
    }

    /// <inheritdoc/>
    public void Stop()
    {
        _active = false;
        _capture.StopRecording();
        _logger.LogInformation("Audio capture stopped");
    }

    /// <summary>Releases the capture device.</summary>
    public void Dispose()
    {
        _capture.DataAvailable -= OnDataAvailable;
        _resamplerProvider?.Let(r => (r as IDisposable)?.Dispose());
        _rawSource?.Dispose();
        _capture.Dispose();
    }

    /// <summary>
    /// Handles raw audio bytes from the capture device.
    ///
    /// Algorithm:
    /// 1. If resampling is needed, feed bytes into RawSourceWaveStream and drain through MediaFoundationResampler to get 16 kHz float32 bytes.
    /// 2. Convert bytes to float32 samples via MemoryMarshal.
    /// 3. Append to accumulator; emit one ChunkReady per full 512-sample block.
    /// </summary>
    private void OnDataAvailable(object? sender, WaveInEventArgs e)
    {
        if (!_active) return;

        ReadOnlySpan<byte> inputBytes = e.Buffer.AsSpan(0, e.BytesRecorded);

        if (_needsResampling)
        {
            ProcessWithResampling(inputBytes);
        }
        else
        {
            ProcessFloat32Bytes(inputBytes);
        }
    }

    private void ProcessFloat32Bytes(ReadOnlySpan<byte> bytes)
    {
        var samples = MemoryMarshal.Cast<byte, float>(bytes);
        foreach (var sample in samples)
        {
            _accumulator[_accumulatedSamples++] = sample;
            if (_accumulatedSamples == TargetChunkSamples)
                FlushChunk();
        }
    }

    private void ProcessWithResampling(ReadOnlySpan<byte> nativeBytes)
    {
        var byteArray = nativeBytes.ToArray();
        _rawSource!.Rewind(byteArray);

        var outputBuffer = new byte[TargetChunkSamples * sizeof(float) * 4];
        int bytesRead;

        while ((bytesRead = _resamplerProvider!.Read(outputBuffer, 0, outputBuffer.Length)) > 0)
        {
            ProcessFloat32Bytes(outputBuffer.AsSpan(0, bytesRead));
        }
    }

    private void FlushChunk()
    {
        var samples = new float[TargetChunkSamples];
        Array.Copy(_accumulator, samples, TargetChunkSamples);
        _accumulatedSamples = 0;

        var chunk = new AudioChunk(samples, DateTimeOffset.UtcNow);
        try
        {
            ChunkReady?.Invoke(chunk);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ChunkReady handler threw");
        }
    }

    private void OnStateChange(AppState oldState, AppState newState)
    {
        if (newState == AppState.Shutdown)
        {
            _shutdown = true;
            Stop();
            return;
        }

        if (newState == AppState.Paused)
        {
            Stop();
            return;
        }

        if (oldState == AppState.Paused && newState == AppState.Running)
        {
            Start();
        }
    }
}

/// <summary>
/// Rewindable <see cref="WaveStream"/> that wraps a byte array in memory,
/// used to feed raw capture bytes into <see cref="MediaFoundationResampler"/>.
/// </summary>
internal sealed class RawSourceWaveStream : WaveStream
{
    private MemoryStream _inner;

    public RawSourceWaveStream(Stream ignored, WaveFormat format)
    {
        WaveFormat = format;
        _inner = new MemoryStream();
    }

    public override WaveFormat WaveFormat { get; }
    public override long Length => _inner.Length;
    public override long Position
    {
        get => _inner.Position;
        set => _inner.Position = value;
    }

    public override int Read(byte[] buffer, int offset, int count) =>
        _inner.Read(buffer, offset, count);

    /// <summary>Replaces the internal buffer with <paramref name="data"/> and resets position to 0.</summary>
    public void Rewind(byte[] data)
    {
        _inner = new MemoryStream(data, writable: false);
    }
}

/// <summary>Extension to allow null-safe invocation inline.</summary>
file static class DisposableExtensions
{
    public static void Let<T>(this T? value, Action<T> action) where T : class
    {
        if (value is not null) action(value);
    }
}
