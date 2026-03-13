using Microsoft.Extensions.Logging.Abstractions;
using NAudio.Wave;
using SttClient.Audio;
using SttClient.State;
using Xunit;

namespace SttClient.Tests.Audio;

/// <summary>
/// Tests for <see cref="WasapiAudioSource"/> covering chunk emission, state-driven
/// start/stop behaviour, and the resampling fallback path.
/// </summary>
public class WasapiAudioSourceTests
{
    private static AppStateManager CreateStateManager() =>
        new(NullLogger<AppStateManager>.Instance);

    private static WasapiAudioSource CreateSource(
        FakeWaveCapture capture,
        AppStateManager? stateManager = null)
    {
        stateManager ??= CreateStateManager();
        return new WasapiAudioSource(
            capture,
            stateManager,
            NullLogger<WasapiAudioSource>.Instance);
    }

    // -------------------------------------------------------------------------
    // Chunk emission
    // -------------------------------------------------------------------------

    [Fact]
    public void Start_ChunkReady_EmittedWith512Samples()
    {
        var capture = new FakeWaveCapture(nativeFormat: WaveFormat.CreateIeeeFloatWaveFormat(16000, 1));
        var source = CreateSource(capture);

        AudioChunk? received = null;
        source.ChunkReady += chunk => received = chunk;

        source.Start();
        capture.EmitSamples(512);

        Assert.NotNull(received);
        Assert.Equal(512, received.Samples.Length);
    }

    [Fact]
    public void Start_EmitsMultipleChunks_EachHas512Samples()
    {
        var capture = new FakeWaveCapture(nativeFormat: WaveFormat.CreateIeeeFloatWaveFormat(16000, 1));
        var source = CreateSource(capture);

        var chunks = new List<AudioChunk>();
        source.ChunkReady += chunk => chunks.Add(chunk);

        source.Start();
        capture.EmitSamples(512);
        capture.EmitSamples(512);
        capture.EmitSamples(512);

        Assert.Equal(3, chunks.Count);
        Assert.All(chunks, c => Assert.Equal(512, c.Samples.Length));
    }

    // -------------------------------------------------------------------------
    // State-driven start / stop
    // -------------------------------------------------------------------------

    [Fact]
    public void StateTransition_RunningToPaused_StopsEmission()
    {
        var stateManager = CreateStateManager();
        var capture = new FakeWaveCapture(nativeFormat: WaveFormat.CreateIeeeFloatWaveFormat(16000, 1));
        var source = CreateSource(capture, stateManager);

        var chunks = new List<AudioChunk>();
        source.ChunkReady += chunk => chunks.Add(chunk);

        source.Start();
        stateManager.SetState(AppState.Running);
        capture.EmitSamples(512);
        int countBeforePause = chunks.Count;

        stateManager.SetState(AppState.Paused);
        capture.EmitSamples(512);

        Assert.Equal(countBeforePause, chunks.Count);
    }

    [Fact]
    public void StateTransition_PausedToRunning_RestartsEmission()
    {
        var stateManager = CreateStateManager();
        var capture = new FakeWaveCapture(nativeFormat: WaveFormat.CreateIeeeFloatWaveFormat(16000, 1));
        var source = CreateSource(capture, stateManager);

        var chunks = new List<AudioChunk>();
        source.ChunkReady += chunk => chunks.Add(chunk);

        source.Start();
        stateManager.SetState(AppState.Running);
        stateManager.SetState(AppState.Paused);

        stateManager.SetState(AppState.Running);
        capture.EmitSamples(512);

        Assert.Single(chunks);
    }

    [Fact]
    public void StateTransition_ToShutdown_StopsPermanently()
    {
        var stateManager = CreateStateManager();
        var capture = new FakeWaveCapture(nativeFormat: WaveFormat.CreateIeeeFloatWaveFormat(16000, 1));
        var source = CreateSource(capture, stateManager);

        var chunks = new List<AudioChunk>();
        source.ChunkReady += chunk => chunks.Add(chunk);

        source.Start();
        stateManager.SetState(AppState.Shutdown);
        capture.EmitSamples(512);

        Assert.Empty(chunks);
    }

    // -------------------------------------------------------------------------
    // Resampling path
    // -------------------------------------------------------------------------

    [Fact]
    public void ResamplingPath_NonNativeFormat_OutputIs512Float32SamplesAt16kHz()
    {
        // Device reports 44100 Hz stereo PCM — triggers MediaFoundationResampler path
        var nativeFormat = new WaveFormat(44100, 16, 2);
        var capture = new FakeWaveCapture(nativeFormat: nativeFormat);
        var source = CreateSource(capture);

        AudioChunk? received = null;
        source.ChunkReady += chunk => received = chunk;

        source.Start();

        // Emit enough 44100/2ch/16-bit PCM bytes to produce at least one 512-sample output chunk.
        // 512 samples at 16 kHz = 32 ms. At 44100 Hz stereo 16-bit: 44100*2*2*0.032 ≈ 5645 bytes.
        // We emit double to ensure the resampler has enough data.
        capture.EmitPcmBytes(nativeFormat, durationMs: 100);

        Assert.NotNull(received);
        Assert.Equal(512, received.Samples.Length);
    }
}

// =============================================================================
// Test helpers
// =============================================================================

/// <summary>
/// Fake implementation of <see cref="IWaveCapture"/> that drives DataAvailable
/// synchronously, allowing tests to control audio emission without real hardware.
/// </summary>
internal sealed class FakeWaveCapture : IWaveCapture
{
    private bool _active;

    /// <summary>Initializes the fake with the given wave format.</summary>
    public FakeWaveCapture(WaveFormat nativeFormat)
    {
        WaveFormat = nativeFormat;
    }

    /// <inheritdoc/>
    public WaveFormat WaveFormat { get; }

    /// <inheritdoc/>
    public event EventHandler<WaveInEventArgs>? DataAvailable;

    /// <inheritdoc/>
    public void StartRecording() => _active = true;

    /// <inheritdoc/>
    public void StopRecording() => _active = false;

    /// <inheritdoc/>
    public void Dispose() { }

    /// <summary>
    /// Emits a DataAvailable event containing <paramref name="sampleCount"/> float32 samples
    /// at the configured WaveFormat (must be IEEE float 16 kHz mono).
    /// </summary>
    public void EmitSamples(int sampleCount)
    {
        if (!_active) return;

        var samples = new float[sampleCount];
        var bytes = new byte[sampleCount * sizeof(float)];
        Buffer.BlockCopy(samples, 0, bytes, 0, bytes.Length);

        DataAvailable?.Invoke(this, new WaveInEventArgs(bytes, bytes.Length));
    }

    /// <summary>
    /// Emits raw PCM bytes matching <paramref name="format"/> for <paramref name="durationMs"/> milliseconds.
    /// Used to test the resampling path with non-native formats.
    /// </summary>
    public void EmitPcmBytes(WaveFormat format, int durationMs)
    {
        if (!_active) return;

        int byteCount = (int)(format.AverageBytesPerSecond * durationMs / 1000.0);
        byteCount -= byteCount % format.BlockAlign;
        var bytes = new byte[byteCount];

        DataAvailable?.Invoke(this, new WaveInEventArgs(bytes, byteCount));
    }
}
