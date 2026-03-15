using System.Net;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using SttClient.Orchestration;
using SttClient.Audio;
using SttClient.Formatting;
using SttClient.Protocol;
using SttClient.Recognition;
using SttClient.State;
using SttClient.Transport;
using SttClient.ViewModels;
using Xunit;

namespace SttClient.Tests.Integration;

/// <summary>
/// Marks integration tests that must not run in parallel with each other.
/// Background WebSocket tasks from one test could steal the HttpListener connection
/// intended for the next test when tests run concurrently.
/// </summary>
[CollectionDefinition(nameof(SequentialIntegration), DisableParallelization = true)]
public sealed class SequentialIntegration { }

/// <summary>
/// Integration tests for <see cref="ClientOrchestrator"/> using an in-process fake WebSocket server
/// backed by <see cref="HttpListener"/> / <see cref="HttpListenerWebSocketContext"/>.
///
/// Each test spins up a real HTTP listener on a random port, accepts exactly one WebSocket
/// connection, exchanges protocol messages, and asserts observable side-effects (state transitions,
/// ViewModel updates, frames received by the server).
/// </summary>
[Collection(nameof(SequentialIntegration))]
public sealed class ClientOrchestratorTests : IAsyncDisposable
{
    private readonly HttpListener _httpListener;
    private readonly string _serverUrl;
    private readonly string _wsUrl;
    private const string SessionId = "test-session-id";
    private const string ProtocolVersion = "v1";

    public ClientOrchestratorTests()
    {
        _httpListener = new HttpListener();
        var port = FindFreePort();
        _httpListener.Prefixes.Add($"http://127.0.0.1:{port}/");
        _httpListener.Start();
        _serverUrl = $"http://127.0.0.1:{port}/";
        _wsUrl = $"ws://127.0.0.1:{port}/";
    }

    public async ValueTask DisposeAsync()
    {
        _httpListener.Stop();
        await Task.CompletedTask;
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static int FindFreePort()
    {
        var listener = new System.Net.Sockets.TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    private static string MakeSessionCreated() =>
        JsonSerializer.Serialize(new
        {
            type = "session_created",
            session_id = SessionId,
            protocol_version = ProtocolVersion,
            server_time = 1000.0
        });

    private static string MakeRecognitionResult(string text, string status = "final") =>
        JsonSerializer.Serialize(new
        {
            type = "recognition_result",
            session_id = SessionId,
            status,
            text,
            start_time = 0.0,
            end_time = 1.0,
            chunk_ids = new[] { 0 },
            utterance_id = 1
        });

    private static string MakeSessionClosed() =>
        JsonSerializer.Serialize(new
        {
            type = "session_closed",
            session_id = SessionId,
            reason = "server_shutdown",
            message = (string?)null
        });

    private ClientOrchestrator BuildOrchestrator(
        out AppStateManager stateManager,
        out SyncDispatcher dispatcher,
        out MainWindowViewModel viewModel,
        out FakeAudioSource audioSource)
    {
        stateManager = new AppStateManager(NullLogger<AppStateManager>.Instance);
        dispatcher = new SyncDispatcher();
        viewModel = new MainWindowViewModel(dispatcher, NullLogger<MainWindowViewModel>.Instance);
        stateManager.AddObserver(viewModel.OnStateChanged);

        var fanOut = new RecognitionResultFanOut(NullLogger<RecognitionResultFanOut>.Instance);
        var decoder = new ServerMessageDecoder(NullLogger<ServerMessageDecoder>.Instance);
        var publisher = new RemoteRecognitionPublisher(fanOut, stateManager, decoder, NullLogger<RemoteRecognitionPublisher>.Instance);
        var formatter = new TextFormatter(viewModel.Apply, NullLogger<TextFormatter>.Instance);
        fanOut.AddSubscriber(formatter);

        var encoder = new AudioFrameEncoder(NullLogger<AudioFrameEncoder>.Instance);
        audioSource = new FakeAudioSource();

        return new ClientOrchestrator(
            serverUrl: _wsUrl,
            stateManager: stateManager,
            publisher: publisher,
            encoder: encoder,
            audioSource: audioSource,
            loggerFactory: NullLoggerFactory.Instance);
    }

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------

    [Fact]
    public async Task Connect_ReceivesSessionCreated_TransitionsToRunning()
    {
        var connectTask = Task.Run(async () =>
        {
            var ctx = await _httpListener.GetContextAsync();
            var wsCtx = await ctx.AcceptWebSocketAsync(null);
            var ws = wsCtx.WebSocket;

            await ws.SendAsync(Encoding.UTF8.GetBytes(MakeSessionCreated()),
                WebSocketMessageType.Text, true, CancellationToken.None);

            // Keep alive briefly so the client can process
            await Task.Delay(300);
            await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "done", CancellationToken.None);
        });

        var orchestrator = BuildOrchestrator(out var stateManager, out _, out _, out _);
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));

        await orchestrator.ConnectAsync(cts.Token);

        Assert.Equal(AppState.Running, stateManager.CurrentState);

        await connectTask;
    }

    [Fact]
    public async Task AudioChunks_ArriveAtFakeServer_WithBinaryFrameFormat()
    {
        var receivedFrames = new List<byte[]>();
        var frameReceived = new TaskCompletionSource<bool>();

        var serverTask = Task.Run(async () =>
        {
            var ctx = await _httpListener.GetContextAsync();
            var wsCtx = await ctx.AcceptWebSocketAsync(null);
            var ws = wsCtx.WebSocket;

            await ws.SendAsync(Encoding.UTF8.GetBytes(MakeSessionCreated()),
                WebSocketMessageType.Text, true, CancellationToken.None);

            var buf = new byte[65536];
            while (ws.State == WebSocketState.Open)
            {
                WebSocketReceiveResult result;
                try { result = await ws.ReceiveAsync(buf, CancellationToken.None); }
                catch { break; }

                if (result.MessageType == WebSocketMessageType.Binary)
                {
                    var frame = buf[..result.Count];
                    receivedFrames.Add(frame);
                    frameReceived.TrySetResult(true);
                }
                else if (result.MessageType == WebSocketMessageType.Close)
                    break;
            }
        });

        var orchestrator = BuildOrchestrator(out _, out _, out _, out var audioSource);
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));

        await orchestrator.ConnectAsync(cts.Token);

        // Emit one audio chunk
        audioSource.EmitChunk(new float[512]);

        await frameReceived.Task.WaitAsync(TimeSpan.FromSeconds(3));

        Assert.NotEmpty(receivedFrames);

        // Verify binary frame format: 4-byte LE uint32 header length + JSON + float32 PCM
        var frame = receivedFrames[0];
        Assert.True(frame.Length >= 4);
        var headerLen = BitConverter.ToUInt32(frame, 0);
        Assert.True(headerLen > 0 && headerLen < frame.Length);
        var headerJson = Encoding.UTF8.GetString(frame, 4, (int)headerLen);
        using var doc = JsonDocument.Parse(headerJson);
        Assert.Equal("audio_chunk", doc.RootElement.GetProperty("type").GetString());

        await orchestrator.StopAsync();
        await serverTask;
    }

    [Fact]
    public async Task OnChunkReady_AudioFrameTimestampIsInSeconds()
    {
        var receivedFrames = new List<byte[]>();
        var frameReceived = new TaskCompletionSource<bool>();

        var serverTask = Task.Run(async () =>
        {
            var ctx = await _httpListener.GetContextAsync();
            var wsCtx = await ctx.AcceptWebSocketAsync(null);
            var ws = wsCtx.WebSocket;

            await ws.SendAsync(Encoding.UTF8.GetBytes(MakeSessionCreated()),
                WebSocketMessageType.Text, true, CancellationToken.None);

            var buf = new byte[65536];
            while (ws.State == WebSocketState.Open)
            {
                WebSocketReceiveResult result;
                try { result = await ws.ReceiveAsync(buf, CancellationToken.None); }
                catch { break; }

                if (result.MessageType == WebSocketMessageType.Binary)
                {
                    receivedFrames.Add(buf[..result.Count]);
                    frameReceived.TrySetResult(true);
                }
                else if (result.MessageType == WebSocketMessageType.Close)
                    break;
            }
        });

        var orchestrator = BuildOrchestrator(out _, out _, out _, out var audioSource);
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));

        await orchestrator.ConnectAsync(cts.Token);

        audioSource.EmitChunk(new float[512]);

        await frameReceived.Task.WaitAsync(TimeSpan.FromSeconds(3));

        var frame = receivedFrames[0];
        var headerLen = BitConverter.ToUInt32(frame, 0);
        var headerJson = Encoding.UTF8.GetString(frame, 4, (int)headerLen);
        using var doc = JsonDocument.Parse(headerJson);
        var timestamp = doc.RootElement.GetProperty("timestamp").GetDouble();

        Assert.True(timestamp > 1e9, $"Timestamp {timestamp} is too small for Unix seconds");
        Assert.True(timestamp < 2e10, $"Timestamp {timestamp} looks like milliseconds, expected seconds");

        await orchestrator.StopAsync();
        await serverTask;
    }

    [Fact]
    public async Task RecognitionResult_FromServer_AppearsInViewModel()
    {
        var orchestrator = BuildOrchestrator(out var stateManager, out _, out var viewModel, out _);

        var serverTask = Task.Run(async () =>
        {
            var ctx = await _httpListener.GetContextAsync();
            var wsCtx = await ctx.AcceptWebSocketAsync(null);
            var ws = wsCtx.WebSocket;

            await ws.SendAsync(Encoding.UTF8.GetBytes(MakeSessionCreated()),
                WebSocketMessageType.Text, true, CancellationToken.None);

            await Task.Delay(100);

            await ws.SendAsync(Encoding.UTF8.GetBytes(MakeRecognitionResult("hello world")),
                WebSocketMessageType.Text, true, CancellationToken.None);

            await Task.Delay(300);
            await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "done", CancellationToken.None);
        });

        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        await orchestrator.ConnectAsync(cts.Token);

        await serverTask;
        await Task.Delay(100); // let ReceiveLoop dispatch

        Assert.Contains("hello world", viewModel.FinalizedText);
    }

    [Fact]
    public async Task ServerSendsSessionClosed_TransitionsToShutdown_WithoutClientSendingControlCommand()
    {
        var receivedTexts = new List<string>();

        var serverTask = Task.Run(async () =>
        {
            var ctx = await _httpListener.GetContextAsync();
            var wsCtx = await ctx.AcceptWebSocketAsync(null);
            var ws = wsCtx.WebSocket;

            await ws.SendAsync(Encoding.UTF8.GetBytes(MakeSessionCreated()),
                WebSocketMessageType.Text, true, CancellationToken.None);

            await Task.Delay(100);

            await ws.SendAsync(Encoding.UTF8.GetBytes(MakeSessionClosed()),
                WebSocketMessageType.Text, true, CancellationToken.None);

            // Drain remaining frames to detect any stray control_command
            var buf = new byte[4096];
            var deadline = DateTime.UtcNow.AddSeconds(1);
            while (DateTime.UtcNow < deadline && ws.State == WebSocketState.Open)
            {
                try
                {
                    using var readCts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
                    var result = await ws.ReceiveAsync(buf, readCts.Token);
                    if (result.MessageType == WebSocketMessageType.Text)
                        receivedTexts.Add(Encoding.UTF8.GetString(buf, 0, result.Count));
                    else if (result.MessageType == WebSocketMessageType.Close)
                        break;
                }
                catch (OperationCanceledException) { break; }
                catch (WebSocketException) { break; }
            }
        });

        var orchestrator = BuildOrchestrator(out var stateManager, out _, out _, out _);
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        await orchestrator.ConnectAsync(cts.Token);

        await serverTask;
        await Task.Delay(200);

        Assert.Equal(AppState.Shutdown, stateManager.CurrentState);

        // Server-initiated: no control_command shutdown should have been sent
        Assert.DoesNotContain(receivedTexts,
            t => t.Contains("control_command") && t.Contains("shutdown"));
    }

    [Fact]
    public async Task ClientInitiatedShutdown_SendsControlCommandBeforeClose()
    {
        var receivedTexts = new List<string>();
        var serverDone = new TaskCompletionSource<bool>();

        var serverTask = Task.Run(async () =>
        {
            var ctx = await _httpListener.GetContextAsync();
            var wsCtx = await ctx.AcceptWebSocketAsync(null);
            var ws = wsCtx.WebSocket;

            await ws.SendAsync(Encoding.UTF8.GetBytes(MakeSessionCreated()),
                WebSocketMessageType.Text, true, CancellationToken.None);

            var buf = new byte[4096];
            while (ws.State == WebSocketState.Open)
            {
                try
                {
                    using var readCts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
                    var result = await ws.ReceiveAsync(buf, readCts.Token);
                    if (result.MessageType == WebSocketMessageType.Text)
                        receivedTexts.Add(Encoding.UTF8.GetString(buf, 0, result.Count));
                    else if (result.MessageType == WebSocketMessageType.Close)
                        break;
                }
                catch { break; }
            }

            serverDone.TrySetResult(true);
        });

        var orchestrator = BuildOrchestrator(out _, out _, out _, out _);
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        await orchestrator.ConnectAsync(cts.Token);

        await orchestrator.StopAsync();
        await serverDone.Task.WaitAsync(TimeSpan.FromSeconds(3));

        Assert.Contains(receivedTexts,
            t => t.Contains("control_command") && t.Contains("shutdown"));
    }

    // -------------------------------------------------------------------------
    // Test doubles
    // -------------------------------------------------------------------------

    private sealed class SyncDispatcher : IDispatcherQueueAdapter
    {
        public bool TryEnqueue(Action action) { action(); return true; }
    }

    private sealed class FakeAudioSource : IAudioSource
    {
        public event Action<AudioChunk>? ChunkReady;

        public void Start() { }
        public void Stop() { }

        public void EmitChunk(float[] samples) =>
            ChunkReady?.Invoke(new AudioChunk(samples, DateTimeOffset.UtcNow));
    }
}
