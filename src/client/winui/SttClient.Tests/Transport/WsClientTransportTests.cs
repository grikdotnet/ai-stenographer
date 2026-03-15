using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using SttClient.Protocol;
using SttClient.Recognition;
using SttClient.State;
using SttClient.Transport;
using Xunit;

namespace SttClient.Tests.Transport;

/// <summary>
/// Tests for <see cref="WsClientTransport"/> covering audio send, queue-full drop,
/// message dispatch, shutdown state, and control command behavior.
/// </summary>
public class WsClientTransportTests
{
    private static AudioFrameEncoder CreateEncoder() =>
        new(NullLogger<AudioFrameEncoder>.Instance);

    private static AppStateManager CreateStateManager() =>
        new(NullLogger<AppStateManager>.Instance);

    private static ServerMessageDecoder CreateDecoder() =>
        new(NullLogger<ServerMessageDecoder>.Instance);

    private static RemoteRecognitionPublisher CreatePublisher(AppStateManager stateManager) =>
        new(
            new RecognitionResultFanOut(NullLogger<RecognitionResultFanOut>.Instance),
            stateManager,
            CreateDecoder(),
            NullLogger<RemoteRecognitionPublisher>.Instance);

    private static WsClientTransport CreateTransport(
        FakeWebSocket fakeWs,
        AppStateManager? stateManager = null,
        RemoteRecognitionPublisher? publisher = null)
    {
        stateManager ??= CreateStateManager();
        publisher ??= CreatePublisher(stateManager);

        return new WsClientTransport(
            fakeWs,
            CreateEncoder(),
            publisher,
            stateManager,
            NullLogger<WsClientTransport>.Instance);
    }

    // -------------------------------------------------------------------------
    // SendAudioChunkAsync
    // -------------------------------------------------------------------------

    [Fact]
    public async Task SendAudioChunkAsync_EnqueuesEncodedBytes()
    {
        var fakeWs = new FakeWebSocket();
        var transport = CreateTransport(fakeWs);

        transport.SendAudioChunkAsync("sess1", chunkId: 1, timestamp: 100.0, samples: [0.1f, 0.2f]);
        transport.StartAsync();

        await fakeWs.WaitForSendCountAsync(minCount: 1, timeout: TimeSpan.FromSeconds(3));

        await transport.StopAsync(serverInitiated: true);

        Assert.True(fakeWs.BinarySends.Count >= 1);
        Assert.True(fakeWs.BinarySends[0].Length > 0);
    }

    [Fact]
    public async Task SendAudioChunkAsync_QueueFull_DropsCurrentFrame_DoesNotBlock()
    {
        var fakeWs = new FakeWebSocket(blockSend: true);
        var transport = CreateTransport(fakeWs);

        transport.StartAsync();

        for (int i = 0; i < 20; i++)
            transport.SendAudioChunkAsync("sess1", i, timestamp: 0.0, samples: [0.1f]);

        var overflowTask = Task.Run(() =>
            transport.SendAudioChunkAsync("sess1", chunkId: 99, timestamp: 0.0, samples: [0.1f]));

        var completedBeforeTimeout = await Task.WhenAny(overflowTask, Task.Delay(200)) == overflowTask;

        Assert.True(completedBeforeTimeout, "SendAudioChunkAsync should return immediately when queue is full");

        await transport.StopAsync(serverInitiated: true);
    }

    [Fact]
    public async Task SendAudioChunkAsync_TimestampIsInSeconds()
    {
        var fakeWs = new FakeWebSocket();
        var transport = CreateTransport(fakeWs);

        var timestampSeconds = 1700000000.123;
        transport.SendAudioChunkAsync("sess1", chunkId: 1, timestamp: timestampSeconds, samples: [0.1f]);
        transport.StartAsync();

        await fakeWs.WaitForSendCountAsync(minCount: 1, timeout: TimeSpan.FromSeconds(3));
        await transport.StopAsync(serverInitiated: true);

        var frame = fakeWs.BinarySends[0];
        var headerLen = BitConverter.ToUInt32(frame, 0);
        var headerJson = System.Text.Encoding.UTF8.GetString(frame, 4, (int)headerLen);
        using var doc = System.Text.Json.JsonDocument.Parse(headerJson);
        var ts = doc.RootElement.GetProperty("timestamp").GetDouble();

        Assert.True(ts < 2e10, $"Timestamp {ts} looks like milliseconds, expected seconds");
        Assert.Equal(timestampSeconds, ts, precision: 3);
    }

    // -------------------------------------------------------------------------
    // ReceiveLoop
    // -------------------------------------------------------------------------

    [Fact]
    public async Task ReceiveLoop_TextFrame_DispatchesToPublisher()
    {
        const string sessionId = "sess-dispatch";
        const string testJson = $$$"""{"type":"recognition_result","session_id":"{{{sessionId}}}","status":"final","text":"hello","start_time":0,"end_time":1,"chunk_ids":[1]}""";

        var fakeWs = new FakeWebSocket();
        fakeWs.EnqueueTextMessage(testJson);

        var stateManager = CreateStateManager();
        var fanOut = new RecognitionResultFanOut(NullLogger<RecognitionResultFanOut>.Instance);
        var publisher = new RemoteRecognitionPublisher(fanOut, stateManager, CreateDecoder(), NullLogger<RemoteRecognitionPublisher>.Instance);
        publisher.SetSessionId(sessionId);

        var resultTcs = new TaskCompletionSource<RecognitionResult>();
        fanOut.AddSubscriber(new LambdaSubscriber(onFinal: r => resultTcs.TrySetResult(r)));

        var transport = new WsClientTransport(
            fakeWs,
            CreateEncoder(),
            publisher,
            stateManager,
            NullLogger<WsClientTransport>.Instance);

        transport.StartAsync();

        var result = await resultTcs.Task.WaitAsync(TimeSpan.FromSeconds(3));

        await transport.StopAsync(serverInitiated: true);

        Assert.Equal("hello", result.Text);
    }

    [Fact]
    public async Task ReceiveLoop_MultiFrameTextMessage_ReassemblesBeforeDispatching()
    {
        const string sessionId = "sess-multiframe";
        const string part1 = """{"type":"recognition_result","session_id":"sess-multiframe","status":"final","text":"hel""";
        const string part2 = """lo","start_time":0,"end_time":1,"chunk_ids":[1]}""";

        var fakeWs = new FakeWebSocket();
        fakeWs.EnqueueTextFrame(part1, endOfMessage: false);
        fakeWs.EnqueueTextFrame(part2, endOfMessage: true);

        var stateManager = CreateStateManager();
        var fanOut = new RecognitionResultFanOut(NullLogger<RecognitionResultFanOut>.Instance);
        var publisher = new RemoteRecognitionPublisher(fanOut, stateManager, CreateDecoder(), NullLogger<RemoteRecognitionPublisher>.Instance);
        publisher.SetSessionId(sessionId);

        var resultTcs = new TaskCompletionSource<RecognitionResult>();
        fanOut.AddSubscriber(new LambdaSubscriber(onFinal: r => resultTcs.TrySetResult(r)));

        var transport = new WsClientTransport(fakeWs, CreateEncoder(), publisher, stateManager, NullLogger<WsClientTransport>.Instance);
        transport.StartAsync();

        var result = await resultTcs.Task.WaitAsync(TimeSpan.FromSeconds(3));
        await transport.StopAsync(serverInitiated: true);

        Assert.Equal("hello", result.Text);
    }

    [Fact]
    public async Task ReceiveLoop_WebSocketClose_SetsShutdownState()
    {
        var fakeWs = new FakeWebSocket();
        fakeWs.EnqueueCloseMessage();

        var stateManager = CreateStateManager();
        var shutdownTcs = new TaskCompletionSource();
        stateManager.AddObserver((_, newState) =>
        {
            if (newState == AppState.Shutdown)
                shutdownTcs.TrySetResult();
        });

        var transport = CreateTransport(fakeWs, stateManager);
        transport.StartAsync();

        await shutdownTcs.Task.WaitAsync(TimeSpan.FromSeconds(3));

        Assert.Equal(AppState.Shutdown, stateManager.CurrentState);
    }

    // -------------------------------------------------------------------------
    // StopAsync
    // -------------------------------------------------------------------------

    [Fact]
    public async Task StopAsync_ServerInitiated_DoesNotSendControlCommand()
    {
        var fakeWs = new FakeWebSocket();
        var transport = CreateTransport(fakeWs);
        transport.SessionId = "sess-abc";
        transport.StartAsync();

        await transport.StopAsync(serverInitiated: true);

        var controlFrames = fakeWs.TextSends
            .Where(t => t.Contains("control_command"))
            .ToList();

        Assert.Empty(controlFrames);
    }

    [Fact]
    public async Task StopAsync_ClientInitiated_SendsControlCommandShutdown()
    {
        var fakeWs = new FakeWebSocket();
        var transport = CreateTransport(fakeWs);
        transport.SessionId = "sess-xyz";
        transport.StartAsync();

        await transport.StopAsync(serverInitiated: false);

        var controlFrames = fakeWs.TextSends
            .Where(t => t.Contains("control_command") && t.Contains("shutdown"))
            .ToList();

        Assert.Single(controlFrames);
    }

    [Fact]
    public async Task StopAsync_ClientInitiated_WaitsForSessionClosed_BeforeCancelling()
    {
        var fakeWs = new FakeWebSocket();
        var stateManager = CreateStateManager();
        var fanOut = new RecognitionResultFanOut(NullLogger<RecognitionResultFanOut>.Instance);
        var publisher = new RemoteRecognitionPublisher(fanOut, stateManager, CreateDecoder(), NullLogger<RemoteRecognitionPublisher>.Instance);
        var transport = new WsClientTransport(fakeWs, CreateEncoder(), publisher, stateManager, NullLogger<WsClientTransport>.Instance);
        publisher.SessionClosedCallback = transport.SignalSessionClosed;
        transport.SessionId = "sess-wait";
        transport.StartAsync();

        // Enqueue session_closed shortly after StopAsync starts
        _ = Task.Run(async () =>
        {
            await Task.Delay(100);
            fakeWs.EnqueueTextMessage("{\"type\":\"session_closed\",\"session_id\":\"sess-wait\"}");
        });

        var sw = System.Diagnostics.Stopwatch.StartNew();
        await transport.StopAsync(serverInitiated: false);
        sw.Stop();

        // Should complete quickly (well under 2 s), because session_closed was received
        Assert.True(sw.ElapsedMilliseconds < 1500, $"StopAsync took {sw.ElapsedMilliseconds} ms — expected < 1500");
    }

    [Fact]
    public async Task SendControlCommandShutdown_JsonIsValidAndContainsSessionId()
    {
        var fakeWs = new FakeWebSocket();
        var transport = CreateTransport(fakeWs);
        transport.SessionId = "sess-json-test";
        transport.StartAsync();

        await transport.StopAsync(serverInitiated: false);

        Assert.Single(fakeWs.TextSends);

        using var doc = JsonDocument.Parse(fakeWs.TextSends[0]);
        var root = doc.RootElement;

        Assert.Equal("control_command", root.GetProperty("type").GetString());
        Assert.Equal("shutdown", root.GetProperty("command").GetString());
        Assert.Equal("sess-json-test", root.GetProperty("session_id").GetString());
        Assert.True(root.TryGetProperty("timestamp", out var ts) && ts.ValueKind == JsonValueKind.Number);
    }

    [Fact]
    public async Task StopAsync_ClientInitiated_TimesOut_WhenNoSessionClosed()
    {
        var fakeWs = new FakeWebSocket();
        var transport = CreateTransport(fakeWs);
        transport.SessionId = "sess-timeout";
        transport.StartAsync();

        var sw = System.Diagnostics.Stopwatch.StartNew();
        await transport.StopAsync(serverInitiated: false);
        sw.Stop();

        // Should wait the full 2 s timeout
        Assert.True(sw.ElapsedMilliseconds >= 1900, $"StopAsync completed in {sw.ElapsedMilliseconds} ms — expected ~2000");
    }
}

// =============================================================================
// Test helpers
// =============================================================================

/// <summary>
/// Fake implementation of <see cref="IWebSocket"/> for unit tests.
/// Records sent frames and serves a configurable queue of received messages.
/// </summary>
internal sealed class FakeWebSocket : IWebSocket
{
    private readonly ConcurrentQueue<FakeMessage> _receiveQueue = new();
    private readonly SemaphoreSlim _receiveSignal = new(0);
    private readonly TaskCompletionSource _closeHandledTcs = new();
    private readonly SemaphoreSlim _sendGate;
    private readonly SemaphoreSlim _sendCountSignal = new(0);

    /// <summary>Initializes the fake, optionally blocking all SendAsync calls until released.</summary>
    /// <param name="blockSend">If true, SendAsync blocks until <see cref="ReleaseSend"/> is called.</param>
    public FakeWebSocket(bool blockSend = false)
    {
        _sendGate = new SemaphoreSlim(blockSend ? 0 : int.MaxValue, int.MaxValue);
    }

    /// <summary>Gets all recorded binary frame payloads.</summary>
    public List<byte[]> BinarySends { get; } = [];

    /// <summary>Gets all recorded text frame payloads (decoded as UTF-8).</summary>
    public List<string> TextSends { get; } = [];

    /// <inheritdoc/>
    public WebSocketState State => WebSocketState.Open;

    /// <summary>Enqueues a text message to be returned by the next ReceiveAsync call.</summary>
    public void EnqueueTextMessage(string text) => EnqueueTextFrame(text, endOfMessage: true);

    /// <summary>Enqueues a text frame with an explicit EndOfMessage flag for multi-frame testing.</summary>
    public void EnqueueTextFrame(string text, bool endOfMessage)
    {
        _receiveQueue.Enqueue(new FakeMessage(text, IsClose: false, EndOfMessage: endOfMessage));
        _receiveSignal.Release();
    }

    /// <summary>Enqueues a WebSocket Close control frame.</summary>
    public void EnqueueCloseMessage()
    {
        _receiveQueue.Enqueue(new FakeMessage(string.Empty, IsClose: true));
        _receiveSignal.Release();
    }

    /// <summary>Unblocks one pending SendAsync (when blockSend was true).</summary>
    public void ReleaseSend() => _sendGate.Release();

    /// <summary>Waits until at least <paramref name="minCount"/> binary sends have been recorded.</summary>
    public async Task WaitForSendCountAsync(int minCount, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (BinarySends.Count < minCount && DateTime.UtcNow < deadline)
            await _sendCountSignal.WaitAsync(TimeSpan.FromMilliseconds(50));
    }

    /// <summary>Waits until the Close frame has been processed by the ReceiveLoop.</summary>
    public Task WaitForCloseHandledAsync(TimeSpan timeout) =>
        _closeHandledTcs.Task.WaitAsync(timeout);

    /// <inheritdoc/>
    public async Task SendAsync(ReadOnlyMemory<byte> buffer, WebSocketMessageType messageType, bool endOfMessage, CancellationToken ct)
    {
        await _sendGate.WaitAsync(ct);

        if (messageType == WebSocketMessageType.Binary)
        {
            lock (BinarySends) BinarySends.Add(buffer.ToArray());
            _sendCountSignal.Release();
        }
        else if (messageType == WebSocketMessageType.Text)
        {
            var text = Encoding.UTF8.GetString(buffer.Span);
            lock (TextSends) TextSends.Add(text);
        }
    }

    /// <inheritdoc/>
    public async Task<WebSocketReceiveResult> ReceiveAsync(Memory<byte> buffer, CancellationToken ct)
    {
        await _receiveSignal.WaitAsync(ct);

        if (!_receiveQueue.TryDequeue(out var msg))
            throw new InvalidOperationException("ReceiveQueue unexpectedly empty");

        if (msg.IsClose)
        {
            _closeHandledTcs.TrySetResult();
            return new WebSocketReceiveResult(0, WebSocketMessageType.Close, endOfMessage: true,
                WebSocketCloseStatus.NormalClosure, "server closed");
        }

        var bytes = Encoding.UTF8.GetBytes(msg.Text);
        bytes.CopyTo(buffer);
        return new WebSocketReceiveResult(bytes.Length, WebSocketMessageType.Text, endOfMessage: msg.EndOfMessage);
    }

    /// <inheritdoc/>
    public Task CloseAsync(WebSocketCloseStatus closeStatus, string? statusDescription, CancellationToken ct) =>
        Task.CompletedTask;

    /// <inheritdoc/>
    public Task CloseOutputAsync(WebSocketCloseStatus closeStatus, string? statusDescription, CancellationToken ct) =>
        Task.CompletedTask;

    /// <inheritdoc/>
    public void Dispose() { }

    private sealed record FakeMessage(string Text, bool IsClose, bool EndOfMessage = true);
}

/// <summary>
/// Test-only <see cref="IRecognitionSubscriber"/> that invokes configurable callbacks,
/// allowing tests to capture recognition events without subclassing sealed production types.
/// </summary>
internal sealed class LambdaSubscriber : IRecognitionSubscriber
{
    private readonly Action<RecognitionResult>? _onPartial;
    private readonly Action<RecognitionResult>? _onFinal;

    /// <summary>Initializes with optional callbacks for partial and final results.</summary>
    public LambdaSubscriber(Action<RecognitionResult>? onPartial = null, Action<RecognitionResult>? onFinal = null)
    {
        _onPartial = onPartial;
        _onFinal = onFinal;
    }

    /// <inheritdoc/>
    public void OnPartialUpdate(RecognitionResult result) => _onPartial?.Invoke(result);

    /// <inheritdoc/>
    public void OnFinalization(RecognitionResult result) => _onFinal?.Invoke(result);
}
