using Microsoft.Extensions.Logging.Abstractions;
using Moq;
using SttClient.Recognition;
using SttClient.State;
using Xunit;

namespace SttClient.Tests.Recognition;

/// <summary>
/// Tests for <see cref="RemoteRecognitionPublisher"/> covering JSON dispatch routing,
/// session ID validation, error handling, and ping/pong keepalive.
/// </summary>
public class RemoteRecognitionPublisherTests
{
    private const string ValidSessionId = "sess-abc-123";

    private static (RemoteRecognitionPublisher publisher, RecognitionResultFanOut fanOut, Mock<IRecognitionSubscriber> subscriber, AppStateManager stateManager)
        MakePublisher()
    {
        var subscriber = new Mock<IRecognitionSubscriber>();
        var fanOut = new RecognitionResultFanOut(NullLogger<RecognitionResultFanOut>.Instance);
        fanOut.AddSubscriber(subscriber.Object);

        var stateManager = new AppStateManager(NullLogger<AppStateManager>.Instance);

        var publisher = new RemoteRecognitionPublisher(
            fanOut,
            stateManager,
            NullLogger<RemoteRecognitionPublisher>.Instance);
        publisher.SetSessionId(ValidSessionId);

        return (publisher, fanOut, subscriber, stateManager);
    }

    private static string PartialResultJson(string sessionId = ValidSessionId) => $$"""
        {
            "type": "recognition_result",
            "session_id": "{{sessionId}}",
            "status": "partial",
            "text": "hello world",
            "start_time": 0.1,
            "end_time": 1.2,
            "utterance_id": 42,
            "chunk_ids": [1, 2, 3],
            "token_confidences": [0.9, 0.8, 0.95]
        }
        """;

    private static string FinalResultJson(string sessionId = ValidSessionId) => $$"""
        {
            "type": "recognition_result",
            "session_id": "{{sessionId}}",
            "status": "final",
            "text": "goodbye",
            "start_time": 2.0,
            "end_time": 3.5,
            "utterance_id": null,
            "chunk_ids": [4, 5],
            "token_confidences": null
        }
        """;

    [Fact]
    public void Dispatch_PartialResult_CallsOnPartialUpdate()
    {
        var (publisher, _, subscriber, _) = MakePublisher();

        publisher.Dispatch(PartialResultJson());

        subscriber.Verify(s => s.OnPartialUpdate(It.Is<RecognitionResult>(r =>
            r.Text == "hello world" &&
            r.StartTime == 0.1 &&
            r.EndTime == 1.2 &&
            r.UtteranceId == 42 &&
            r.ChunkIds.SequenceEqual(new[] { 1, 2, 3 })
        )), Times.Once);
        subscriber.Verify(s => s.OnFinalization(It.IsAny<RecognitionResult>()), Times.Never);
    }

    [Fact]
    public void Dispatch_FinalResult_CallsOnFinalization()
    {
        var (publisher, _, subscriber, _) = MakePublisher();

        publisher.Dispatch(FinalResultJson());

        subscriber.Verify(s => s.OnFinalization(It.Is<RecognitionResult>(r =>
            r.Text == "goodbye" &&
            r.StartTime == 2.0 &&
            r.EndTime == 3.5 &&
            r.UtteranceId == null &&
            r.ChunkIds.SequenceEqual(new[] { 4, 5 })
        )), Times.Once);
        subscriber.Verify(s => s.OnPartialUpdate(It.IsAny<RecognitionResult>()), Times.Never);
    }

    [Fact]
    public void Dispatch_SessionClosed_SetsShutdownState()
    {
        var (publisher, _, _, stateManager) = MakePublisher();

        stateManager.SetState(AppState.Running);
        publisher.Dispatch($$"""{"type":"session_closed","session_id":"{{ValidSessionId}}","reason":"server_shutdown","message":null}""");

        Assert.Equal(AppState.Shutdown, stateManager.CurrentState);
    }

    [Fact]
    public void Dispatch_SessionIdMismatch_DiscardsAndLogsWarning()
    {
        var (publisher, _, subscriber, _) = MakePublisher();

        publisher.Dispatch(PartialResultJson(sessionId: "wrong-session-id"));

        subscriber.Verify(s => s.OnPartialUpdate(It.IsAny<RecognitionResult>()), Times.Never);
        subscriber.Verify(s => s.OnFinalization(It.IsAny<RecognitionResult>()), Times.Never);
    }

    [Fact]
    public void Dispatch_UnknownType_IsIgnored_NoException()
    {
        var (publisher, _, subscriber, _) = MakePublisher();

        var ex = Record.Exception(() => publisher.Dispatch("""{"type":"some_future_type","data":42}"""));

        Assert.Null(ex);
        subscriber.Verify(s => s.OnPartialUpdate(It.IsAny<RecognitionResult>()), Times.Never);
        subscriber.Verify(s => s.OnFinalization(It.IsAny<RecognitionResult>()), Times.Never);
    }

    [Fact]
    public void Dispatch_MalformedJson_IsSwallowed_NoException()
    {
        var (publisher, _, _, _) = MakePublisher();

        var ex = Record.Exception(() => publisher.Dispatch("{not valid json at all!!!"));

        Assert.Null(ex);
    }

    [Fact]
    public async Task Dispatch_PingMessage_InvokesPongSender()
    {
        var (publisher, _, _, _) = MakePublisher();

        string? capturedPong = null;
        publisher.PongSender = json =>
        {
            capturedPong = json;
            return Task.CompletedTask;
        };

        publisher.Dispatch("""{"type":"ping","timestamp":1234567.89}""");

        await Task.Yield();

        Assert.NotNull(capturedPong);

        using var doc = System.Text.Json.JsonDocument.Parse(capturedPong!);
        var root = doc.RootElement;

        Assert.Equal("pong", root.GetProperty("type").GetString());
        Assert.Equal(ValidSessionId, root.GetProperty("session_id").GetString());
        Assert.Equal(1234567.89, root.GetProperty("timestamp").GetDouble(), precision: 2);
    }
}
