using Microsoft.Extensions.Logging.Abstractions;
using Moq;
using SttClient.Recognition;
using Xunit;

namespace SttClient.Tests.Recognition;

/// <summary>
/// Tests for <see cref="RecognitionResultFanOut"/> covering dispatch, error isolation, and thread safety.
/// </summary>
public class RecognitionResultFanOutTests
{
    private static RecognitionResult MakeResult() =>
        new("hello", 0.0, 1.0, null, [1, 2], null);

    private static RecognitionResultFanOut MakeFanOut() =>
        new(NullLogger<RecognitionResultFanOut>.Instance);

    [Fact]
    public void OnPartialUpdate_CallsAllSubscribers()
    {
        var fanOut = MakeFanOut();
        var sub1 = new Mock<IRecognitionSubscriber>();
        var sub2 = new Mock<IRecognitionSubscriber>();
        fanOut.AddSubscriber(sub1.Object);
        fanOut.AddSubscriber(sub2.Object);

        var result = MakeResult();
        fanOut.OnPartialUpdate(result);

        sub1.Verify(s => s.OnPartialUpdate(result), Times.Once);
        sub2.Verify(s => s.OnPartialUpdate(result), Times.Once);
    }

    [Fact]
    public void OnFinalization_CallsAllSubscribers()
    {
        var fanOut = MakeFanOut();
        var sub1 = new Mock<IRecognitionSubscriber>();
        var sub2 = new Mock<IRecognitionSubscriber>();
        fanOut.AddSubscriber(sub1.Object);
        fanOut.AddSubscriber(sub2.Object);

        var result = MakeResult();
        fanOut.OnFinalization(result);

        sub1.Verify(s => s.OnFinalization(result), Times.Once);
        sub2.Verify(s => s.OnFinalization(result), Times.Once);
    }

    [Fact]
    public void OnPartialUpdate_SubscriberThrows_OtherSubscribersStillCalled()
    {
        var fanOut = MakeFanOut();
        var faultySubscriber = new Mock<IRecognitionSubscriber>();
        faultySubscriber.Setup(s => s.OnPartialUpdate(It.IsAny<RecognitionResult>()))
            .Throws<InvalidOperationException>();
        var goodSubscriber = new Mock<IRecognitionSubscriber>();

        fanOut.AddSubscriber(faultySubscriber.Object);
        fanOut.AddSubscriber(goodSubscriber.Object);

        var result = MakeResult();
        var act = () => fanOut.OnPartialUpdate(result);

        act.Should_NotThrow();
        goodSubscriber.Verify(s => s.OnPartialUpdate(result), Times.Once);
    }

    [Fact]
    public void OnFinalization_SubscriberThrows_OtherSubscribersStillCalled()
    {
        var fanOut = MakeFanOut();
        var faultySubscriber = new Mock<IRecognitionSubscriber>();
        faultySubscriber.Setup(s => s.OnFinalization(It.IsAny<RecognitionResult>()))
            .Throws<InvalidOperationException>();
        var goodSubscriber = new Mock<IRecognitionSubscriber>();

        fanOut.AddSubscriber(faultySubscriber.Object);
        fanOut.AddSubscriber(goodSubscriber.Object);

        var result = MakeResult();
        var act = () => fanOut.OnFinalization(result);

        act.Should_NotThrow();
        goodSubscriber.Verify(s => s.OnFinalization(result), Times.Once);
    }

    [Fact]
    public void AddSubscriber_DuringDispatch_IsThreadSafe()
    {
        var fanOut = MakeFanOut();
        var dispatching = new ManualResetEventSlim(false);
        var addDone = new ManualResetEventSlim(false);
        var result = MakeResult();

        var blockingSubscriber = new Mock<IRecognitionSubscriber>();
        blockingSubscriber.Setup(s => s.OnPartialUpdate(It.IsAny<RecognitionResult>()))
            .Callback(() =>
            {
                dispatching.Set();
                Thread.Sleep(20);
            });

        fanOut.AddSubscriber(blockingSubscriber.Object);

        Exception? caughtException = null;

        var dispatchThread = new Thread(() =>
        {
            try { fanOut.OnPartialUpdate(result); }
            catch (Exception ex) { caughtException = ex; }
        });
        dispatchThread.Start();

        dispatching.Wait();

        var addThread = new Thread(() =>
        {
            try
            {
                var newSub = new Mock<IRecognitionSubscriber>();
                fanOut.AddSubscriber(newSub.Object);
            }
            catch (Exception ex) { caughtException = ex; }
            finally { addDone.Set(); }
        });
        addThread.Start();

        addDone.Wait();
        dispatchThread.Join();

        Assert.Null(caughtException);
    }

    [Fact]
    public void ConcurrentDispatch_MultipleThreads_AllSubscribersReceiveAllCalls()
    {
        var fanOut = MakeFanOut();
        const int threadCount = 8;
        const int dispatchesPerThread = 50;

        var callCount = 0;
        var countingSubscriber = new Mock<IRecognitionSubscriber>();
        countingSubscriber.Setup(s => s.OnPartialUpdate(It.IsAny<RecognitionResult>()))
            .Callback(() => Interlocked.Increment(ref callCount));

        fanOut.AddSubscriber(countingSubscriber.Object);

        var result = MakeResult();
        var threads = Enumerable.Range(0, threadCount)
            .Select(_ => new Thread(() =>
            {
                for (int i = 0; i < dispatchesPerThread; i++)
                    fanOut.OnPartialUpdate(result);
            }))
            .ToList();

        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join());

        Assert.Equal(threadCount * dispatchesPerThread, callCount);
    }
}

/// <summary>
/// Minimal assertion helper used instead of FluentAssertions to avoid extra dependencies.
/// </summary>
file static class ActionExtensions
{
    public static void Should_NotThrow(this Action action)
    {
        var ex = Record.Exception(action);
        Assert.Null(ex);
    }
}
