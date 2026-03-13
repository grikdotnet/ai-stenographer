using Microsoft.Extensions.Logging;

namespace SttClient.Recognition;

/// <summary>
/// Thread-safe fan-out dispatcher that forwards recognition results to all registered subscribers.
/// Uses a copy-on-notify pattern so subscribers can be safely added from any thread, even during dispatch.
/// Implements <see cref="IRecognitionSubscriber"/> so it can be composed in the pipeline.
/// </summary>
public sealed class RecognitionResultFanOut : IRecognitionSubscriber
{
    private readonly ILogger<RecognitionResultFanOut> _logger;
    private readonly List<IRecognitionSubscriber> _subscribers = [];
    private readonly object _lock = new();

    /// <summary>
    /// Initializes a new fan-out with no subscribers.
    /// </summary>
    /// <param name="logger">Logger for subscriber exception warnings.</param>
    public RecognitionResultFanOut(ILogger<RecognitionResultFanOut> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Registers a subscriber to receive future recognition events.
    /// Safe to call from any thread, including during an active dispatch.
    /// </summary>
    /// <param name="subscriber">The subscriber to add.</param>
    public void AddSubscriber(IRecognitionSubscriber subscriber)
    {
        lock (_lock)
        {
            _subscribers.Add(subscriber);
        }
    }

    /// <summary>
    /// Forwards a partial recognition result to all registered subscribers.
    /// If a subscriber throws, the exception is caught and logged; remaining subscribers still receive the call.
    /// </summary>
    /// <param name="result">The partial recognition result.</param>
    public void OnPartialUpdate(RecognitionResult result)
    {
        foreach (var subscriber in CopySubscribers())
            InvokeSubscriber(subscriber, s => s.OnPartialUpdate(result), nameof(OnPartialUpdate));
    }

    /// <summary>
    /// Forwards a final recognition result to all registered subscribers.
    /// If a subscriber throws, the exception is caught and logged; remaining subscribers still receive the call.
    /// </summary>
    /// <param name="result">The final recognition result.</param>
    public void OnFinalization(RecognitionResult result)
    {
        foreach (var subscriber in CopySubscribers())
            InvokeSubscriber(subscriber, s => s.OnFinalization(result), nameof(OnFinalization));
    }

    private List<IRecognitionSubscriber> CopySubscribers()
    {
        lock (_lock)
        {
            return new List<IRecognitionSubscriber>(_subscribers);
        }
    }

    private void InvokeSubscriber(IRecognitionSubscriber subscriber, Action<IRecognitionSubscriber> action, string methodName)
    {
        try
        {
            action(subscriber);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Subscriber {Subscriber} threw during {Method}", subscriber.GetType().Name, methodName);
        }
    }
}
