namespace SttClient.ViewModels;

/// <summary>
/// Abstraction over <c>DispatcherQueue.TryEnqueue</c> for marshalling work to the UI thread.
/// Inject a synchronous fake in tests; use the real WinUI adapter in production.
/// </summary>
public interface IDispatcherQueueAdapter
{
    /// <summary>
    /// Schedules <paramref name="action"/> to run on the UI thread.
    /// </summary>
    /// <param name="action">The delegate to execute on the UI thread.</param>
    /// <returns><c>true</c> if the item was enqueued; <c>false</c> if the queue is unavailable (e.g. during shutdown).</returns>
    bool TryEnqueue(Action action);
}
