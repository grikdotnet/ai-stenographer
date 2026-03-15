using Microsoft.Extensions.Logging;

namespace SttClient.State;

/// <summary>
/// Thread-safe state machine managing application lifecycle transitions.
/// Notifies registered observers after each successful transition, invoking them outside the lock.
/// </summary>
public class AppStateManager
{
    private readonly ILogger<AppStateManager> _logger;
    private readonly List<Action<AppState, AppState>> _observers = [];
    private readonly object _lock = new();
    private AppState _currentState = AppState.Starting;

    private static readonly Dictionary<AppState, HashSet<AppState>> ValidTransitions = new()
    {
        [AppState.Starting] = [AppState.Running, AppState.Shutdown],
        [AppState.Running]  = [AppState.Paused,  AppState.Shutdown],
        [AppState.Paused]   = [AppState.Running,  AppState.Shutdown],
        [AppState.Shutdown] = [],
    };

    /// <summary>
    /// Initializes the state machine in the <see cref="AppState.Starting"/> state.
    /// </summary>
    /// <param name="logger">Logger for state transition diagnostics.</param>
    public AppStateManager(ILogger<AppStateManager> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Gets the current application state. Thread-safe.
    /// </summary>
    public AppState CurrentState
    {
        get { lock (_lock) return _currentState; }
    }

    /// <summary>
    /// Transitions to <paramref name="newState"/>.
    /// Shutdown→Shutdown is a silent no-op. All other invalid transitions throw.
    /// </summary>
    /// <param name="newState">The target state.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the transition from <see cref="CurrentState"/> to <paramref name="newState"/> is not allowed.
    /// </exception>
    public void SetState(AppState newState)
    {
        List<Action<AppState, AppState>> observerSnapshot;
        AppState oldState;

        lock (_lock)
        {
            oldState = _currentState;

            if (oldState == AppState.Shutdown && newState == AppState.Shutdown)
            {
                _logger.LogDebug("State transition no-op: Shutdown -> Shutdown");
                return;
            }

            if (!ValidTransitions[oldState].Contains(newState))
            {
                throw new InvalidOperationException(
                    $"Invalid state transition: {oldState} -> {newState}");
            }

            _currentState = newState;
            observerSnapshot = [.._observers];
        }

        _logger.LogInformation("State transition: {OldState} -> {NewState}", oldState, newState);

        foreach (var observer in observerSnapshot)
        {
            try
            {
                observer(oldState, newState);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Observer threw during state transition {OldState} -> {NewState}", oldState, newState);
            }
        }
    }

    /// <summary>
    /// Registers an observer invoked after each successful state transition.
    /// </summary>
    /// <param name="observer">
    /// Callback receiving (oldState, newState). Called outside the internal lock.
    /// </param>
    public void AddObserver(Action<AppState, AppState> observer)
    {
        lock (_lock)
        {
            _observers.Add(observer);
        }
    }

    /// <summary>
    /// Removes a previously registered observer. No-op if the observer is not found.
    /// </summary>
    /// <param name="observer">The observer delegate to remove.</param>
    public void RemoveObserver(Action<AppState, AppState> observer)
    {
        lock (_lock)
        {
            _observers.Remove(observer);
        }
    }
}
