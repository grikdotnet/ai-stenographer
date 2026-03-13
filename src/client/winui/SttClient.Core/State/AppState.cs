namespace SttClient.State;

/// <summary>
/// Application lifecycle states. Terminal state is Shutdown.
/// </summary>
public enum AppState
{
    Starting,
    Running,
    Paused,
    Shutdown
}
