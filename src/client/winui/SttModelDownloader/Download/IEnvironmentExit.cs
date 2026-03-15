namespace SttModelDownloader.Download;

/// <summary>
/// Abstraction over process exit, enabling unit testing of shutdown logic
/// without terminating the test runner.
/// </summary>
public interface IEnvironmentExit
{
    /// <summary>
    /// Terminates the current process with the given exit code.
    /// </summary>
    /// <param name="exitCode">Exit code to pass to the OS.</param>
    void Exit(int exitCode);
}

/// <summary>
/// Production implementation that delegates to Environment.Exit.
/// </summary>
public sealed class EnvironmentExit : IEnvironmentExit
{
    /// <inheritdoc />
    public void Exit(int exitCode) => Environment.Exit(exitCode);
}
