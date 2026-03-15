using Microsoft.Windows.ApplicationModel.DynamicDependency;
using System.Runtime.InteropServices;

namespace SttModelDownloader;

/// <summary>
/// Application entry point. Bootstraps the Windows App SDK before starting WinUI.
/// </summary>
internal static class Program
{
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int MessageBoxW(nint hWnd, string text, string caption, uint type);

    private const uint MB_OK = 0x0;
    private const uint MB_ICONERROR = 0x10;

    /// <summary>
    /// Initializes the Windows App Runtime and launches the WinUI application.
    /// Shows a MessageBox if the runtime is not installed.
    /// </summary>
    [STAThread]
    static void Main()
    {
        try
        {
            // 0x00010008 encodes major=1, minor=8: matches Microsoft.WindowsAppSDK 1.8.x
            Bootstrap.Initialize(0x00010008);
        }
        catch (Exception ex)
        {
            MessageBoxW(
                0,
                $"Windows App Runtime 1.8 is required but was not found.\n\n" +
                $"Download it from:\nhttps://learn.microsoft.com/windows/apps/windows-app-sdk/downloads\n\n" +
                $"Details: {ex.Message}",
                "Missing Windows App Runtime",
                MB_OK | MB_ICONERROR);
            return;
        }

        try
        {
            WinRT.ComWrappersSupport.InitializeComWrappers();
            Microsoft.UI.Xaml.Application.Start(p =>
            {
                var context = new Microsoft.UI.Dispatching.DispatcherQueueSynchronizationContext(
                    Microsoft.UI.Dispatching.DispatcherQueue.GetForCurrentThread());
                System.Threading.SynchronizationContext.SetSynchronizationContext(context);
                new App();
            });
        }
        finally
        {
            Bootstrap.Shutdown();
        }
    }
}
