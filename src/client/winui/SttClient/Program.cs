using Microsoft.Windows.ApplicationModel.DynamicDependency;
using System.Runtime.InteropServices;

namespace SttClient;

internal static class Program
{
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int MessageBoxW(nint hWnd, string text, string caption, uint type);

    private const uint MB_OK = 0x0;
    private const uint MB_ICONERROR = 0x10;

    /// <summary>
    /// Custom entry point that bootstraps the Windows App SDK before starting the WinUI application.
    /// Bootstrap.Initialize is a no-op when running inside an MSIX package, so this works for
    /// both unpackaged (direct exe) and packaged (MSIX) deployment.
    /// If the Windows App Runtime is not installed, shows a MessageBox with download instructions.
    /// </summary>
    [STAThread]
    static void Main(string[] args)
    {
        try
        {
            // 0x00010008 encodes major=1, minor=8: matches Microsoft.WindowsAppSDK 1.8.x in the csproj
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
