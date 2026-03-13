using Microsoft.UI.Xaml;
using System;
using System.IO;

namespace SttClient
{
    /// <summary>
    /// Application entry point. Startup logic is in <see cref="AppStartup"/>.
    /// </summary>
    public partial class App : Application
    {
        private readonly AppStartup _startup = new();

        public App()
        {
            AppDomain.CurrentDomain.UnhandledException += (_, e) =>
                File.AppendAllText("logs/crash.log",
                    $"[{DateTime.Now:O}] UnhandledException (IsTerminating={e.IsTerminating})\n{e.ExceptionObject}\n\n");

            UnhandledException += (_, e) =>
            {
                e.Handled = true;
                File.AppendAllText("logs/crash.log",
                    $"[{DateTime.Now:O}] WinUI UnhandledException\n{e.Exception}\n\n");
            };
        }

        protected override void OnLaunched(LaunchActivatedEventArgs args)
        {
            _startup.Run();
        }
    }
}
