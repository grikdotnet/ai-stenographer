using Microsoft.UI.Xaml;

namespace SttClient.Views;

/// <summary>
/// Host window for the <see cref="LoadingPage"/>, shown during server connection.
/// </summary>
public sealed partial class LoadingWindow : Window
{
    /// <summary>
    /// Initializes <see cref="LoadingWindow"/> and navigates the frame to the given page.
    /// </summary>
    /// <param name="page">The loading page instance to display.</param>
    public LoadingWindow(LoadingPage page)
    {
        InitializeComponent();
        RootFrame.Content = page;
    }
}
