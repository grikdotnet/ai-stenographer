using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace SttClient.Views;

/// <summary>
/// Splash page shown while the client connects to the server.
/// Has two visual states:
/// - Connecting: spinner + "Connecting…" label (default).
/// - Error: error message text + Exit button (shown by <see cref="ShowError"/>).
/// </summary>
public sealed partial class LoadingPage : Page
{
    /// <summary>Initializes the loading page in the Connecting state.</summary>
    public LoadingPage()
    {
        InitializeComponent();
    }

    /// <summary>
    /// Switches the page to the Error state, displaying the given message.
    /// Must be called on the UI thread.
    /// </summary>
    /// <param name="message">The error message to display.</param>
    public void ShowError(string message)
    {
        ConnectingSpinner.Visibility = Visibility.Collapsed;
        ConnectingLabel.Visibility = Visibility.Collapsed;
        ErrorMessageText.Text = message;
        ErrorMessageText.Visibility = Visibility.Visible;
        ExitButton.Visibility = Visibility.Visible;
    }

    private void ExitButton_Click(object sender, RoutedEventArgs e)
    {
        Application.Current.Exit();
    }
}
