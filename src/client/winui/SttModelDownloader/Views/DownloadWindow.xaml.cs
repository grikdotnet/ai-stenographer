using System.ComponentModel;
using Microsoft.UI;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using SttModelDownloader.Download;
using Windows.Graphics;

namespace SttModelDownloader.Views;

/// <summary>
/// WinUI window for the model download dialog.
/// Binds to <see cref="DownloadViewModel"/> and wires button clicks.
/// Subscribes to <see cref="INotifyPropertyChanged"/> to update visibility-based UI states
/// that cannot be expressed with simple x:Bind expressions.
/// </summary>
public sealed partial class DownloadWindow : Window
{
    private readonly DownloadViewModel _viewModel;
    private CancellationTokenSource? _downloadCts;

    /// <summary>
    /// Initializes the window with the given ViewModel, configures fixed size and position,
    /// and subscribes to property changes for dynamic UI updates.
    /// </summary>
    /// <param name="viewModel">The ViewModel driving download state.</param>
    public DownloadWindow(DownloadViewModel viewModel)
    {
        _viewModel = viewModel;
        InitializeComponent();

        ConfigureWindow();
        UpdateUiFromViewModel();
        _viewModel.PropertyChanged += OnViewModelPropertyChanged;
        AppWindow.Closing += OnWindowClosing;
    }

    private void ConfigureWindow()
    {
        var appWindow = AppWindow;
        appWindow.Resize(new SizeInt32(500, 380));

        var displayArea = DisplayArea.GetFromWindowId(appWindow.Id, DisplayAreaFallback.Primary);
        var x = (displayArea.WorkArea.Width - 500) / 2;
        var y = (displayArea.WorkArea.Height - 380) / 2;
        appWindow.Move(new PointInt32(x, y));

        if (appWindow.Presenter is OverlappedPresenter presenter)
            presenter.IsResizable = false;
    }

    private async void OnDownloadClicked(object sender, RoutedEventArgs e)
    {
        _downloadCts = new CancellationTokenSource();
        await _viewModel.StartDownloadAsync(_downloadCts.Token);
        if (_viewModel.State == DownloadState.Complete)
            Close();
    }

    private void OnExitClicked(object sender, RoutedEventArgs e)
    {
        _viewModel.NotifyWindowClosing();
    }

    private void OnWindowClosing(AppWindow sender, AppWindowClosingEventArgs e)
    {
        _viewModel.NotifyWindowClosing();
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        DispatcherQueue.TryEnqueue(UpdateUiFromViewModel);
    }

    /// <summary>
    /// Synchronizes all dynamic UI elements with current ViewModel state.
    /// Called on every PropertyChanged notification and on initial construction.
    /// </summary>
    private void UpdateUiFromViewModel()
    {
        StatusTextBlock.Text = _viewModel.StatusText;

        DownloadProgressBar.Value = _viewModel.ProgressPercentage;
        DownloadProgressBar.Visibility = _viewModel.IsDownloading
            ? Visibility.Visible
            : Visibility.Collapsed;

        var hasError = !string.IsNullOrEmpty(_viewModel.ErrorMessage);
        ErrorTextBlock.Text = _viewModel.ErrorMessage;
        ErrorTextBlock.Visibility = hasError ? Visibility.Visible : Visibility.Collapsed;

        DownloadButton.IsEnabled = _viewModel.CanDownload || _viewModel.CanRetry;
        DownloadButton.Content = _viewModel.CanRetry ? "Retry" : "Download";
    }
}
