using ClientOrchestrator = SttClient.Orchestration.ClientOrchestrator;
using Microsoft.UI.Xaml;
using SttClient.Insertion;
using SttClient.ViewModels;

namespace SttClient.Views;

/// <summary>
/// Main transcription window. Displays finalized and partial transcript text,
/// a pause/resume button, and an insertion toggle.
/// The code-behind is intentionally thin — all state lives in <see cref="MainWindowViewModel"/>.
/// </summary>
public sealed partial class MainWindow : Window
{
    private readonly ClientOrchestrator _orchestrator;
    private readonly InsertionController _insertionController;

    /// <summary>Gets the ViewModel bound to this window.</summary>
    public MainWindowViewModel ViewModel { get; }

    /// <summary>
    /// Initializes <see cref="MainWindow"/> with its ViewModel and orchestrator.
    /// </summary>
    /// <param name="viewModel">The view model providing bindable state.</param>
    /// <param name="orchestrator">Used to pause/resume on button click.</param>
    /// <param name="insertionController">Toggled by the insertion toggle button.</param>
    public MainWindow(MainWindowViewModel viewModel, ClientOrchestrator orchestrator, InsertionController insertionController)
    {
        ViewModel = viewModel;
        _orchestrator = orchestrator;
        _insertionController = insertionController;
        InitializeComponent();
        Closed += OnClosed;
    }

    private void PauseButton_Click(object sender, RoutedEventArgs e)
    {
        if (ViewModel.IsPaused)
            _orchestrator.Resume();
        else
            _orchestrator.Pause();
    }

    private void InsertionToggle_Click(object sender, RoutedEventArgs e)
    {
        _insertionController.Toggle();
    }

    private void OnClosed(object sender, WindowEventArgs args)
    {
        _ = _orchestrator.StopAsync();
    }
}
