using ClientOrchestrator = SttClient.Orchestration.ClientOrchestrator;
using Microsoft.UI;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;
using SttClient.Insertion;
using SttClient.ViewModels;
using System.ComponentModel;

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
        ViewModel.PropertyChanged += OnViewModelPropertyChanged;
        Activated += OnFirstActivated;
        Closed += OnClosed;
    }

    private void OnFirstActivated(object sender, WindowActivatedEventArgs e)
    {
        Activated -= OnFirstActivated;
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(ViewModel.IsPaused) or nameof(ViewModel.IsRunningOrPaused))
            UpdatePauseButtonState();

        if (e.PropertyName is nameof(ViewModel.FinalizedText) or nameof(ViewModel.PartialText))
            UpdateTranscriptVisibility();
    }

    private void UpdatePauseButtonState()
    {
        if (ViewModel.IsRunningOrPaused)
        {
            PauseButtonIcon.Glyph = ViewModel.IsPaused ? "\uE768" : "\uE769";
            PauseButtonLabel.Text = ViewModel.IsPaused ? "Resume" : "Pause";
            StatusLabel.Text = ViewModel.IsPaused ? "Paused" : "Running";
            StatusDot.Fill = ViewModel.IsPaused
                ? (Brush)Application.Current.Resources["SystemFillColorCautionBrush"]
                : (Brush)Application.Current.Resources["SystemFillColorSuccessBrush"];
        }
        else
        {
            PauseButtonIcon.Glyph = "\uE768";
            PauseButtonLabel.Text = "Start";
            StatusLabel.Text = "Idle";
            StatusDot.Fill = (Brush)Application.Current.Resources["SystemFillColorAttentionBrush"];
        }
    }

    private void UpdateTranscriptVisibility()
    {
        bool hasText = !string.IsNullOrEmpty(ViewModel.FinalizedText)
                    || !string.IsNullOrEmpty(ViewModel.PartialText);
        EmptyStatePlaceholder.Visibility = hasText ? Visibility.Collapsed : Visibility.Visible;
        TranscriptScrollViewer.Visibility = hasText ? Visibility.Visible : Visibility.Collapsed;
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
        InsertionLabel.Text = InsertionToggle.IsChecked == true ? "Insert: ON" : "Insert: OFF";
    }

    private void ClearButton_Click(object sender, RoutedEventArgs e) { }

    private void SettingsButton_Click(object sender, RoutedEventArgs e) { }

    private async void OnClosed(object sender, WindowEventArgs args)
    {
        args.Handled = true;
        ViewModel.PropertyChanged -= OnViewModelPropertyChanged;
        await _orchestrator.StopAsync();
        Closed -= OnClosed;
        Close();
    }
}
