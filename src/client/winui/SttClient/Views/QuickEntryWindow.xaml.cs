using Microsoft.UI.Xaml;
using SttClient.QuickEntry;
using SttClient.ViewModels;

namespace SttClient.Views;

/// <summary>
/// Borderless floating popup window for the QuickEntry feature.
/// Displays live transcription text and accepts Enter (submit) / Escape (cancel).
///
/// Responsibilities:
/// - Implements <see cref="IQuickEntryPopup"/> so <see cref="QuickEntryController"/> can
///   show/hide/update it without depending on WinUI directly.
/// - Routes keyboard events: Enter → submit callback, Escape → cancel callback.
/// </summary>
public sealed partial class QuickEntryWindow : Window, IQuickEntryPopup
{
    private Action? _onSubmit;
    private Action? _onCancel;

    /// <summary>Gets the ViewModel bound to this window.</summary>
    public QuickEntryViewModel ViewModel { get; }

    /// <summary>
    /// Initializes <see cref="QuickEntryWindow"/> with its ViewModel.
    /// </summary>
    /// <param name="viewModel">ViewModel providing the live text binding.</param>
    public QuickEntryWindow(QuickEntryViewModel viewModel)
    {
        ViewModel = viewModel;
        InitializeComponent();

        Content.KeyDown += OnKeyDown;
    }

    /// <inheritdoc/>
    public void Show(Action onSubmit, Action onCancel)
    {
        _onSubmit = onSubmit;
        _onCancel = onCancel;
        Activate();
    }

    /// <inheritdoc/>
    public void Hide()
    {
        AppWindow.Hide();
    }

    /// <inheritdoc/>
    public void SetText(string text)
    {
        ViewModel.SetLiveText(text);
    }

    private void OnKeyDown(object sender, Microsoft.UI.Xaml.Input.KeyRoutedEventArgs e)
    {
        if (e.Key == Windows.System.VirtualKey.Enter)
        {
            e.Handled = true;
            _onSubmit?.Invoke();
        }
        else if (e.Key == Windows.System.VirtualKey.Escape)
        {
            e.Handled = true;
            _onCancel?.Invoke();
        }
    }
}
