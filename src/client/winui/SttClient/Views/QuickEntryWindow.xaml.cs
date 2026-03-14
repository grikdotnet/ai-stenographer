using System.Runtime.InteropServices;
using Microsoft.UI.Xaml;
using SttClient.QuickEntry;
using SttClient.ViewModels;
using Windows.Graphics;

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

    [DllImport("user32.dll")]
    private static extern bool SetWindowPos(nint hWnd, nint hWndInsertAfter, int x, int y, int cx, int cy, uint uFlags);

    [DllImport("user32.dll")]
    private static extern nint SetActiveWindow(nint hWnd);

    private const nint HwndTopmost = -1;
    private const uint SwpNomove = 0x0002;
    private const uint SwpNosize = 0x0001;
    private const uint SwpNoactivate = 0x0010;

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
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(null);

        Content.KeyDown += OnKeyDown;
    }

    /// <inheritdoc/>
    public void Show(Action onSubmit, Action onCancel)
    {
        _onSubmit = onSubmit;
        _onCancel = onCancel;
        AppWindow.Resize(new SizeInt32(400, 80));
        // Show without activating so we don't steal focus from the target window.
        AppWindow.Show(activateWindow: false);
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        SetWindowPos(hwnd, HwndTopmost, 0, 0, 0, 0, SwpNomove | SwpNosize | SwpNoactivate);
        // SetActiveWindow makes WinUI commit the DComp render tree for this window
        // (so text updates paint immediately) without changing the OS foreground process.
        SetActiveWindow(hwnd);
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
