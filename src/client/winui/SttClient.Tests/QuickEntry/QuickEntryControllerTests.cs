using Microsoft.Extensions.Logging.Abstractions;
using SttClient.Insertion;
using SttClient.QuickEntry;
using SttClient.Recognition;
using Xunit;

namespace SttClient.Tests.QuickEntry;

/// <summary>
/// Tests for <see cref="QuickEntryController"/> — verifies popup show/hide lifecycle,
/// submit (types text + restores focus), and cancel semantics.
/// </summary>
public class QuickEntryControllerTests
{
    private sealed class FakeKeyboardSimulator : IKeyboardSimulator
    {
        public List<string> TypedTexts { get; } = [];
        public void TypeText(string text) => TypedTexts.Add(text);
    }

    private sealed class FakeFocusTracker : IFocusTracker
    {
        public int SaveCount { get; private set; }
        public int RestoreCount { get; private set; }

        public void SaveFocus() => SaveCount++;
        public void RestoreFocus() => RestoreCount++;
    }

    private sealed class FakePopupHotkeyRegistrar : IPopupHotkeyRegistrar
    {
        public bool IsRegistered { get; private set; }
        public Action? RegisteredSubmit { get; private set; }
        public Action? RegisteredCancel { get; private set; }

        public void RegisterPopupHotkeys(Action onSubmit, Action onCancel)
        {
            IsRegistered = true;
            RegisteredSubmit = onSubmit;
            RegisteredCancel = onCancel;
        }

        public void UnregisterPopupHotkeys()
        {
            IsRegistered = false;
            RegisteredSubmit = null;
            RegisteredCancel = null;
        }
    }

    private sealed class FakePopup : IQuickEntryPopup
    {
        public bool IsVisible { get; private set; }
        public string DisplayedText { get; private set; } = string.Empty;
        public Action? OnSubmitCallback { get; set; }
        public Action? OnCancelCallback { get; set; }

        public void Show(Action onSubmit, Action onCancel)
        {
            IsVisible = true;
            OnSubmitCallback = onSubmit;
            OnCancelCallback = onCancel;
        }

        public void Hide() => IsVisible = false;
        public void SetText(string text) => DisplayedText = text;
    }

    private static RecognitionResult MakeResult(string text) =>
        new(text, 0.0, 1.0, null, [], null);

    private static (QuickEntryController controller, FakeKeyboardSimulator keyboard,
        FakeFocusTracker focus, FakePopup popup, QuickEntrySubscriber subscriber,
        FakePopupHotkeyRegistrar registrar)
        MakeController()
    {
        var keyboard = new FakeKeyboardSimulator();
        var focus = new FakeFocusTracker();
        var popup = new FakePopup();
        var registrar = new FakePopupHotkeyRegistrar();
        var subscriber = new QuickEntrySubscriber(_ => { }, NullLogger<QuickEntrySubscriber>.Instance);
        var controller = new QuickEntryController(
            subscriber, focus, keyboard, popup, registrar,
            NullLogger<QuickEntryController>.Instance);
        return (controller, keyboard, focus, popup, subscriber, registrar);
    }

    [Fact]
    public void OnHotkey_ShowsPopup()
    {
        var (controller, _, _, popup, _, _) = MakeController();

        controller.OnHotkey();

        Assert.True(popup.IsVisible);
    }

    [Fact]
    public void OnHotkey_SavesFocus()
    {
        var (controller, _, focus, _, _, _) = MakeController();

        controller.OnHotkey();

        Assert.Equal(1, focus.SaveCount);
    }

    [Fact]
    public void OnHotkey_WhenAlreadyShowing_HidesPopup()
    {
        var (controller, _, _, popup, _, _) = MakeController();
        controller.OnHotkey();

        controller.OnHotkey();

        Assert.False(popup.IsVisible);
    }

    [Fact]
    public void Submit_TypesAccumulatedText()
    {
        var (controller, keyboard, _, _, subscriber, _) = MakeController();
        controller.OnHotkey();
        subscriber.OnFinalization(MakeResult("hello world"));

        controller.Submit();

        Assert.Equal(["hello world"], keyboard.TypedTexts);
    }

    [Fact]
    public void Submit_HidesPopup()
    {
        var (controller, _, _, popup, _, _) = MakeController();
        controller.OnHotkey();

        controller.Submit();

        Assert.False(popup.IsVisible);
    }

    [Fact]
    public void Submit_RestoresFocus()
    {
        var (controller, _, focus, _, _, _) = MakeController();
        controller.OnHotkey();

        controller.Submit();

        Assert.Equal(1, focus.RestoreCount);
    }

    [Fact]
    public void Submit_WithEmptyText_DoesNotTypeAnything()
    {
        var (controller, keyboard, _, _, _, _) = MakeController();
        controller.OnHotkey();

        controller.Submit();

        Assert.Empty(keyboard.TypedTexts);
    }

    [Fact]
    public void Cancel_HidesPopup()
    {
        var (controller, _, _, popup, _, _) = MakeController();
        controller.OnHotkey();

        controller.Cancel();

        Assert.False(popup.IsVisible);
    }

    [Fact]
    public void Cancel_RestoresFocus()
    {
        var (controller, _, focus, _, _, _) = MakeController();
        controller.OnHotkey();

        controller.Cancel();

        Assert.Equal(1, focus.RestoreCount);
    }

    [Fact]
    public void Cancel_DoesNotTypeText()
    {
        var (controller, keyboard, _, _, subscriber, _) = MakeController();
        controller.OnHotkey();
        subscriber.OnFinalization(MakeResult("some text"));

        controller.Cancel();

        Assert.Empty(keyboard.TypedTexts);
    }

    [Fact]
    public void OnHotkey_RegistersPopupHotkeys()
    {
        var (controller, _, _, _, _, registrar) = MakeController();

        controller.OnHotkey();

        Assert.True(registrar.IsRegistered);
    }

    [Fact]
    public void Submit_UnregistersPopupHotkeys()
    {
        var (controller, _, _, _, _, registrar) = MakeController();
        controller.OnHotkey();

        controller.Submit();

        Assert.False(registrar.IsRegistered);
    }

    [Fact]
    public void Cancel_UnregistersPopupHotkeys()
    {
        var (controller, _, _, _, _, registrar) = MakeController();
        controller.OnHotkey();

        controller.Cancel();

        Assert.False(registrar.IsRegistered);
    }

    [Fact]
    public void RegisteredEnterCallback_SubmitsText()
    {
        var (controller, keyboard, _, _, subscriber, registrar) = MakeController();
        controller.OnHotkey();
        subscriber.OnFinalization(MakeResult("hello"));

        registrar.RegisteredSubmit!();

        Assert.Equal(["hello"], keyboard.TypedTexts);
    }

    [Fact]
    public void RegisteredEscapeCallback_CancelsWithoutTyping()
    {
        var (controller, keyboard, _, _, subscriber, registrar) = MakeController();
        controller.OnHotkey();
        subscriber.OnFinalization(MakeResult("hello"));

        registrar.RegisteredCancel!();

        Assert.Empty(keyboard.TypedTexts);
    }

    [Fact]
    public void Submit_ClearsPopupText()
    {
        var (controller, _, _, popup, subscriber, _) = MakeController();
        controller.OnHotkey();
        subscriber.OnFinalization(MakeResult("hello world"));

        controller.Submit();

        Assert.Equal(string.Empty, popup.DisplayedText);
    }

    [Fact]
    public void Cancel_ClearsPopupText()
    {
        var (controller, _, _, popup, subscriber, _) = MakeController();
        controller.OnHotkey();
        subscriber.OnFinalization(MakeResult("hello world"));

        controller.Cancel();

        Assert.Equal(string.Empty, popup.DisplayedText);
    }
}
