using Microsoft.Extensions.Logging.Abstractions;
using SttClient.Insertion;
using SttClient.Recognition;
using Xunit;

namespace SttClient.Tests.Insertion;

/// <summary>
/// Tests for <see cref="TextInserter"/> — verifies that finalized text is typed when enabled
/// and silently skipped when disabled.
/// </summary>
public class TextInserterTests
{
    private sealed class FakeKeyboardSimulator : IKeyboardSimulator
    {
        public List<string> TypedTexts { get; } = [];

        public void TypeText(string text) => TypedTexts.Add(text);
    }

    private static RecognitionResult MakeResult(string text) =>
        new(text, 0.0, 1.0, null, [], null);

    [Fact]
    public void OnFinalization_WhenEnabled_TypesText()
    {
        var keyboard = new FakeKeyboardSimulator();
        var inserter = new TextInserter(keyboard, NullLogger<TextInserter>.Instance);
        inserter.SetEnabled(true);

        inserter.OnFinalization(MakeResult("hello"));

        Assert.Equal(["hello"], keyboard.TypedTexts);
    }

    [Fact]
    public void OnFinalization_WhenDisabled_DoesNotTypeText()
    {
        var keyboard = new FakeKeyboardSimulator();
        var inserter = new TextInserter(keyboard, NullLogger<TextInserter>.Instance);
        inserter.SetEnabled(false);

        inserter.OnFinalization(MakeResult("hello"));

        Assert.Empty(keyboard.TypedTexts);
    }

    [Fact]
    public void OnFinalization_AfterDisable_StopsTyping()
    {
        var keyboard = new FakeKeyboardSimulator();
        var inserter = new TextInserter(keyboard, NullLogger<TextInserter>.Instance);
        inserter.SetEnabled(true);
        inserter.OnFinalization(MakeResult("first"));
        inserter.SetEnabled(false);

        inserter.OnFinalization(MakeResult("second"));

        Assert.Equal(["first"], keyboard.TypedTexts);
    }

    [Fact]
    public void OnPartialUpdate_WhenEnabled_DoesNotTypeText()
    {
        var keyboard = new FakeKeyboardSimulator();
        var inserter = new TextInserter(keyboard, NullLogger<TextInserter>.Instance);
        inserter.SetEnabled(true);

        inserter.OnPartialUpdate(MakeResult("partial"));

        Assert.Empty(keyboard.TypedTexts);
    }
}
