using Microsoft.Extensions.Logging.Abstractions;
using SttClient.Formatting;
using SttClient.Recognition;
using Xunit;

namespace SttClient.Tests.Formatting;

/// <summary>
/// Tests for <see cref="TextFormatter"/> covering partial update rendering,
/// finalization accumulation, duplicate suppression, and paragraph break insertion.
/// </summary>
public class TextFormatterTests
{
    private static RecognitionResult MakeResult(string text, double start = 0.0, double end = 1.0) =>
        new(text, start, end, null, [], null);

    private static (TextFormatter formatter, List<DisplayInstructions> captured) MakeFormatter()
    {
        var captured = new List<DisplayInstructions>();
        var formatter = new TextFormatter(
            instructions => captured.Add(instructions),
            NullLogger<TextFormatter>.Instance
        );
        return (formatter, captured);
    }

    [Fact]
    public void OnPartialUpdate_SetsPartialText()
    {
        var (formatter, captured) = MakeFormatter();

        formatter.OnPartialUpdate(MakeResult("hello world"));

        Assert.Single(captured);
        Assert.Equal(DisplayAction.RerenderAll, captured[0].Action);
        Assert.Single(captured[0].PreliminarySegments);
        Assert.Equal("hello world", captured[0].PreliminarySegments[0]);
    }

    [Fact]
    public void OnPartialUpdate_DoesNotChangeFinalizedText()
    {
        var (formatter, captured) = MakeFormatter();

        formatter.OnPartialUpdate(MakeResult("still speaking"));

        Assert.Single(captured);
        Assert.Equal(string.Empty, captured[0].FinalizedText);
    }

    [Fact]
    public void OnFinalization_AppendsFinalizedText()
    {
        var (formatter, captured) = MakeFormatter();

        formatter.OnFinalization(MakeResult("hello", 0.0, 1.0));
        formatter.OnFinalization(MakeResult("world", 1.5, 2.5));

        Assert.Equal(2, captured.Count);
        Assert.Equal(DisplayAction.Finalize, captured[1].Action);
        Assert.Equal("hello world", captured[1].FinalizedText);
        Assert.Empty(captured[1].PreliminarySegments);
    }

    [Fact]
    public void OnFinalization_DuplicateText_IsIgnored()
    {
        var (formatter, captured) = MakeFormatter();

        formatter.OnFinalization(MakeResult("hello", 0.0, 1.0));
        formatter.OnFinalization(MakeResult("hello", 1.0, 2.0));

        Assert.Single(captured);
    }

    [Fact]
    public void OnFinalization_GapAboveThreshold_InsertsParagraphBreak()
    {
        var (formatter, captured) = MakeFormatter();

        formatter.OnFinalization(MakeResult("first sentence", 0.0, 1.0));
        // partial arrives with start_time well after the finalized end_time (gap >= 2.0s)
        formatter.OnPartialUpdate(MakeResult("new paragraph", 3.5, 4.0));

        var lastInstructions = captured[^1];
        Assert.Equal(DisplayAction.RerenderAll, lastInstructions.Action);
        Assert.EndsWith("\n", lastInstructions.FinalizedText);
    }

    [Fact]
    public void OnFinalization_GapBelowThreshold_NoParagraphBreak()
    {
        var (formatter, captured) = MakeFormatter();

        formatter.OnFinalization(MakeResult("first sentence", 0.0, 1.0));
        // gap of 1.0s is below the 2.0s threshold
        formatter.OnPartialUpdate(MakeResult("continuing", 2.0, 2.5));

        var lastInstructions = captured[^1];
        Assert.Equal(DisplayAction.RerenderAll, lastInstructions.Action);
        Assert.False(lastInstructions.FinalizedText.EndsWith("\n"),
            "FinalizedText should not end with newline when gap is below threshold");
    }

    [Fact]
    public void OnFinalization_FirstResult_EmptyFinalizedText_NoParagraphBreak()
    {
        var (formatter, captured) = MakeFormatter();

        // No previous finalization — _lastFinalizedEndTime is 0
        formatter.OnPartialUpdate(MakeResult("opening words", 5.0, 6.0));

        var instructions = captured[0];
        Assert.Equal(DisplayAction.RerenderAll, instructions.Action);
        Assert.Equal(string.Empty, instructions.FinalizedText);
        Assert.False(instructions.FinalizedText.EndsWith("\n"),
            "No paragraph break should be inserted when there is no prior finalized text");
    }
}
