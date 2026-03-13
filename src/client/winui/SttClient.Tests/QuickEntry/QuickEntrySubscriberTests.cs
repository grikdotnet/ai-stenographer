using Microsoft.Extensions.Logging.Abstractions;
using SttClient.QuickEntry;
using SttClient.Recognition;
using Xunit;

namespace SttClient.Tests.QuickEntry;

/// <summary>
/// Tests for <see cref="QuickEntrySubscriber"/> — verifies accumulation of partial+final
/// text and activation/deactivation gating.
/// </summary>
public class QuickEntrySubscriberTests
{
    private static RecognitionResult MakeResult(string text) =>
        new(text, 0.0, 1.0, null, [], null);

    private static QuickEntrySubscriber MakeSubscriber(Action<string>? onTextChange = null)
    {
        return new QuickEntrySubscriber(onTextChange ?? (_ => { }), NullLogger<QuickEntrySubscriber>.Instance);
    }

    [Fact]
    public void WhenInactive_PartialUpdate_IsIgnored()
    {
        var received = new List<string>();
        var sub = MakeSubscriber(t => received.Add(t));

        sub.OnPartialUpdate(MakeResult("hello"));

        Assert.Empty(received);
    }

    [Fact]
    public void WhenInactive_Finalization_IsIgnored()
    {
        var received = new List<string>();
        var sub = MakeSubscriber(t => received.Add(t));

        sub.OnFinalization(MakeResult("hello"));

        Assert.Empty(received);
    }

    [Fact]
    public void Activate_ClearsAccumulatedText()
    {
        var sub = MakeSubscriber();
        sub.Activate();
        sub.OnFinalization(MakeResult("old text"));
        sub.Deactivate();

        sub.Activate();

        Assert.Equal(string.Empty, sub.GetAccumulatedText());
    }

    [Fact]
    public void PartialUpdate_WhenActive_FiresCallback()
    {
        var received = new List<string>();
        var sub = MakeSubscriber(t => received.Add(t));
        sub.Activate();

        sub.OnPartialUpdate(MakeResult("speaking"));

        Assert.Single(received);
    }

    [Fact]
    public void PartialUpdate_WhenActive_CallbackIncludesPartialText()
    {
        string? lastText = null;
        var sub = MakeSubscriber(t => lastText = t);
        sub.Activate();

        sub.OnPartialUpdate(MakeResult("partial here"));

        Assert.Equal("partial here", lastText);
    }

    [Fact]
    public void Finalization_WhenActive_AccumulatesText()
    {
        var sub = MakeSubscriber();
        sub.Activate();
        sub.OnFinalization(MakeResult("one"));
        sub.OnFinalization(MakeResult("two"));

        Assert.Equal("one two", sub.GetAccumulatedText());
    }

    [Fact]
    public void Finalization_WhenActive_ClearsPartialInCallback()
    {
        string? lastText = null;
        var sub = MakeSubscriber(t => lastText = t);
        sub.Activate();
        sub.OnPartialUpdate(MakeResult("in flight"));
        sub.OnFinalization(MakeResult("done"));

        Assert.Equal("done", lastText);
    }

    [Fact]
    public void Finalization_AndPartial_CombinedDisplayText()
    {
        string? lastText = null;
        var sub = MakeSubscriber(t => lastText = t);
        sub.Activate();
        sub.OnFinalization(MakeResult("first"));
        sub.OnPartialUpdate(MakeResult("second"));

        Assert.Equal("first second", lastText);
    }

    [Fact]
    public void Deactivate_StopsAccumulation()
    {
        var sub = MakeSubscriber();
        sub.Activate();
        sub.OnFinalization(MakeResult("kept"));
        sub.Deactivate();

        sub.OnFinalization(MakeResult("ignored"));

        Assert.Equal("kept", sub.GetAccumulatedText());
    }
}
