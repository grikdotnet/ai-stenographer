namespace SttClient.Formatting;

/// <summary>Discriminates the rendering action a <see cref="DisplayInstructions"/> record requests.</summary>
public enum DisplayAction
{
    /// <summary>Redraw the entire text area from scratch using current state.</summary>
    RerenderAll,

    /// <summary>Commit the latest finalization result; clear preliminary overlay.</summary>
    Finalize
}

/// <summary>
/// Immutable value object carrying instructions for the text display layer.
/// Produced by <see cref="TextFormatter"/> and consumed by the view.
/// </summary>
/// <param name="Action">The rendering action the view must perform.</param>
/// <param name="FinalizedText">The accumulated committed transcript, possibly containing paragraph breaks.</param>
/// <param name="PreliminarySegments">In-flight partial result texts shown as a transient overlay.</param>
public sealed record DisplayInstructions(
    DisplayAction Action,
    string FinalizedText,
    IReadOnlyList<string> PreliminarySegments
);
