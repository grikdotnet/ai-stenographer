namespace SttClient.Recognition;

/// <summary>
/// Immutable record representing one recognition result from the STT server.
/// </summary>
public sealed record RecognitionResult(
    string Text,
    double StartTime,
    double EndTime,
    int? UtteranceId,
    int[] ChunkIds,
    double[]? TokenConfidences
);
