namespace SttClient.Recognition;

/// <summary>
/// Receives recognition results from the remote STT server.
/// Implementations are called synchronously on the ReceiveLoop task thread.
/// </summary>
public interface IRecognitionSubscriber
{
    /// <summary>Called when a partial (in-progress) recognition result arrives.</summary>
    void OnPartialUpdate(RecognitionResult result);

    /// <summary>Called when a final (committed) recognition result arrives.</summary>
    void OnFinalization(RecognitionResult result);
}
