Web search results for query: "NeMo Parakeet STT model hallucination "yeah" short audio segments 2024"

Links: [{"title":"Streaming audio transcription using Parakeet | Modal Docs","url":"https://modal.com/docs/examples/streaming_parakeet"}]

I found information about the NeMo Parakeet STT model and the hallucination issue you're asking about. 

Parakeet is the name of a family of ASR models built using NVIDIA's NeMo Framework. The documentation reveals a known issue with the model: it tends to output utterances like "Yeah" or "Mm-hmm" when it runs on silent audio.

To address this hallucination problem, pre-processing the incoming audio using pydub's silence detection ensures that silence isn't passed into the model. This is a practical workaround implemented in streaming applications to prevent the model from generating false transcriptions during silent segments.

This appears to be a characteristic behavior of the Parakeet model when processing short audio segments that contain silence, and developers working with the model need to implement silence detection as a pre-processing step to avoid these hallucinations.